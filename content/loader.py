import json
from typing import Dict, List, Tuple, Union, Iterator, Iterable, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod
from easydict import EasyDict
import numpy as np
import cv2
import pickle as pkl

class Data(ABC):
    data_type: str
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._data = None
        
    @property
    def data(self):
        if self._data is None:
            self._data = self._get()
        return self._data
    
    @abstractmethod
    def _get(self): pass
    
class GPS(Data):
    data_type = "GPS"
    
    def __init__(self, file_path: Path):
        super().__init__(file_path=file_path)
    
    def _get(self) -> EasyDict:
        with self.file_path.open('rb') as f:
            _data = EasyDict(pkl.load(f))
        return _data
    
    @property
    def timestamp(self) -> int:
        return self.data.timestamp
    
    @property
    def inspvas(self) -> Dict[str, Any]:
        return self.data.inspvas
    
    @property
    def corrimus(self) -> List:
        return self.data.corrimus
    
    @property
    def insstdevs(self) -> List:
        return self.data.insstdevs
    
    @property
    def insspds(self) -> List:
        return self.data.insspds
    
    
class Camera(Data):
    data_type = "Camera"
    
    def __init__(self, file_path: Path, resize: Optional[Tuple[int, int]] = None):
        self.resize = resize
        super().__init__(file_path=file_path)
    
    def _get(self) -> np.ndarray:
        img = cv2.imread(str(self.file_path))
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
        if self.resize is not None:
            img = cv2.resize(img, self.resize)
        return img
    
class Lidar(Data):
    data_type = "Lidar"
    
    def __init__(self, file_path: Path):
        super().__init__(file_path=file_path)
    
    def _get(self) -> EasyDict:
        with self.file_path.open('rb') as f:
            _data = EasyDict(pkl.load(f))
        return _data
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.data.metadata
    
    @property
    def frame(self) -> np.ndarray:
        return self.data.frame
    
class BoundingBox:
    def __init__(self, coords: List[float], class_name: str, class_id: int, additional_data: Dict[str,Any]):
        self.coords = coords
        self.x = coords[0]
        self.y = coords[1]
        self.w = coords[2]
        self.h = coords[3]
        self.class_name = class_name
        self.class_id = class_id
        self.additional_data = EasyDict(additional_data)
        
    
class Labels(Data):
    data_type = "Labels"
    CLASSES_VAL = {
        "Pedestrian": 0.0,
        "Pushing_stroller": 0.33,
        "In_a_wheelchair": 0.67,
        "Cyclist": 1.0,

        "Tram/train": 0.0,
        "Slow-Moving_vehicle": 0.5,
        "Slow-Moving_vehicle_on_trailer": 1.0,

        "Moving": 0.0,
        "Parked": 0.5,
        "On trailer": 1.0,

        "Special situation@@NL": -1.0,
        "Special situation@@Normal road": 0.0,
        "Special situation@@Bridge/overpass": 0.25,
        "Special situation@@Overpass ahead": 0.5,
        "Special situation@@Tunnel ahead": 0.75,
        "Special situation@@In a tunnel": 1.0,
    }
    situation_names = {
        -1.0: "NL",
        0.0: "Normal road",
        0.25: "Bridge/overpass",
        0.5: "Overpass ahead",
        0.75: "Tunnel ahead",
        1.0: "In a tunnel",
    }
        
    def __init__(self, _labels: Dict[str,Any]):
        self._labels = _labels
        self._classname2id = {
             'Persons': 0,
             'Motorcycle': 1,
             'Car': 2,
             'Truck': 3,
             'Bus': 4,
             'Special vehicles': 5,
             'Sign': 6,
         }
        super().__init__(file_path=None)
        
    @property
    def bboxs(self):
        return self.data.bboxs
    
    @property
    def classes(self):
        return self.data.classes
        
    def _get(self) -> Union[List[BoundingBox], Tuple[str, int]]:
        if self._labels['annotations'] == 0:
            return EasyDict({'bboxs':{}, 'classes':{}})
        latest_labels = self._labels['annotations'][-1]['result']
        bbs = []
        classes = []
        for ann in latest_labels:
            if ann['type'] == 'rectanglelabels':
                bbs.append(self._get_bb(ann))
            elif ann['type'] == 'choices':
                classes.append(self._get_classes(ann))
            else:
                raise ValueError('Unknown annotation type: {}'.format(ann['type']))
        uncombined_classes = self._combine_classes(bbs, classes)
        classification = {}
        for key, value in uncombined_classes.items():
            base = key.split('###')[0]
            name = key.split('###')[1]
            if name == "Special situation":
                classification["Special situation"] = (value, self.situation_names.get(value))
        return EasyDict({'bboxs':bbs, 'classes':classification})

    def _combine_classes(self, bbs: List[BoundingBox], classes: List[Dict[str, int]]):
        ret_cls = {}
        for kv in classes:
            unused = True
            k, v = next(iter(kv.items()))
            try:
                base = k.split('###')[0]
                name = k.split('###')[1]
                for bb in bbs:
                    if base == bb.additional_data['id']:
                        bb.additional_data[name] = v
                        unused = False
                if unused:
                    ret_cls[k] = v
            except Exception:
                ret_cls[k] = v
        return ret_cls

    def _get_bb(self, ann: dict):
        l, t, w, h = ann['value']['x'], ann['value']['y'], ann['value']['width'], ann['value']['height']
        x = l / 100. * ann['original_width']
        y = t / 100. * ann['original_height']
        # x = (l + w/2.) / 100. * ann['original_width']
        # y = (t + h/2.) / 100. * ann['original_height']
        real_w = w / 100. * ann['original_width']
        real_h = h / 100. * ann['original_height']
        try:
            class_name = ann['value']['rectanglelabels'][0]
        except:
            print(ann)
        return BoundingBox(
            coords=[x, y, real_w, real_h],
            class_name=class_name,
            class_id=self.get_class_id(class_name),
            additional_data={'id': ann['id']},
        )

    def _get_classes(self, ann: dict):
        key = ann['from_name']
        value = ann['value']['choices'][0]
        return {f"{ann['id']}###{key}": self.CLASSES_VAL[value]}
    
    def get_class_id(self, class_name: str) -> int:
        cid = self._classname2id.get(class_name, None)
        return cid

    
class Sample:
    def __init__(self, sample_id: str, labels: Labels=None, camera: Camera=None, lidar: Lidar=None, gps: GPS=None):
        self.sample_id = sample_id
        self.labels = labels
        self.camera = camera
        self.lidar = lidar
        self.gps = gps
        
    def available_sensors():
        return {
            "Camera": self.camera is not None,
            "Lidar": self.lidar is not None,
            "GPS": self.GPS is not None
        }
    
    
def remove_prefix(text:str, prefix:str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
    

class AGHDriveLoader:
    
    def __init__(self, config: Union[str, Path, Dict]):
        self.config = self._get_config(config)
        self._validate_paths_exist()
        self._labels = self._read_annotations_file(annotations_file_path = self.config.annotations_path)
        self._validate_files()
        
    def _get_config(self, config: Union[str, Path, dict]) -> EasyDict:
        if type(config) == dict:
            config_ed = EasyDict(config)
        else:
            config_path = Path(str(config))
            with config_path.open('r') as f:
                config_ed = json.load(f)
        config_ed.data_root_path = Path(str(config_ed.data_root_path))
        if not str(config_ed.camera_path).startswith(str(config_ed.data_root_path)):
            config_ed.camera_path = config_ed.data_root_path.joinpath(config_ed.camera_path)
        else:
            config_ed.camera_path = Path(config_ed.camera_path)
        
        if not str(config_ed.lidar_path).startswith(str(config_ed.data_root_path)):
            config_ed.lidar_path = config_ed.data_root_path.joinpath(config_ed.lidar_path)
        else:
            config_ed.lidar_path = Path(config_ed.lidar_path)

        if not str(config_ed.gps_path).startswith(str(config_ed.data_root_path)):
            config_ed.gps_path = config_ed.data_root_path.joinpath(config_ed.gps_path)
        else:
            config_ed.gps_path = Path(config_ed.gps_path)

        if not str(config_ed.annotations_path).startswith(str(config_ed.data_root_path)):
            config_ed.annotations_path = config_ed.data_root_path.joinpath(config_ed.annotations_path)
        else:
            config_ed.annotations_path = Path(config_ed.annotations_path)
            
        return config_ed
            
        
    def _validate_paths_exist(self) -> None:
        assert self.config.data_root_path.exists(), f'{self.config.data_root_path} does not exists'
        assert self.config.annotations_path.exists(), f'{self.config.annotations_path} does not exists'
        if self.config.read_camera and not self.config.camera_path.exists():
            print(f'{self.config.camera_path} does not exists')
        if self.config.read_lidar and not self.config.lidar_path.exists():
            print(f'{self.config.lidar_path} does not exists')
        if self.config.read_gps and not self.config.gps_path.exists():
            print(f'{self.config.gps_path} does not exists')
    
    def _validate_files(self):
        not_found_data_keys = []
        for sample_id in sorted(self._labels.keys()):
            camera_file_path = self.config.camera_path.joinpath(f'{sample_id}.png')
            lidar_file_path = self.config.lidar_path.joinpath(f'{sample_id}.pkl')
            gps_file_path = self.config.gps_path.joinpath(f'{sample_id}.pkl')
            if (self.config.read_camera and not camera_file_path.exists()) \
                or (self.config.read_lidar and not lidar_file_path.exists()) \
                or (self.config.read_gps and not gps_file_path.exists()):
                not_found_data_keys.append(sample_id)
        if not_found_data_keys:
            print(f'WARNING: full data not found for {len(not_found_data_keys)} labels.')
            print(f'\tRemoving those labels from set and continuing.')
            for sample_id in not_found_data_keys:
                self._labels.pop(sample_id)

    @staticmethod
    def _read_annotations_file(annotations_file_path: Path) -> Dict[str,Dict]:
        with annotations_file_path.open('r') as f:
            json_data = json.load(f)
        data = {key[9:].split('.')[0]: value for key, value in json_data.items()}
        return data
    
    def generator(self) -> Tuple[Union[Iterator, Iterable], int]:
        def data_iter():
            for sample_id in sorted(self._labels.keys()):
                camera = Camera(file_path=self.config.camera_path.joinpath(f'{sample_id}.png')) if self.config.read_camera else None
                lidar = Lidar(file_path=self.config.lidar_path.joinpath(f'{sample_id}.pkl')) if self.config.read_lidar else None
                gps = GPS(file_path=self.config.gps_path.joinpath(f'{sample_id}.pkl')) if self.config.read_gps else None
                labels = Labels(self._labels[sample_id])
                
                yield Sample(
                    sample_id=sample_id,
                    labels=labels,
                    camera=camera,
                    lidar=lidar,
                    gps=gps
                )
        
        return iter(data_iter()), len(self._labels)
    
    
def main():
    config = {
        "read_camera": True,
        "read_lidar": True,
        "read_gps": True,
        
        "data_root_path": "/home/gjz0zd/Projects/AGHdrive/aghdrive/",
        "camera_path": "data/images",
        "lidar_path": "data/lidar_pkl",
        "gps_path": "data/gps_pkl",
        "annotations_path": "annotations.json"
    }
    
    loader = AGHDriveLoader(config)
    dataset_generator, dataset_length = loader.generator()
    print(dataset_length)
    
    
if __name__ == "__main__":
    main()
