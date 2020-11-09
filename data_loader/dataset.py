import os
import cv2
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import order_points_clockwise, logger
from . import modules


class BaseDataset(Dataset):
    def __init__(self, data_path, pre_processes, filter_keys, ignore_tags, transforms=None):
        '''
        pre_processes: [{type, args: {}}, ...]
        '''        
        self.ignore_tags = ignore_tags
        data_path = [x for x in data_path if x is not None]
        assert len(data_path) > 0, f'Check data in {data_path}'
        self.data_list = self.load_data(data_path)
        item_keys = ['img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags']
        for item in item_keys:
            assert item in self.data_list[0], f'data_list from load_data must contains {item_keys}'
        self.filter_keys = filter_keys

        if len(transforms):
            self.transform = T.Compose([getattr(T, item['type'])(**item['args']) for item in transforms])
        else:
            self.transform = None
        
        self.augs = []
        for item in pre_processes:
            if item['type'] == 'IaaAugment':
                self.augs.append(getattr(modules, item['type'])(item['args']))
            else:
                self.augs.append(getattr(modules, item['type'])(**item['args']))

    def apply_pre_processes(self, data):
        for aug in self.augs:
            data = aug(data)
        return data

    def __getitem__(self, index):
        data = self.data_list[index].copy()
        im = cv2.imread(data['img_path'])[:, :, ::-1]
        data['img'] = im
        data['shape'] = im.shape[:2]
        data = self.apply_pre_processes(data)
        if self.transform:
            data['img'] = self.transform(data['img'])        
        data['text_polys'] = data['text_polys'].tolist()
        for k in self.filter_keys:
            data.pop(k, None)

        return data

    def __len__(self):
        return len(self.data_list)


class ContainerDataset(BaseDataset):
    def __init__(self, data_path, pre_processes, filter_keys, ignore_tags, transforms=None):
        super().__init__(data_path, pre_processes, filter_keys, ignore_tags, transforms)

    def load_data(self, data_path:list) -> list:
        img_path_list, label_path_list = [], []
        for path in data_path:
            root_path = os.path.dirname(path)
            phase = os.path.basename(path)[:-4]

            with open(path, 'r') as f:
                image_list = f.readlines()
            for l in image_list:
                img_path_list.append(f'{root_path}/{phase}_images/{l.strip()}')
                label_path_list.append(f"{root_path}/{phase}_gts/{l.strip()}.txt")
        
        t_data_list = []
        for img_path, label_path in zip(img_path_list, label_path_list):
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                data.update({'img_path': img_path, 'img_name': os.path.basename(img_path)})
                t_data_list.append(data)
            else:
                logger.warning(f'No suit bbox in {label_path}')
        logger.info(f'Total {len(t_data_list)} images for Container dataset.')
        
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes, texts, ignores = [], [], []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except:
                    logger.error(f'load label failed on {label_path}')
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data


class ICDAR2015Dataset(BaseDataset):
    def __init__(self, data_path, pre_processes, filter_keys, ignore_tags, transforms=None):
        super().__init__(data_path, pre_processes, filter_keys, ignore_tags, transforms)

    def load_data(self, data_path:list) -> list:
        img_path_list, label_path_list = [], []
        for path in data_path:
            root_path = os.path.dirname(path)
            phase = os.path.basename(path)[:-4]

            with open(path, 'r') as f:
                image_list = f.readlines()
            for l in image_list:
                img_path_list.append(f'{root_path}/{phase}_images/{l.strip()}')
                label_path_list.append(f"{root_path}/{phase}_gts/gt_{l.strip().replace('jpg', 'txt')}")
        
        t_data_list = []
        for img_path, label_path in zip(img_path_list, label_path_list):
            data = self._get_annotation(label_path)
            if len(data['text_polys']) > 0:
                data.update({'img_path': img_path, 'img_name': os.path.basename(img_path)})
                t_data_list.append(data)
            else:
                logger.warning(f'No suit bbox in {label_path}')
        logger.info(f'Total {len(t_data_list)} images for ICDAR2015 dataset.')
        
        return t_data_list

    def _get_annotation(self, label_path: str) -> dict:
        boxes, texts, ignores = [], [], []
        with open(label_path, encoding='utf-8', mode='r') as f:
            for line in f.readlines():
                params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
                try:
                    box = order_points_clockwise(np.array(list(map(float, params[:8]))).reshape(-1, 2))
                    if cv2.contourArea(box) > 0:
                        boxes.append(box)
                        label = params[8]
                        texts.append(label)
                        ignores.append(label in self.ignore_tags)
                except:
                    logger.error(f'load label failed on {label_path}')
        data = {
            'text_polys': np.array(boxes),
            'texts': texts,
            'ignore_tags': ignores,
        }
        return data


class SynthTextDataset(BaseDataset):
    def __init__(self, data_path, pre_processes, filter_keys, ignore_tags, transforms=None):
        super().__init__(data_path, pre_processes, filter_keys, ignore_tags, transforms)

    def load_data(self, data_path: str) -> list:
        data_path = data_path[0]
        t_data_list = []
        for imageName, wordBBoxes, texts in zip(*self._get_annotations(data_path)):
            item = {}
            wordBBoxes = np.expand_dims(wordBBoxes, axis=2) if (wordBBoxes.ndim == 2) else wordBBoxes
            _, _, numOfWords = wordBBoxes.shape
            text_polys = wordBBoxes.reshape([8, numOfWords], order='F').T  # num_words * 8
            text_polys = text_polys.reshape(numOfWords, 4, 2)  # num_of_words * 4 * 2
            transcripts = [word for line in texts for word in line.split()]
            if numOfWords != len(transcripts):
                continue
            if not self._validate_polys(text_polys):
                continue
            item['img_path'] = os.path.join(data_path, imageName)
            item['img_name'] = os.path.basename(imageName)
            item['text_polys'] = text_polys
            item['texts'] = transcripts
            item['ignore_tags'] = [x in self.ignore_tags for x in transcripts]
            t_data_list.append(item)
        logger.info(f'Total {len(t_data_list)} images for SynthText dataset.')

        return t_data_list
    
    @staticmethod
    def _validate_polys(text_polys):
        for poly in text_polys:
            if cv2.contourArea(poly) < 10:
                return False
        return True

    def _get_annotations(self, data_path):
        gt_path = os.path.join(data_path, 'gt.mat')
        targets = sio.loadmat(gt_path, squeeze_me=True, struct_as_record=False,
                              variable_names=['imnames', 'wordBB', 'txt'])
        
        return targets['imnames'], targets['wordBB'], targets['txt']