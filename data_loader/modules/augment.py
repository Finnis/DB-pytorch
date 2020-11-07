import math
import random
import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """Add noise to image
        :param data: {'img':, 'text_polys':, 'texts':, 'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im_type = data['img'].dtype
        data['img'] = (random_noise(data['img'], mode='gaussian', clip=True) * 255).astype(im_type)

        return data


class RandomScale:
    def __init__(self, scales: list, random_rate: float):
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        """
        if random.random() > self.random_rate:
            return data
        rd_scale = float(np.random.choice(self.scales))
        data['img'] = cv2.resize(data['img'], dsize=None, fx=rd_scale, fy=rd_scale)
        data['text_polys'] *= rd_scale

        return data


class RandomRotateImgBox:
    def __init__(self, degrees, random_rate, keep_size=False):
        """
        :param degrees: float or [float]
        :param ramdon_rate: float number
        :param keep_size: Wether the size of result is same with origin image
        :return:
        """
        if isinstance(degrees, (float, int)):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, (list, tuple)):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.keep_size = keep_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        # ---------------------- Rotate image ----------------------
        h, w = im.shape[:2]
        angle = np.random.uniform(*self.degrees)

        if self.keep_size:
            nw, nh = w, h
        else:
            rangle = np.deg2rad(angle)  # 角度变弧度
            # 计算旋转之后图像的w, h
            nw = abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)
            nh = abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_text_polys = []
        for bbox in text_polys:
            point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
            point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
            point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
            point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
            rot_text_polys.append([point1, point2, point3, point4])
        data['img'] = rot_img
        data['text_polys'] = np.array(rot_text_polys)

        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        size: [h, w]
        ramdon_rate
        keep_ratio
        """
        if isinstance(size, int):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, (list, tuple)):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        data: {'img':, 'text_polys':, 'texts':, 'ignore_tags':}
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        if self.keep_ratio:
            # pad short side of image
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=im.dtype)
            im_padded[:h, :w, :] = im.copy()
            im = im_padded

        h, w, _ = im.shape
        data['img'] = cv2.resize(im, (self.size[1], self.size[0]))
        w_scale = self.size[1] / w
        h_scale = self.size[0] / h
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale
        data['text_polys'] = text_polys

        return data


class ResizeImage(object):
    """Resize image to a size multiple of 32 which is required by the network
    """
    def __init__(self, min_len=512, max_len=1280, resize_text_polys=False):
        self.min_len = min_len
        self.max_len = max_len
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict):
        im = data['img']
        h, w, _ = im.shape
        short_size, long_size = np.sort([h, w])
        if short_size >= self.min_len and long_size <= self.max_len:
            resize_h, resize_w = h, w
        elif short_size < self.min_len and long_size > self.max_len:
            if h < w:
                resize_h, resize_w = self.min_len, self.max_len
            else:
                resize_h, resize_w = self.max_len, self.min_len
        elif short_size < self.min_len and long_size <= self.max_len:
            if h < w:
                resize_h = self.min_len
                resize_w = min(int(resize_h / h * w), self.max_len)
            else:
                resize_w = self.min_len
                resize_h = min(int(resize_w / w * h), self.max_len)
        elif short_size >= self.min_len and long_size > self.max_len:
            if h < w:
                resize_w = self.max_len
                resize_h = max(int(resize_w / w * h), self.min_len)
            else:
                resize_h = self.max_len
                resize_w = max(int(resize_h / h * w), self.min_len)
        else:
            raise NotImplementedError('resize image failed.')

        resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
        resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
        data['img'] = cv2.resize(im, (resize_w, resize_h))

        if self.resize_text_polys:
            data['text_polys'][:, 0] *= resize_w / w
            data['text_polys'][:, 1] *= resize_h / h
        
        return data


class HorizontalFlip:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        w = im.shape[1]

        flip_text_polys = data['text_polys'].copy()
        data['img'] = cv2.flip(im, 1)
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]
        data['text_polys'] = flip_text_polys

        return data


class VerticallFlip:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        data: {'img':, 'text_polys':, 'texts':, 'ignore_tags':}
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        h = im.shape[0]

        flip_text_polys = data['text_polys'].copy()
        data['img'] = cv2.flip(im, 0)
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data['text_polys'] = flip_text_polys

        return data
