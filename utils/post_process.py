import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from .utils import order_points_clockwise


# For post_process
class SegDetectorRepresenter(object):
    def __init__(self, thresh=0.3, box_thresh=0.7, max_candidates=1000, unclip_ratio=1.5, min_size=3):
        self.min_size = min_size
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, shapes, preds, is_output_polygon=False):
        '''
        shape: numpy array, image size of type (N, h, w)
        pred:  prob/binary
            prob: text region segmentation map, with shape (N, H, W)
            binary: binarized with threshhold, (N, H, W)
        '''
        assert len(preds.shape) == 3
        bitmaps = (preds > self.thresh).astype(np.uint8)  # binarize
        boxes_batch = []
        scores_batch = []
        for pred, bitmap, (h, w) in zip(preds, bitmaps, shapes):
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred, bitmap, w, h)
            else:
                boxes, scores = self.boxes_from_bitmap(pred, bitmap, w, h)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def polygons_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        '''
        bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
        '''
        height, width = bitmap.shape
        boxes, scores = [], []
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points, unclip_ratio=self.unclip_ratio)
                if len(box) > 1:
                    continue
            else:
                continue

            #_, sside = self.get_mini_box(box.reshape((-1, 1, 2)))
            sside = min(cv2.minAreaRect(box.reshape((-1, 1, 2)))[1])
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, bitmap, dest_width, dest_height):
        '''
        bitmap: single map with shape (H, W), whose values are binarized to {0, 1}
        '''
        height, width = bitmap.shape
        contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes, scores = [], []

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, sside = self.get_mini_box(contour)
            if sside < self.min_size:
                continue

            score = self.box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points, unclip_ratio=self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_box(box)
            if sside < self.min_size + 2:
                continue

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            
            boxes.append(box)
            scores.append(score)

        return np.array(boxes, dtype=np.int32), np.array(scores, dtype=np.float32)

    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_box(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = order_points_clockwise(box)
        
        return box, min(rect[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
