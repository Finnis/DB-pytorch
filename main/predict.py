import yaml
from tqdm import tqdm
import argparse
import os
import torch
import numpy as np
import cv2
from torchvision import transforms as T

from nets import DBModel
from utils import SegDetectorRepresenter, logger, prepare_dir
from data_loader.modules import ResizeImage


class Prediction:
    def __init__(self, arg):
        config = yaml.load(open(arg.config_file), Loader=yaml.FullLoader)
        torch.backends.cudnn.benchmark = config['eval']['benchmark']

        self.model = DBModel(config)
        if os.path.isfile(config['eval']['model_path']):
            model_path = config['eval']['model_path']
        else:
            model_path = config['eval']['model_path'] + config['arch']['backbone']['type'] + '/checkpoints'
            model_path = os.path.join(model_path, sorted(os.listdir(model_path))[-1])
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt['model'])
        logger.info(f'Model loaded from {model_path}')
        self.model.cuda()
        self.model.eval()

        self.post_process = eval(config['post_process']['type'])(**config['post_process']['args'])

        self.resize_image = ResizeImage(min_len=arg.min_len, max_len=arg.max_len)

        self.transforms = []
        for l in config['eval']['dataset']['args']['transforms']:
            self.transforms.append(getattr(T, l['type'])(**l['args']))
        self.transforms = T.Compose(self.transforms)

        self.is_output_polygon = arg.is_output_polygon
        self.save_poly_path = arg.save_poly_path
        self.save_poly = arg.save_poly
        self.prob_map_path = arg.prob_map_path
        self.save_prob_map = arg.save_prob_map
        self.show_poly_path = arg.show_poly_path
        self.show_poly = arg.show_poly
        prepare_dir([self.save_poly_path, self.prob_map_path, self.show_poly_path])
    
    def predict(self, imgs_path):
        imgs_list = sorted(os.listdir(imgs_path))
        logger.info(f'Find {len(imgs_list)} images in {imgs_path}')
        for name in tqdm(imgs_list):
            self._predict(os.path.join(imgs_path, name))

    def _predict(self, img_path):
        img_name = os.path.basename(img_path)
        with torch.no_grad():
            im = cv2.imread(img_path)
            shape = im.shape[:2]  # 1, H, W
            img = im[:, :, ::-1].copy()
            img = self.resize_image({'img': img})['img']
            img = self.transforms(img)
            img = img.unsqueeze(0).cuda()
            preds = self.model(img)
            probs = preds.cpu().squeeze(1).numpy()  # (N,H,W) [0,1]
        boxes_batch, scores_batch = self.post_process([shape], probs, self.is_output_polygon)
        boxes, scores = boxes_batch[0], scores_batch[0]

        if self.save_poly:
            self._save_poly(img_name, boxes, scores)
        if self.save_prob_map:
            self._save_prob_map(img_name, probs[0])
        if self.show_poly:
            self._show_poly(im, img_name, boxes)
    
    def _save_poly(self, img_name, box_list, score_list):
        #if self.is_output_polygon:
        with open(os.path.join(self.save_poly_path, img_name[:-3]+'txt'), 'wt') as res:
            for box, score in zip(box_list, score_list):
                result = ",".join([str(int(x)) for x in box.reshape(-1)])
                res.write(result + ',' + str(score) + "\n")
    
    def _save_prob_map(self, img_name, prob):
        out_path = os.path.join(self.prob_map_path, img_name)
        cv2.imwrite(out_path, prob * 255)

    def _show_poly(self, img, img_name, result):
        if len(result) > 0: 
            cv2.polylines(img, result, True, (0, 255, 0), 2)      
        cv2.imwrite(os.path.join(self.show_poly_path, img_name), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBNet')
    parser.add_argument('-c', '--config_file', default='configs/resnet18_eval_pred.yaml')
    parser.add_argument('--save_poly_path', default='results/poly_txt')
    parser.add_argument('--prob_map_path', default='results/prob_map')
    parser.add_argument('--show_poly_path', default='results/poly_vis')

    parser.add_argument('--min_len', default=512, type=int)
    parser.add_argument('--max_len', default=1280, type=int)

    parser.add_argument('--is_output_polygon', action='store_true')
    parser.add_argument('--save_poly', action='store_true')    
    parser.add_argument('--save_prob_map', action='store_true')    
    parser.add_argument('--show_poly', action='store_true')
    
    parser.add_argument('--image_path')

    args = parser.parse_args()
    assert args.image_path is not None

    pred = Prediction(args)
    if os.path.isfile(args.image_path):
        pred._predict(args.image_path)
    else:
        pred.predict(args.image_path)
