import yaml
import os
import torch
from tqdm import tqdm
import argparse

from nets import DBModel
from data_loader import get_dataloader
from utils import SegDetectorRepresenter, logger, QuadMetric


class Eval:
    def __init__(self, config_file):
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        config.update(yaml.load(open(config['base']), Loader=yaml.FullLoader))
        
        torch.backends.cudnn.benchmark = config['eval']['benchmark']

        config['arch']['backbone']['pretrained'] = False
        self.model = DBModel(config)
        if os.path.isfile(config['eval']['model_path']):
            model_path = config['eval']['model_path']
        else:
            model_path = config['eval']['model_path'] + config['arch']['backbone']['type'] + '/checkpoints'
            model_path = os.path.join(model_path, sorted(os.listdir(model_path))[-1])
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(ckpt['model'])
        self.model.cuda()

        self.post_process = eval(config['post_process']['type'])(**config['post_process']['args'])
        self.metric_cls = eval(config['metric']['type'])(**config['metric']['args'])

        self.eval_data_loader, _ = get_dataloader(config['eval']['dataset'], distributed=False)

        with open('outputs/debug_configs.yaml', 'w') as f:
            yaml.dump(config, f, indent=1)

    def val(self):
        self.model.eval()
        raw_metrics = []
        with torch.no_grad():
            for sample in tqdm(self.eval_data_loader):
                sample['img'] = sample['img'].cuda()
                preds = self.model(sample.pop('img'))
                probs = preds.cpu().squeeze(1).numpy()  # (N,H,W)
                boxes, scores = self.post_process(sample['shape'], probs, self.metric_cls.is_output_polygon)
                raw_metric = self.metric_cls.validate_measure(sample, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        
        logger.info(f"recall:{metrics['recall'].avg:.6f}, "
                    f"precision:{metrics['precision'].avg:.6f}, "
                    f"fmeasure:{metrics['fmeasure'].avg:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBNet')
    parser.add_argument('-c', '--config_file', default='configs/resnet18_eval_pred.yaml', type=str)
    args = parser.parse_args()

    Eval(args.config_file).val()