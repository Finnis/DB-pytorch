import os
import yaml
import argparse
from tqdm import tqdm
import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from nets import DBModel
from loss import seg_detector_loss as Losses
from utils import prepare_dir, clean_ckpts, logger, WarmupPolyLR, TextScores,\
    SegDetectorRepresenter, QuadMetric, AverageMeter
from data_loader import get_dataloader


class Trainer:
    def __init__(self, config_file: str):
        self.cfg = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.cfg.update(yaml.load(open(self.cfg['base']), Loader=yaml.FullLoader))

        # Set random seed
        np.random.seed(self.cfg['trainer']['seed'])
        random.seed(self.cfg['trainer']['seed'])
        torch.manual_seed(self.cfg['trainer']['seed'])
        torch.cuda.manual_seed_all(self.cfg['trainer']['seed'])

        dist.init_process_group(backend='nccl', init_method='env://')
        self.local_rank = torch.distributed.get_rank()

        self.model = DBModel(self.cfg)
        self.model = self.model.cuda(self.local_rank)
        process_group = torch.distributed.new_group(list(range(torch.cuda.device_count())))
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)
        # broadcast_buffers==True if use sync bn
        self.model = DDP(self.model, device_ids=[self.local_rank],
                         output_device=self.local_rank, find_unused_parameters=True, broadcast_buffers=True)

        self.criterion = getattr(Losses, self.cfg['loss']['type'])(**self.cfg['loss']['args'])
        self.criterion = self.criterion.cuda(self.local_rank)

        self.optimizer = getattr(torch.optim, self.cfg['optimizer']['type'])\
            (self.model.parameters(), **self.cfg['optimizer']['args'])

        # Create learning_rate scheduler
        self.lr_scheduler = WarmupPolyLR(
            self.optimizer,
            target_lr=self.cfg['lr_scheduler']['args']['last_lr'],
            max_iters=self.cfg['trainer']['epochs'],
            warmup_iters=self.cfg['lr_scheduler']['args']['warmup_epoch'],
            last_epoch=-1
        )

        # Load model from previous checkpoint
        self.step, self.epoch = 0, 0
        self.ckpt_path = self.cfg['trainer']['outputs'] + self.cfg['arch']['backbone']['type'] + '/checkpoints'
        if self.cfg['trainer']['restore']:
            ckpt = os.path.join(self.ckpt_path, sorted(os.listdir(self.ckpt_path))[-1])
            ckpt = torch.load(ckpt)
            self.model.module.load_state_dict(ckpt['model'], strict=True)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.step = ckpt['step']
            self.epoch = ckpt['epoch'] + 1
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            if self.local_rank == 0:
                logger.info(f'Contatinue training from step {self.step} and epoch {self.epoch}.')
        elif self.local_rank == 0:
            prepare_dir(self.ckpt_path)
            logger.info(f"{self.ckpt_path} was cleaned.")
        
        if self.local_rank == 0:
            # Create train score calculator
            self.cal_score = TextScores(n_classes=2, thresh=self.cfg['post_process']['args']['thresh'])
            # Create valadate metric_cls
            self.metric_cls = eval(self.cfg['metric']['type'])(**self.cfg['metric']['args'])
            # Create post_process module
            self.post_process = eval(self.cfg['post_process']['type'])(**self.cfg['post_process']['args'])

            # Create tensorboard summary writer
            summary_dir = self.cfg['trainer']['outputs'] + self.cfg['arch']['backbone']['type']+ '/summary'
            if not self.cfg['trainer']['restore']:
                prepare_dir(summary_dir)
                logger.info(f"{summary_dir} was cleaned.")
            self.writer = SummaryWriter(summary_dir)

            #add graph
            if self.cfg['trainer']['tensorboard']:
                self.writer.add_graph(self.model.module, torch.zeros(1, 3, 640, 640).cuda(self.local_rank))
                torch.cuda.empty_cache()
   
        # Create Dataloader
        self.train_data_loader, self.sampler = get_dataloader(self.cfg['dataset']['train'], distributed=True)
        # Wether use validate or not
        self.val_enable = 'val' in self.cfg['dataset'] \
            and len(self.cfg['dataset']['val']['args']['data_path']) > 0
        if self.val_enable and self.local_rank == 0:
            self.cfg['dataset']['val']['loader']['pin_memory'] = True
            self.val_data_loader, _ = get_dataloader(self.cfg['dataset']['val'], distributed=False)

        if self.local_rank == 0:
            with open(self.cfg['trainer']['outputs'] + 'debug_configs.yaml', 'w') as f:
                yaml.dump(self.cfg, f, indent=1)
            
            self.train_loss = AverageMeter()

    def train(self):
        while self.epoch < self.cfg['trainer']['epochs']:
            self._train_step()
            if self.local_rank == 0 and self.val_enable:
                self._val_step()
            if self.local_rank == 0:
                self._save_network()
            
            self.epoch += 1

    def _train_step(self,):
        if self.local_rank == 0:
            logger.info('----------------------------------------------------------')
            logger.info(f'Training for epoch {self.epoch} ...')
            self.cal_score.reset()
        train_losses_list = []
        torch.backends.cudnn.benchmark = self.cfg['trainer']['use_benchmark']
        self.model.train()
        self.sampler.set_epoch(self.epoch)
        for sample in self.train_data_loader:
            self.step += 1
            sample_cuda = {}
            for k ,v in sample.items():
                if isinstance(v, torch.Tensor):  # text_polys and ignore_tags are list objects
                    sample_cuda[k] = v.cuda(self.local_rank)
            
            preds = self.model(sample_cuda.pop('img'))
            loss, losses_dict = self.criterion(preds, sample_cuda)

            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['trainer']['clip_grad_norm'])
            self.optimizer.step()

            if self.local_rank == 0:
                # Calculate and update scores
                pred_prob = preds['prob'].detach().cpu().squeeze(1)
                self.cal_score.update(pred_prob, sample['gt'], sample['mask'])
                self.train_loss.update(loss.item())
                train_losses_list.append(losses_dict)

                if self.step % self.cfg['trainer']['log_iter'] == 0:
                    lr = self.lr_scheduler.get_lr()[0]
                    scores = self.cal_score.scores
                    logger.info(f'Step:{self.step},loss:{self.train_loss.avg:.6f},lr:{lr:.7f},'
                                f"MeanAcc:{scores['Mean Acc']:.4f},MeanIoU:{scores['Mean IoU']:.4f}")
                    
                    if self.cfg['trainer']['tensorboard']:
                        self.writer.add_scalar('Train/LearningRate', lr, self.step)
                        self.writer.add_scalar('Train/MainLoss', self.train_loss.avg, self.step)
                        for k in train_losses_list[0]:
                            loss = torch.tensor([loss[k] for loss in train_losses_list]).mean()
                            self.writer.add_scalar(f'Train/{k}', loss, self.step)
                        self.writer.add_scalar('Train/MeanAcc', scores['Mean Acc'], self.step)
                        self.writer.add_scalar('Train/MeanIoU', scores['Mean IoU'], self.step)
                        self.writer.flush()
                
                    self.train_loss.reset()
                    self.cal_score.reset()  # Clear scores cache
        self.lr_scheduler.step()  # Update learning_rates in optimizer for each epoch
        torch.cuda.empty_cache()
    
    def _val_step(self,):
        logger.info('**********************************************************')
        logger.info(f'Validating for epoch {self.epoch} ...')
        torch.backends.cudnn.benchmark = False  # Since image shape is different
        self.model.eval()
        raw_metrics = []
        with torch.no_grad():
            for sample in tqdm(self.val_data_loader):
                sample['img'] = sample['img'].cuda(self.local_rank)
                preds = self.model(sample.pop('img'))
                probs = preds.cpu().squeeze(1).numpy()  # (N,H,W)
                boxes, scores = self.post_process(sample['shape'], probs, self.metric_cls.is_output_polygon)
                raw_metric = self.metric_cls.validate_measure(sample, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        
        logger.info(f"Epoch:{self.epoch},recall:{metrics['recall'].avg:.6f},"
                    f"precision:{metrics['precision'].avg:.6f},fmeasure:{metrics['fmeasure'].avg:.6f}")
        if self.cfg['trainer']['tensorboard']:
            self.writer.add_scalar('Val/recall', metrics['recall'].avg, self.step)
            self.writer.add_scalar('Val/precision', metrics['precision'].avg, self.step)
            self.writer.add_scalar('Val/fmeasure', metrics['fmeasure'].avg, self.step)
            self.writer.flush()
        torch.cuda.empty_cache()

    def _save_network(self):
        # Save model weights
        state = {
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(state, self.ckpt_path + f'/model_epoch_{self.epoch}_step_{self.step}.pth')
        clean_ckpts(self.ckpt_path, num_max=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DBNet')
    parser.add_argument('-c', '--config_file', default='configs/resnet18_train.yaml', type=str)
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    trainer = Trainer(args.config_file)
    trainer.train()
