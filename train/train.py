import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorboardX import SummaryWriter
import torch
import time
import datetime
import csv
import random
import shutil
from networks.GAIC_model import build_crop_model
from dataset.cropping_dataset import GAICDataset
from config.GAIC_config import cfg, refresh_yaml_params
from evaluate.test import evaluate_on_GAICD_official as evaluate_on_GAICD
import argparse
import warnings
warnings.filterwarnings('ignore')

def create_dataloader():
    dataset = GAICDataset(split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,
                                             shuffle=True, num_workers=cfg.num_workers,
                                             drop_last=False, pin_memory=False)
    print('training set has {} samples, {} batches'.format(len(dataset), len(dataloader)))
    return dataloader


class Trainer:
    def __init__(self, model):
        self.model = model
        self.epoch = 0
        self.iters = 0
        self.max_epoch = cfg.max_epoch
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.optimizer, self.lr_scheduler = self.get_optimizer()
        self.train_loader = create_dataloader()
        self.eval_results = []
        self.best_results = {'acc1_5':0., 'acc2_5':0., 'acc3_5':0., 'acc4_5':0., 'acc5':0.,
                             'acc1_10':0., 'acc2_10':0., 'acc3_10':0, 'acc4_10':0, 'acc10':0.,
                             'srcc':0., 'pcc':0.}
        self.criterion = torch.nn.SmoothL1Loss(reduction='mean')
        self.contain_BN = False
        for name,m in self.model.Feat_ext.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                self.contain_BN = True
                break

    def get_optimizer(self):
        optim = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, milestones=cfg.lr_decay_epoch, gamma=cfg.lr_decay
        )
        return optim, lr_scheduler

    def run(self):
        print(("========  Begin Training  ========="))
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train()
            if epoch % cfg.eval_freq == 0:
                self.eval()
                self.record_eval_results()
            self.lr_scheduler.step()

    def train(self):
        self.model.train()
        if self.contain_BN:
            self.model.Feat_ext.eval()
        device = next(self.model.parameters()).device
        start = time.time()
        batch_loss = 0
        total_batch = len(self.train_loader)
        total_loss = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            self.iters += 1
            im = batch_data[0].to(device)
            rois = batch_data[1].to(device)
            scores = batch_data[2].to(device)
            width = batch_data[3].to(device)
            height = batch_data[4].to(device)

            random_ID = list(range(0, rois.shape[1]))
            random.shuffle(random_ID)
            chosen_ID = random_ID[:64]

            rois = rois[:,chosen_ID]
            scores = scores[:, chosen_ID]

            pred_scores = self.model(im, rois)
            loss = self.criterion(pred_scores.squeeze(), scores.squeeze())
            batch_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % cfg.display_freq == 0:
                avg_loss = batch_loss / (1 + batch_idx)
                cur_lr   = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/loss', avg_loss, self.iters)
                self.writer.add_scalar('train/lr',   cur_lr,   self.iters)

                time_per_batch = (time.time() - start) / (batch_idx + 1.)
                last_batches = (self.max_epoch - self.epoch - 1) * total_batch + (total_batch - batch_idx - 1)
                last_time = int(last_batches * time_per_batch)
                time_str = str(datetime.timedelta(seconds=last_time))

                print('=== epoch:{}/{}, step:{}/{} | Loss:{:.4f} | lr:{:.6f} | estimated remaining time:{} ==='.format(
                    self.epoch, self.max_epoch, batch_idx, total_batch, avg_loss, cur_lr, time_str
                ))

    def eval(self):
        self.model.eval()
        avg_srcc, avg_pcc, avg_acc5, avg_acc10, acc4_5, acc4_10 = evaluate_on_GAICD(self.model)
        self.eval_results.append([self.epoch, avg_srcc, avg_pcc, avg_acc5, avg_acc10,
                                  acc4_5[0], acc4_5[1], acc4_5[2], acc4_5[3],
                                  acc4_10[0], acc4_10[1], acc4_10[2], acc4_10[3]])
        epoch_result = {'srcc': avg_srcc, 'pcc': avg_pcc, 'acc5': avg_acc5, 'acc10': avg_acc10,
                        'acc1_5': acc4_5[0], 'acc2_5': acc4_5[1], 'acc3_5': acc4_5[2], 'acc4_5': acc4_5[3],
                        'acc1_10': acc4_10[0], 'acc2_10': acc4_10[1], 'acc3_10': acc4_10[2], 'acc4_10': acc4_10[3]}
        for m in epoch_result.keys():
            update = False
            if (epoch_result[m] > self.best_results[m]):
                update = True
            if update:
                self.best_results[m] = epoch_result[m]
                checkpoint_path = os.path.join(cfg.checkpoint_dir, 'best-{}.pth'.format(m))
                torch.save(self.model.state_dict(), checkpoint_path)
                print('Update best {} model, best {}={:.4f}'.format(m, m, self.best_results[m]))
            if m in ['srcc', 'acc5']:
                self.writer.add_scalar('test/{}'.format(m), epoch_result[m], self.epoch)
                self.writer.add_scalar('test/best-{}'.format(m), self.best_results[m], self.epoch)
        if self.epoch % cfg.save_freq == 0:
            checkpoint_path = os.path.join(cfg.checkpoint_dir, 'epoch-{}.pth'.format(self.epoch))
            torch.save(self.model.state_dict(), checkpoint_path)

    def record_eval_results(self):
        csv_path = os.path.join(cfg.exp_path, '..', '{}.csv'.format(cfg.exp_name))
        header = ['epoch', 'srcc', 'pcc',  'acc5', 'acc10',
                  'acc1_5', 'acc2_5', 'acc3_5', 'acc4_5',
                  'acc1_10', 'acc2_10', 'acc3_10', 'acc4_10']
        # Limit the number of decimal places in the result
        limit_results = []
        for epoch, result in enumerate(self.eval_results):
            limit_results.append([])
            for i,r in enumerate(result):
                if i == 0: # epoch
                    limit_results[epoch].append(r)
                else:
                    limit_results[epoch].append(round(r, 3))
        # find the best results
        rows = [header] + limit_results
        metrics = [[] for i in header]
        for result in limit_results:
            for i, r in enumerate(result):
                metrics[i].append(r)
        for name, m in zip(header, metrics):
            if name == 'epoch':
                continue
            index = m.index(max(m))
            title = 'best {}(epoch-{})'.format(name, index)
            row = [l[index] for l in metrics]
            row[0] = title
            rows.append(row)
        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ', csv_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a GAIC model")
    parser.add_argument('--gpu', type=int, dest='gpu_id',
                        help='gpu_id', default=0)
    parser.add_argument('--backbone', type=str, choices=['vgg16', 'mobilenetv2', 'resnet50', 'shufflenetv2'],
                        help='the architecture of backbone network', default='vgg16')
    parser.add_argument('--reddim', type=int, choices=[64, 32, 16, 8],
                        help='the reduced channel dimension of the feature map', default=32)
    parser.add_argument('--alignsize', type=int, choices=[3, 5, 9],
                        help='RoIAlign and RoDAlign output size', default=9)
    parser.add_argument('--num_workers', type=int, help='number of dataloader workers', default=8)
    args = parser.parse_args()
    refresh_yaml_params(args)


if __name__ == '__main__':
    parse_args()
    cfg.refresh_params()
    cfg.create_path()
    device = torch.device('cuda:{}'.format(cfg.gpu_id))
    torch.cuda.set_device(device)
    for file in ['config/GAIC_config.py', 'config/GAIC_params.yaml', 'dataset/cropping_dataset.py',
                 'evaluate/test.py', 'networks/GAIC_model.py', 'train/train.py']:
        if not os.path.exists(file):
            file = os.path.join('..', file)
        shutil.copy(file, cfg.code_dir)
        print('backup', file)
    net = build_crop_model(scale='multi', alignsize=cfg.alignsize, reddim=cfg.reddim,
                           loadweight=True, model=cfg.backbone)
    net = net.to(device)
    trainer = Trainer(net)
    trainer.run()
