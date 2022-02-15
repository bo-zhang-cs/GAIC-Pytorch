import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm
import pickle
from scipy.stats import spearmanr, pearsonr
import math
from dataset.cropping_dataset import GAICDataset
from config.GAIC_config import cfg
from networks.GAIC_model import build_crop_model
from thop import profile

def compute_acc(gt_scores, pr_scores):
    assert (len(gt_scores) == len(pr_scores)), '{} vs. {}'.format(len(gt_scores), len(pr_scores))
    sample_cnt = 0
    acc4_5  = [0 for i in range(4)]
    acc4_10 = [0 for i in range(4)]
    for i in range(len(gt_scores)):
        gts, preds = gt_scores[i], pr_scores[i]
        id_gt = sorted(range(len(gts)), key=lambda j : gts[j], reverse=True)
        id_pr = sorted(range(len(preds)), key=lambda j : preds[j], reverse=True)
        for k in range(4):
            temp_acc4_5  = 0.
            temp_acc4_10 = 0.
            for j in range(k+1):
                if gts[id_pr[j]] >= gts[id_gt[4]]:
                    temp_acc4_5 += 1.0
                if gts[id_pr[j]] >= gts[id_gt[9]]:
                    temp_acc4_10 += 1.0
            acc4_5[k]  += (temp_acc4_5 / (k+1.0))
            acc4_10[k] += ((temp_acc4_10) / (k+1.0))
        sample_cnt += 1
    acc4_5  = [round(i / sample_cnt,3) for i in acc4_5]
    acc4_10 = [round(i / sample_cnt,3) for i in acc4_10]
    # print('acc4_5', acc4_5)
    # print('acc4_10', acc4_10)
    return acc4_5, acc4_10

def evaluate_on_GAICD_official(model):
    # https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
    model.eval()
    device = next(model.parameters()).device
    print('='*5, 'Evaluating on GAICD dataset', '='*5)
    count = 0
    test_dataset = GAICDataset(split='test')
    test_loader  = torch.utils.data.DataLoader(
                        test_dataset, batch_size=1,
                        shuffle=False, num_workers=cfg.num_workers,
                        drop_last=False)
    acc4_5 = []
    acc4_10 = []
    wacc4_5 = []
    wacc4_10 = []
    srcc = []
    pcc = []
    for n in range(4):
        acc4_5.append(0)
        acc4_10.append(0)
        wacc4_5.append(0)
        wacc4_10.append(0)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(device)
            rois = batch_data[1].to(device)
            MOS = batch_data[2].reshape(-1,1)
            width = batch_data[3]
            height = batch_data[4]
            count += im.shape[0]

            out = model(im, rois)
            id_MOS = sorted(range(len(MOS)), key=lambda k: MOS[k], reverse=True)
            id_out = sorted(range(len(out)), key=lambda k: out[k], reverse=True)

            rank_of_returned_crop = []
            for k in range(4):
                rank_of_returned_crop.append(id_MOS.index(id_out[k]))

            for k in range(4):
                temp_acc_4_5 = 0.0
                temp_acc_4_10 = 0.0
                for j in range(k + 1):
                    if MOS[id_out[j]] >= MOS[id_MOS[4]]:
                        temp_acc_4_5 += 1.0
                    if MOS[id_out[j]] >= MOS[id_MOS[9]]:
                        temp_acc_4_10 += 1.0
                acc4_5[k] += temp_acc_4_5 / (k + 1.0)
                acc4_10[k] += temp_acc_4_10 / (k + 1.0)

            for k in range(4):
                temp_wacc_4_5 = 0.0
                temp_wacc_4_10 = 0.0
                temp_rank_of_returned_crop = rank_of_returned_crop[:(k + 1)]
                temp_rank_of_returned_crop.sort()
                for j in range(k + 1):
                    if temp_rank_of_returned_crop[j] <= 4:
                        temp_wacc_4_5 += 1.0 * math.exp(-0.2 * (temp_rank_of_returned_crop[j] - j))
                    if temp_rank_of_returned_crop[j] <= 9:
                        temp_wacc_4_10 += 1.0 * math.exp(-0.1 * (temp_rank_of_returned_crop[j] - j))
                wacc4_5[k] += temp_wacc_4_5 / (k + 1.0)
                wacc4_10[k] += temp_wacc_4_10 / (k + 1.0)

            MOS_arr = []
            out = torch.squeeze(out).cpu().detach().numpy()
            for k in range(len(MOS)):
                MOS_arr.append(MOS[k].numpy()[0])
            srcc.append(spearmanr(MOS_arr, out)[0])
            pcc.append(pearsonr(MOS_arr, out)[0])

        for k in range(4):
            acc4_5[k] = acc4_5[k] / count
            acc4_10[k] = acc4_10[k] / count
            wacc4_5[k] = wacc4_5[k] / count
            wacc4_10[k] = wacc4_10[k] / count

        avg_srcc  = sum(srcc) / count
        avg_pcc   = sum(pcc) / count
        avg_acc5  = sum(acc4_5) / len(acc4_5)
        avg_acc10 = sum(acc4_10) / len(acc4_10)

        sys.stdout.write('Acc4_5:[%.3f, %.3f, %.3f, %.3f] Acc4_10:[%.3f, %.3f, %.3f, %.3f]\n' % (
        acc4_5[0], acc4_5[1], acc4_5[2], acc4_5[3], acc4_10[0], acc4_10[1], acc4_10[2], acc4_10[3]))
        sys.stdout.write('WAcc4_5:[%.3f, %.3f, %.3f, %.3f] WAcc4_10:[%.3f, %.3f, %.3f, %.3f]\n' % (
        wacc4_5[0], wacc4_5[1], wacc4_5[2], wacc4_5[3], wacc4_10[0], wacc4_10[1], wacc4_10[2], wacc4_10[3]))
        sys.stdout.write('[Avg SRCC: %.3f] [Avg PCC: %.3f] [Acc5: %.3f] [Acc10: %.3f]\n' % (
            avg_srcc, avg_pcc, avg_acc5, avg_acc10))
    return avg_srcc, avg_pcc, avg_acc5, avg_acc10, acc4_5, acc4_10

def evaluate_on_GAICD(model):
    device = next(model.parameters()).device
    model.eval()
    print('='*5, 'Evaluating on GAICD dataset', '='*5)
    srcc_list = []
    pcc_list  = []
    gt_scores = []
    pr_scores = []
    count = 0
    test_dataset = GAICDataset(split='test')
    test_loader  = torch.utils.data.DataLoader(
                        test_dataset, batch_size=1,
                        shuffle=False, num_workers=cfg.num_workers,
                        drop_last=False)
    test_results = dict()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(device)
            rois = batch_data[1].to(device)
            scores = batch_data[2].cpu().numpy().reshape(-1)
            width = batch_data[3]
            height = batch_data[4]
            image_name = batch_data[5][0]
            count += im.shape[0]

            pre_scores = model(im, rois)
            pre_scores = pre_scores.cpu().detach().numpy().reshape(-1)
            srcc_list.append(spearmanr(scores, pre_scores)[0])
            pcc_list.append(pearsonr(scores, pre_scores)[0])
            gt_scores.append(scores)
            pr_scores.append(pre_scores)
            test_results[image_name] = pre_scores.tolist()
    avg_srcc = sum(srcc_list) / len(srcc_list)
    avg_pcc  = sum(pcc_list)  / len(pcc_list)
    acc4_5, acc4_10 = compute_acc(gt_scores, pr_scores)
    avg_acc5 = sum(acc4_5) / len(acc4_5)
    avg_acc10 = sum(acc4_10) / len(acc4_10)
    sys.stdout.write('Acc4_5:[%.3f, %.3f, %.3f, %.3f] Acc4_10:[%.3f, %.3f, %.3f, %.3f]\n' % (
        acc4_5[0], acc4_5[1], acc4_5[2], acc4_5[3], acc4_10[0], acc4_10[1], acc4_10[2], acc4_10[3]))
    sys.stdout.write('[Avg SRCC: %.3f] [Avg PCC: %.3f] [Acc5: %.3f] [Acc10: %.3f]\n' % (
        avg_srcc, avg_pcc, avg_acc5, avg_acc10))
    return avg_srcc, avg_pcc, avg_acc5, avg_acc10, acc4_5, acc4_10


if __name__ == '__main__':
    device = torch.device('cuda:{}'.format(cfg.gpu_id))
    torch.cuda.set_device(device)
    backbone, reddim = 'vgg16', 32
    pretrained_weight = 'pretrained_models/GAIC-{}-reddim{}.pth'.format(backbone, reddim)
    model = build_crop_model(scale='multi', alignsize=9, reddim=32,
                             loadweight=False, model=backbone)
    model = model.eval().to(device)
    print('load pretrained weights from ', pretrained_weight)
    model.load_state_dict(torch.load(pretrained_weight), strict=False)
    evaluate_on_GAICD_official(model)

    # roi = torch.tensor([[0, 0, 128, 128], [64, 64, 223, 223]]).float()
    # roi = roi.unsqueeze(0).to(device)
    # img = torch.randn((1, 3, 256, 256)).to(device)
    # flops, params = profile(model, inputs=(img, roi))
    # print("params: %.2fMB    flops: %.2fG" % (params / (1000 ** 2), flops / (1000 ** 3)))