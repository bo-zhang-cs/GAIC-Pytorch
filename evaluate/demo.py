import sys,os
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_PATH)
import numpy as np
import torch
from networks.GAIC_model import build_crop_model
import argparse
import cv2
from PIL import Image
from dataset.candidate_generation import generate_anchors, generate_anchors_aspect_ratio_specific
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD  = [0.229, 0.224, 0.225]
image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])


def parse_args():
    parser = argparse.ArgumentParser(description="Run cropping model on images")
    parser.add_argument('--gpu', type=int, dest='gpu_id',
                        help='gpu_id', default=0)
    parser.add_argument('--backbone', type=str, choices=['vgg16', 'mobilenetv2', 'shufflenetv2'],
                        help='the architecture of backbone network', default='vgg16')
    parser.add_argument('--image_dir', type=str, default='test_images',
                        help='the directory of test images')
    parser.add_argument('--save_dir', type=str, default='result_images',
                        help='the directory of saving resulting images')
    args = parser.parse_args()
    assert os.path.exists(args.image_dir), args.image_dir
    os.makedirs(args.save_dir, exist_ok=True)
    return args

def build_network(backbone):
    if backbone in ['vgg16','shufflenetv2']:
        reddim = 32
    elif backbone == 'mobilenetv2':
        reddim = 16
    else:
        raise Exception('undefined backbone architecture', backbone)
    net = build_crop_model(scale='multi', alignsize=9, reddim=reddim,
                           loadweight=False, model=backbone)
    weights_path = 'pretrained_models/GAIC-{}-reddim{}.pth'.format(backbone, reddim)
    assert os.path.exists(weights_path), weights_path
    print('load pretrained weights from ', weights_path)
    net.load_state_dict(torch.load(weights_path))
    return net

def get_image_list(args):
    img_list = []
    if os.path.isdir(args.image_dir):
        for file in os.listdir(args.image_dir):
            if file.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                img_list.append(os.path.join(args.image_dir, file))
    else:
        if args.image_dir.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            img_list.append(args.image_dir)
    print('find total {} images'.format(len(img_list)))
    return img_list

def image_preprocessing(im):
    im_width,im_height = im.size
    scale = 256. / min(im_height, im_width)
    h = round(im_height * scale / 32.0) * 32
    w = round(im_width * scale / 32.0) * 32
    resized_image = im.resize((w, h), Image.ANTIALIAS)
    im_tensor = image_transformer(resized_image).unsqueeze(0)
    return im_tensor

def predict_best_crop(model, im_tensor, anchors, im):
    im_width, im_height = im.size
    with torch.no_grad():
        rois = anchors.astype(np.float32)
        rois = torch.from_numpy(rois).unsqueeze(0).to(im_tensor.device)
        scores = model(im_tensor, rois)
        scores = scores.detach().cpu().numpy().reshape(-1)
    pr_idx = np.argmax(scores)
    # mapping the coordinates of predefined anchors to source image
    rescale_anchors = anchors.astype(np.float32)
    rescale_anchors[:,0::2] = rescale_anchors[:,0::2] / im_tensor.shape[-1] * im_width
    rescale_anchors[:,1::2] = rescale_anchors[:,1::2] / im_tensor.shape[-2] * im_height
    rescale_anchors = rescale_anchors.astype(np.int32)
    pr_bbox = rescale_anchors[pr_idx].tolist()
    x1, y1, x2, y2 = pr_bbox
    pr_crop = im.crop((x1, y1, x2, y2))
    # pr_crop = np.asarray(pr_crop)[:,:,::-1] # convert to opencv format
    return pr_crop, pr_bbox


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:{}'.format(args.gpu_id))
    net = build_network(args.backbone)
    net = net.eval().to(device)
    img_list = get_image_list(args)

    for i,img in enumerate(img_list):
        im_name = os.path.basename(img)
        src = Image.open(img).convert('RGB')
        src_tensor = image_preprocessing(src).to(device)
        input_w, input_h = src_tensor.shape[-1], src_tensor.shape[-2]
        # generate aspect-ratio-agnostic crops
        anchors = generate_anchors(input_w, input_h)
        best_crop, bbox = predict_best_crop(net, src_tensor, anchors, src)
        print('source image:{} {}, num_candidates:{}, best crop bbox:{}, crop(w,h):{}'.format(
            im_name, src.size, anchors.shape[0], bbox, best_crop.size))

        # generage aspect-ratio-specific crops
        anchors_1_1  = generate_anchors_aspect_ratio_specific(input_w, input_h, (1,1), bins=30)
        crop_1_1, bbox_1_1 = predict_best_crop(net, src_tensor, anchors_1_1, src)
        print('aspect_ratio=1:1, num_candidates:{}, best crop bbox:{}, crop(w,h):{}'.format(
            anchors_1_1.shape[0], bbox_1_1, crop_1_1.size))

        anchors_4_3  = generate_anchors_aspect_ratio_specific(input_w, input_h, (4,3), bins=20)
        crop_4_3, bbox_4_3 = predict_best_crop(net, src_tensor, anchors_4_3, src)
        print('aspect_ratio=4:3, num_candidates:{}, best crop bbox:{}, crop(w,h):{}'.format(
            anchors_4_3.shape[0], bbox_4_3, crop_4_3.size))

        anchors_16_9 = generate_anchors_aspect_ratio_specific(input_w, input_h, (16,9), bins=15)
        crop_16_9, bbox_16_9 = predict_best_crop(net, src_tensor, anchors_16_9, src)
        print('aspect_ratio=16:9, num_candidates:{}, best crop bbox:{}, crop(w,h):{}'.format(
            anchors_16_9.shape[0], bbox_16_9, crop_16_9.size))

        # results visualization
        crop_list = [src, best_crop, crop_1_1, crop_4_3, crop_16_9]
        title_list = ['source image', 'best crop', '1:1', '4:3', '16:9']
        fig_cols = 5
        fig_rows = (len(crop_list) + fig_cols - 1) // fig_cols
        fig = plt.figure(figsize=(20,5))
        for i in range(len(crop_list)):
            ax = fig.add_subplot(fig_rows, fig_cols, i+1)
            ax.imshow(crop_list[i])
            ax.set_axis_off()
            ax.set_title(title_list[i])
        fig.tight_layout()
        result_file = os.path.join(args.save_dir, im_name)
        plt.savefig(result_file)
        plt.close()
        print('Save results to ', result_file)
        print()

