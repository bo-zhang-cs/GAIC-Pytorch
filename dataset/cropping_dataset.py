import os
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import cv2
import json
from config.GAIC_config import cfg

MOS_MEAN = 2.95
MOS_STD  = 0.8
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

def rescale_crops(boxes, ratio_w, ratio_h):
    boxes = np.array(boxes).reshape(-1, 4)
    boxes[:, 0] = np.floor(boxes[:, 0] * ratio_w)
    boxes[:, 1] = np.floor(boxes[:, 1] * ratio_h)
    boxes[:, 2] = np.ceil(boxes[:, 2] * ratio_w)
    boxes[:, 3] = np.ceil(boxes[:, 3] * ratio_h)
    return boxes.astype(np.float32)

def is_number(s):
    if not isinstance(s, str):
        return False
    if s.isdigit():
        return True
    else:
        try:
            float(s)
            return True
        except:
            return False

class GAICDataset(Dataset):
    def __init__(self, split):
        self.split = split
        assert self.split in ['train', 'test'], self.split
        self.keep_aspect = cfg.keep_aspect_ratio
        self.data_dir = cfg.GAIC_folder
        assert os.path.exists(self.data_dir), self.data_dir
        self.image_dir = os.path.join(self.data_dir, 'images', split)
        assert os.path.exists(self.image_dir), self.image_dir
        self.image_list = [file for file in os.listdir(self.image_dir) if file.endswith('.jpg')]
        # print('GAICD {} set contains {} images'.format(split, len(self.image_list)))
        self.anno_dir  = os.path.join(self.data_dir, 'annotations')
        assert os.path.exists(self.anno_dir), self.anno_dir
        self.annos = self.parse_annotations()

        self.image_size = cfg.image_size
        self.augmentation = (cfg.data_augmentation and self.split == 'train')
        self.PhotometricDistort = transforms.ColorJitter(
            brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05)
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)])

    def parse_annotations(self):
        image_annos = dict()
        for image_name in self.image_list:
            anno_file = os.path.join(self.anno_dir, image_name.replace('.jpg', '.txt'))
            assert os.path.exists(anno_file), anno_file
            with open(anno_file, 'r') as f:
                crops,scores = [],[]
                for line in f.readlines():
                    line = line.strip().split(' ')
                    values = [s for s in line if is_number(s)]
                    y1,x1,y2,x2 = [int(s) for s in values[0:4]]
                    s = float(values[-1])
                    if s > -2:
                        crops.append([x1,y1,x2,y2])
                        scores.append(s)
                if len(crops) == 0:
                    print(image_name, anno_file)
                else:
                    # rank all crops
                    rank = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                    scores = [scores[i] for i in rank]
                    crops  = [crops[i]  for i in rank]
                    image_annos[image_name] = {'crops':crops, 'scores':scores}
        return image_annos

    def __len__(self):
            return len(self.image_list)

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image_file = os.path.join(self.image_dir, image_name)
        image = Image.open(image_file).convert('RGB')
        im_width, im_height = image.size
        if self.keep_aspect:
            scale = float(cfg.image_size[0]) / min(im_height, im_width)
            h = round(im_height * scale / 32.0) * 32
            w = round(im_width * scale / 32.0) * 32
        else:
            h = cfg.image_size[1]
            w = cfg.image_size[0]
        resized_image = image.resize((w, h), Image.ANTIALIAS)
        crop  = self.annos[image_name]['crops']
        rs_width, rs_height = resized_image.size
        ratio_w = float(rs_width) / im_width
        ratio_h = float(rs_height) / im_height
        crop  = rescale_crops(crop, ratio_w, ratio_h)
        score = np.array(self.annos[image_name]['scores']).reshape((-1)).astype(np.float32)
        if self.augmentation:
            if random.uniform(0,1) > 0.5:
                resized_image = ImageOps.mirror(resized_image)
                temp_x1 = crop[:, 0].copy()
                crop[:, 0] = rs_width - crop[:, 2]
                crop[:, 2] = rs_width - temp_x1
            resized_image = self.PhotometricDistort(resized_image)
        im = self.image_transformer(resized_image)
        return im, crop, score, im_width, im_height, image_file

if __name__ == '__main__':
    GAICD_testset = GAICDataset(split='train')
    print('GAICD training set has {} images'.format(len(GAICD_testset)))
    dataloader = DataLoader(GAICD_testset, batch_size=1, num_workers=0)
    for batch_idx, data in enumerate(dataloader):
        im, crops, scores, w, h, file = data
        print(im.shape, crops.shape, scores.shape, w.shape, h.shape)