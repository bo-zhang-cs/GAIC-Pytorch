import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from untils.roi_align.modules.roi_align import RoIAlignAvg, RoIAlign
from untils.rod_align.modules.rod_align import RoDAlignAvg, RoDAlign
from torchvision.models.mobilenetv2 import mobilenet_v2 as MobileNetV2
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_0 as ShuffleNetV2
import torch.nn.init as init
from thop import profile
import warnings
warnings.filterwarnings('ignore')

class vgg_base(nn.Module):
    def __init__(self, loadweights=True):
        super(vgg_base, self).__init__()

        vgg = models.vgg16(pretrained=loadweights)
        self.feature3 = nn.Sequential(vgg.features[:23])
        self.feature4 = nn.Sequential(vgg.features[23:30])
        self.feature5 = nn.Sequential(vgg.features[30:])

        # img = torch.randn((1, 3, 256, 256))
        # flops, params = profile(vgg.features[:-1], inputs=(img,))
        # print("params: %.2fMB    flops: %.2fG" % (params / (1000 ** 2), flops / (1000 ** 3)))
        # params: 14.71MB    flops: 20.06G

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

class resnet50_base(nn.Module):
    def __init__(self, loadweights=True):
        super(resnet50_base, self).__init__()

        resnet50 = models.resnet50(pretrained=True)

        self.feature3 = nn.Sequential(resnet50.conv1, resnet50.bn1,
                                      resnet50.relu,resnet50.maxpool,
                                      resnet50.layer1,resnet50.layer2)
        self.feature4 = nn.Sequential(resnet50.layer3)
        self.feature5 = nn.Sequential(resnet50.layer4)

        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class mobilenetv2_base(nn.Module):

    def __init__(self, loadweights=True):
        super(mobilenetv2_base, self).__init__()

        model = MobileNetV2(pretrained=loadweights)

        self.feature3 = nn.Sequential(model.features[:7])
        self.feature4 = nn.Sequential(model.features[7:14])
        self.feature5 = nn.Sequential(model.features[14:-1])
        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5


class shufflenetv2_base(nn.Module):

    def __init__(self, loadweights=True):
        super(shufflenetv2_base, self).__init__()

        model = ShuffleNetV2(pretrained=loadweights)

        self.feature3 = nn.Sequential(model.conv1, model.maxpool, model.stage2)
        self.feature4 = nn.Sequential(model.stage3)
        self.feature5 = nn.Sequential(model.stage4)
        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        #return self.feature(x)
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

def fc_layers(reddim=32, alignsize=8):
    conv1 = nn.Sequential(nn.Conv2d(reddim, 768, kernel_size=alignsize, padding=0),
                          nn.ReLU(inplace=True))
    conv2 = nn.Sequential(nn.Conv2d(768, 128, kernel_size=1),
                          nn.ReLU(inplace=True))
    conv3 = nn.Conv2d(128, 1, kernel_size=1)
    layers = nn.Sequential(conv1, conv2, conv3)
    return layers

class crop_model_single_scale(nn.Module):

    def __init__(self, alignsize = 8, reddim = 8, loadweight = True, model = None):
        super(crop_model_single_scale, self).__init__()

        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight)
            self.DimRed = nn.Conv2d(232, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight)
            self.DimRed = nn.Conv2d(96, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight)
            self.DimRed = nn.Conv2d(512, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight)
            self.DimRed = nn.Conv2d(1024, reddim, kernel_size=1, padding=0)
        downsample = 4
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.FC_layers = fc_layers(reddim*2, alignsize)

        #flops, params = profile(self.FC_layers, input_size=(1,reddim*2,9,9))

    def forward(self, im_data, boxes):

        f3,base_feat,f5 = self.Feat_ext(im_data)
        red_feat = self.DimRed(base_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)
        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)

class crop_model_multi_scale_shared(nn.Module):

    def __init__(self, alignsize = 8, reddim = 32, loadweight = True, model = None, ):
        super(crop_model_multi_scale_shared, self).__init__()
        downsample = 4
        if model == 'shufflenetv2':
            self.Feat_ext = shufflenetv2_base(loadweight)
            self.DimRed = nn.Conv2d(812, reddim, kernel_size=1, padding=0)
        elif model == 'mobilenetv2':
            self.Feat_ext = mobilenetv2_base(loadweight)
            self.DimRed = nn.Conv2d(448, reddim, kernel_size=1, padding=0)
        elif model == 'vgg16':
            self.Feat_ext = vgg_base(loadweight)
            self.DimRed = nn.Conv2d(1536, reddim, kernel_size=1, padding=0)
        elif model == 'resnet50':
            self.Feat_ext = resnet50_base(loadweight)
            self.DimRed = nn.Conv2d(3584, reddim, kernel_size=1, padding=0)

        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.FC_layers = fc_layers(reddim*2, alignsize)

    def forward(self, im_data, boxes):
        # print(im_data.shape, im_data.dtype, im_data.device, boxes.shape, boxes.dtype, boxes.device)
        B, N, _ = boxes.shape
        if boxes.shape[-1] == 4:
            index = torch.arange(B).view(-1, 1).repeat(1, N).reshape(B, N, 1).to(boxes.device)
            boxes = torch.cat((index, boxes),dim=-1).contiguous()
        if boxes.dim() == 3:
            boxes = boxes.view(-1,5)

        f3,f4,f5 = self.Feat_ext(im_data)
        f3 = F.interpolate(f3, size=f4.shape[2:], mode='bilinear', align_corners=True)
        f5 = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=True)
        cat_feat = torch.cat((f3,f4,0.5*f5),1)

        red_feat = self.DimRed(cat_feat)
        RoI_feat = self.RoIAlign(red_feat, boxes)
        RoD_feat = self.RoDAlign(red_feat, boxes)

        final_feat = torch.cat((RoI_feat, RoD_feat), 1)
        prediction = self.FC_layers(final_feat)
        return prediction

    def _init_weights(self):
        print('Initializing weights...')
        self.DimRed.apply(weights_init)
        self.FC_layers.apply(weights_init)

def cropping_rank_loss(pre_score, gt_score):
    '''
    :param pre_score:
    :param gt_score:
    :return:
    '''
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    N = pre_score.shape[0]
    pair_num = N * (N-1) / 2
    pre_diff = pre_score[:,None] - pre_score[None,:]
    gt_diff  = gt_score[:,None]  - gt_score[None,:]
    indicat  = -1 * torch.sign(gt_diff) * (pre_diff - gt_diff)
    diff     = torch.maximum(indicat, torch.zeros_like(indicat))
    rank_loss= torch.sum(diff) / pair_num
    return rank_loss

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def build_crop_model(scale='single', alignsize=9, reddim=32, loadweight=True, model=None):
    if scale=='single':
        return crop_model_single_scale(alignsize, reddim, loadweight, model)
    elif scale=='multi':
        return crop_model_multi_scale_shared(alignsize, reddim, loadweight, model)


if __name__ == '__main__':
    net = build_crop_model(scale='multi', alignsize=9,
                           reddim=32, loadweight=True,
                           model='vgg16')
    net = net.eval().cuda()
    roi = torch.tensor([[0, 0, 128, 128], [64, 64, 223, 223]]).float()
    roi = roi.unsqueeze(0).cuda()
    roi = roi.repeat(2,1,1)
    img = torch.randn((2, 3, 224, 224)).cuda()
    out = net(img, roi)
    print(out.shape, out)



