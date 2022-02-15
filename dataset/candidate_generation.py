import numpy as np

# this code is modified from https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
def generate_anchors(im_w, im_h, bins=12):
    assert (im_w > 100) and (im_h > 100), (im_w, im_h)
    step_h = im_h / bins
    step_w = im_w / bins
    pdefined_anchors = []
    for x1 in range(0, int(bins / 3)):
        for y1 in range(0, int(bins / 3)):
            for x2 in range(int(bins / 3 * 2), bins):
                for y2 in range(int(bins / 3 * 2), bins):
                    area = (x2 - x1) * (y2 - y1) / float(bins * bins)
                    aspect_ratio = (y2 - y1) * step_h / ((x2 - x1) * step_w)
                    if area > 0.4999 and aspect_ratio > 0.5 and aspect_ratio < 2.0:
                        crop_x1 = int(step_w * (0.5+x1))
                        crop_y1 = int(step_h * (0.5+y1))
                        crop_x2 = int(step_w * (0.5 + x2))
                        crop_y2 = int(step_h * (0.5+y2))
                        pdefined_anchors.append([crop_x1, crop_y1, crop_x2, crop_y2])
    pdefined_anchors = np.array(pdefined_anchors).reshape(-1,4)
    # print('image size:({},{}), obtain {} pre-defined anchors.'.format(
    #     im_w, im_h, pdefined_anchors.shape[0]))
    return pdefined_anchors


def generate_anchors_aspect_ratio_specific(im_w, im_h, aspect_ratio, bins=20):
    assert (im_w > 100) and (im_h > 100), (im_w, im_h)
    assert isinstance(aspect_ratio, tuple), \
        'undefined aspect ratio type: {}'.format(aspect_ratio)
    assert aspect_ratio[0] >= 1 and aspect_ratio[1] >= 1, \
        'undefined aspect ratio type: {}'.format(aspect_ratio)
    w_step, h_step = int(aspect_ratio[0]), int(aspect_ratio[1])

    max_step = int(min(im_w / w_step, im_h / h_step))
    # limit the search space by increasing the step size
    if max_step > bins:
        scale = int(max(im_w / w_step / bins, im_h / h_step / bins))
        h_step *= scale
        w_step *= scale
        max_step = int(min(im_w / w_step, im_h / h_step))
    # print('image_size:{}, aspect_ratio: {}, step:{}, max_steps:{}'.format(
    #     (im_w, im_h), aspect_ratio, (w_step, h_step), max_step))
    min_step = int(max_step / 2. - 1)
    pdefined_anchors = []
    for i in range(min_step, max_step):
        out_h = h_step * i
        out_w = w_step * i
        if out_h < im_h and out_w < im_w and (out_w * out_h > 0.4 * im_w * im_h):
            for w_start in range(0, im_w - out_w, w_step):
                for h_start in range(0, im_h - out_h, h_step):
                    x1 = int(w_start)
                    y1 = int(h_start)
                    x2 = int(w_start + out_w - 1)
                    y2 = int(h_start + out_h - 1)
                    pdefined_anchors.append([x1, y1, x2, y2])
    pdefined_anchors = np.array(pdefined_anchors).reshape(-1, 4)
    # print('aspect-ratio:{}, image size:({},{}), obtain {} pre-defined anchors.'.format(
    #     aspect_ratio, im_w, im_h, pdefined_anchors.shape[0]))
    return pdefined_anchors

if __name__ == '__main__':
    print(generate_anchors(384, 256))
    # print(generate_anchors_aspect_ratio_specific(128, 128, (1,  1), bins=20))
    # print(generate_anchors_aspect_ratio_specific(512, 512, (3,  4), bins=20))
    # print(generate_anchors_aspect_ratio_specific(512, 512, (16, 9), bins=20))