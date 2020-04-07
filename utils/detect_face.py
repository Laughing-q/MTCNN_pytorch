import torch
from torchvision.transforms import functional as F
# import torch.functional as F
from torchvision.ops.boxes import batched_nms
import cv2
from PIL import Image
import numpy as np
import os
from utils.utils import *


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    pnet.eval()
    rnet.eval()
    onet.eval()
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        imgs = torch.as_tensor(imgs, device=device)
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception('MTCNN batch processing only compatible with equal-dimension images.')
        imgs = np.stack([np.uint8(img) for img in imgs])

        imgs = torch.as_tensor(imgs, device=device)

    imgs = imgs.permute(0, 3, 1, 2).float()

    batch_size = imgs.shape[0]
    h, w = imgs.shape[2:]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m

    # create scale pyramid
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor

    # First stage
    boxes = []
    image_inds = []
    all_inds = []
    all_i = 0
    for scale in scales:
        imgs_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        imgs_data = (imgs_data - 127.5) * 0.0078125
        conf, reg = pnet(imgs_data)

        boxes_scale, image_inds_scale = generateBoundingBox(conf[0, :], reg, scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)
        all_inds.append(all_i + image_inds_scale)
        all_i += batch_size

    boxes = torch.cat(boxes, dim=0)  # 各个尺度的boxes拼接在一起
    image_inds = torch.cat(image_inds, dim=0).cpu()
    all_inds = torch.cat(all_inds, dim=0)
    # print(boxes.shape)
    # NMS with each scale + image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], all_inds, 0.5)
    boxes, image_inds = boxes[pick], image_inds[pick]
    # print(boxes.shape)
    # NMS with each image
    pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]
    # print(boxes.shape)
    regw = boxes[:, 2] - boxes[:, 0]
    regh = boxes[:, 3] - boxes[:, 1]
    qq1 = boxes[:, 0] + boxes[:, 5] * regw
    qq2 = boxes[:, 1] + boxes[:, 6] * regh
    qq3 = boxes[:, 2] + boxes[:, 7] * regw
    qq4 = boxes[:, 3] + boxes[:, 8] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)  # (len(boxes), 5)

    boxes = to_square(boxes)
    y, ey, x, ex = pad(boxes, w, h)

    # Second stage
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) / 127.5
        out = rnet(im_data)
        out0 = out[0].permute(1, 0)  # (2, len(boxes))
        out1 = out[1].permute(1, 0)  # (4, len(boxes))
        score = out0[0, :]
        ipass = score > threshold[1]
        # print(ipass)
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)  # (len(ipass), 5)
        image_inds = image_inds[ipass]
        mv = out1[:, ipass].permute(1, 0)  # (4, len(ipass))
        # print(boxes.shape)
        # NMS with each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        # print(boxes.shape)
        boxes = bbreg(boxes, mv)
        boxes = to_square(boxes)
        # print(boxes.shape)
    # Third stage
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) / 127.5
        out = onet(im_data)

        out0 = out[0].permute(1, 0)  # (2, len(boxes))
        out1 = out[1].permute(1, 0)  # (4, len(boxes))
        out2 = out[2].permute(1, 0)  # (10, len(boxes))
        score = out0[0, :]
        points = out2
        ipass = score > threshold[2]
        points = points[:, ipass]
        boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)  # (len(ipass), 5)
        image_inds = image_inds[ipass]
        mv = out1[:, ipass].permute(1, 0)

        w_i = boxes[:, 2] - boxes[:, 0] + 1
        h_i = boxes[:, 3] - boxes[:, 1] + 1

        points = torch.stack((points[0, :], points[2, :], points[4, :], points[6, :], points[8, :], points[1, :],
                             points[3, :], points[5, :], points[7, :], points[9, :]), dim=0)

        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1  # (5, len(boxes))
        points_y = h_i.repeat(5, 1) * points[5:, :] + boxes[:, 1].repeat(5, 1) - 1  # (5, len(boxes))
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)  # (len(boxes), 5, 2)
        boxes = bbreg(boxes, mv)  # (len(boxes), 5)

        # NMS with each image using 'Min' strategy
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()

    batch_boxes = []
    batch_points = []

    for b in range(batch_size):
        b_indx = np.where(image_inds == b)
        batch_boxes.append(boxes[b_indx].copy())
        batch_points.append(points[b_indx].copy())

    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)
    return batch_boxes, batch_points
