from torch.nn.functional import interpolate
import torch
import numpy as np


def box_iou(box1, box2, method=None):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)
    w = np.maximum(0, x2 - x1 + 1)
    h = np.maximum(0, y2 - y1 + 1)
    inter = w * h
    area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    if method == 'Min':
        area = np.minimum(area1, area2)
    else:
        area = area1 + area2 - inter
    iou = inter / area
    return iou
#
#
# def nms(boxes, score, threshold, method=None):
#     I = np.argsort(score)
#     keep = []
#     while len(I) > 0:
#         i = I[-1]
#         index = I[0:-1]
#         keep.append(boxes[i])
#         iou = box_iou(boxes[i].unsqueeze(0), boxes[index], method)
#         I = I[np.where(iou < threshold)]
#     return keep


def imresample(img, size):
    return interpolate(img, size=size, mode='area')


def generateBoundingBox(conf, reg, scale, thresh):
    """
    create boxes
    :param conf: tensor(N*H*W)
    :param reg: tensor(N*4*H*W)
    :param scale: number
    :param thresh: number
    :return:boundingbox: tensor(len(mask), 9), box+Confidence+regression
    :return:image_inds: tensor(len(mask), )
    """
    stride = 2
    cellsize = 12

    reg = reg.permute(1, 0, 2, 3)  # (C, N, H, W)

    mask = conf >= thresh
    # print(mask.shape)
    # print(reg.shape)
    mask_inds = mask.nonzero()  # (len(mask), 3)
    image_inds = mask_inds[:, 0]  # (len(mask), )
    score = conf[mask]  # (len(mask), )
    reg = reg[:, mask].permute(1, 0)  # (len(mask), 4)
    bb = mask_inds[:, 1:].float().flip(1)  # (len(mask), 2)
    q1 = ((stride * bb + 1) / scale).floor()  # (len(mask), 2)
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)  # (len(mask), 9)
    return boundingbox, image_inds


def to_square(boxes):
    """
    :param boxes:
    :return:
    """
    h = boxes[:, 3] - boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]

    l = torch.max(w, h)
    boxes[:, 0] = boxes[:, 0] + w * 0.5 - l * 0.5
    boxes[:, 1] = boxes[:, 1] + h * 0.5 - l * 0.5
    boxes[:, 2:4] = boxes[:, :2] + l.repeat(2, 1).permute(1, 0)

    return boxes


def pad(boxes, w, h):
    """
    规范超出image的box
    :param boxes:
    :param w:
    :param h:
    :return:
    """
    # trunc()返回一个新张量，包含输入input张量每个元素的截断值,标量x的截断值是最接近其的整数
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex


def bbreg(boxes, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    bx1 = boxes[:, 0] + reg[:, 0] * w
    by1 = boxes[:, 1] + reg[:, 1] * h
    bx2 = boxes[:, 2] + reg[:, 2] * w
    by2 = boxes[:, 3] + reg[:, 3] * h

    boxes[:, :4] = torch.stack([bx1, by1, bx2, by2]).permute(1, 0)  # (len(boxes), 5)

    return boxes


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)  # (len(boxes), )
    boxes_for_nms = boxes + offsets[:, None]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0].copy()
    y1 = boxes[:, 1].copy()
    x2 = boxes[:, 2].copy()
    y2 = boxes[:, 3].copy()

    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    I = np.argsort(s)  # 从小到大排序索引
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]

        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()

        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()

        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]

    pick = pick[:counter].copy()
    return pick


if __name__ == '__main__':
    a = torch.rand(20, 4)
    to_square(a)
