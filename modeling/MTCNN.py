from modeling.layers.ONet import *
from modeling.layers.PNet import *
from modeling.layers.RNet import *
from utils.detect_face import *
import torch
from torch import nn


class MTCNN(nn.Module):
    def __init__(self, img_size=160, min_face_size=20, margin=0,
                 thresholds=None, factor=0.709, post_process=True,
                 select_largest=True, keep_all=False, device=None):
        super(MTCNN, self).__init__()
        if thresholds is None:
            thresholds = [0.6, 0.7, 0.7]
        self.pnet = PNet(pretrained=True)
        self.rnet = RNet(pretrained=True)
        self.onet = ONet(pretrained=True)
        self.image_size = img_size
        self.min_face_size = min_face_size
        self.margin = margin
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.device = torch.device('cpu')

        if device is not None:
            self.device = device
        self.to(device)

    def detect(self, img, landmarks=False):
        """
        :param img:
        :param landmarks:
        :return:
        """
        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor,
                self.device
            )
        boxes, conf, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)  # (len(box), 5)
            point = np.array(point)  # (len(box), 5)
            if len(box) == 0:
                boxes.append(None)
                conf.append(None)
                points.append(None)
            elif self.select_largest:
                # 按照框的面积从大到小排序索引
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                conf.append(box[:, 4])
                points.append(point)
                pass
            else:
                boxes.append(box[:, :4])
                conf.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes)
        conf = np.array(conf)
        points = np.array(points)

        if not isinstance(img, (list, tuple)):
            boxes = boxes[0]
            conf = conf[0]
            points = points[0]
        if landmarks:
            return boxes, conf, points

        return boxes, conf
