import numpy as np
from utils import *
import os
import traceback
from PIL import Image
import random


box_path = r'D:\BaiduNetdiskDownload\Anno\list_bbox_celeba.txt'
mark_path = r'D:\BaiduNetdiskDownload\Anno\list_landmarks_celeba.txt'
img_path = r'D:\BaiduNetdiskDownload\img_celeba'

save_path = '../data'
shuffle_index = list(range(0, 202599))
random.shuffle(shuffle_index)
shuffle_index = np.array(shuffle_index)
float_num = [0.1, 0.5, 0.5, 0.95, 0.95, 0.95, 0.99, 0.99, 0.99, 0.99]


def gen_sample(face_size, stop_value):
    pos_image_dir = os.path.join(save_path, str(face_size), 'positive')
    neg_image_dir = os.path.join(save_path, str(face_size), 'negative')
    part_image_dir = os.path.join(save_path, str(face_size), 'part')

    for path in [pos_image_dir, neg_image_dir, part_image_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    pos_label_filename = os.path.join(save_path, str(face_size), 'positive.txt')
    neg_label_filename = os.path.join(save_path, str(face_size), 'negative.txt')
    part_label_filename = os.path.join(save_path, str(face_size), 'part.txt')

    pos_count = 0
    neg_count = 0
    part_count = 0

    try:
        pos_label_file = open(pos_label_filename, 'w')
        neg_label_file = open(neg_label_filename, 'w')
        part_label_file = open(part_label_filename, 'w')

        shuffle_index = list(range(0, 202599))
        random.shuffle(shuffle_index)
        shuffle_index = np.array(shuffle_index)

        with open(box_path, 'r') as f:
            annos = f.readlines()
            annos = np.array(annos[2:])[shuffle_index]

        with open(mark_path, 'r') as f:
            mark = f.readlines()
            mark = np.array(mark[2:])[shuffle_index]

        for i, (box, mark) in enumerate(zip(annos, mark)):
            if i < 2:
                continue
            try:
                box = box.split()
                mark = mark.split()
                img_name = box[0].strip()
                img_file = os.path.join(img_path, img_name)
                image = Image.open(img_file).convert('RGB')
                img_w, img_h = image.size
                x1 = float(box[1].strip())
                y1 = float(box[2].strip())
                w = float(box[3].strip())
                h = float(box[4].strip()) * 0.9
                x2 = w + x1
                y2 = h + y1

                px1 = float(mark[1].strip())
                py1 = float(mark[2].strip())
                px2 = float(mark[3].strip())
                py2 = float(mark[4].strip())
                px3 = float(mark[5].strip())
                py3 = float(mark[6].strip())
                px4 = float(mark[7].strip())
                py4 = float(mark[8].strip())
                px5 = float(mark[9].strip())
                py5 = float(mark[10].strip())

                if max(w, h) < 40 or x1 < 0 or y1 < 0 or w < 0 or h < 0:
                    continue

                boxes = [[x1, y1, x2, y2]]

                cx = x1 + w / 2
                cy = y1 + h / 2
                side_len = max(w, h)
                seed = float_num[np.random.randint(0, len(float_num))]
                count = 0
                for _ in range(5):
                    _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))
                    _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))
                    _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))

                    _x1 = _cx - _side_len / 2
                    _y1 = _cy - _side_len / 2
                    _x2 = _x1 + _side_len
                    _y2 = _y1 + _side_len

                    if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:
                        continue

                    offset_x1 = (x1 - _x1) / _side_len
                    offset_y1 = (y1 - _y1) / _side_len
                    offset_x2 = (x2 - _x2) / _side_len
                    offset_y2 = (y2 - _y2) / _side_len

                    offset_px1 = (px1 - _x1) / _side_len
                    offset_py1 = (py1 - _y1) / _side_len
                    offset_px2 = (px2 - _x1) / _side_len
                    offset_py2 = (py2 - _y1) / _side_len
                    offset_px3 = (px3 - _x1) / _side_len
                    offset_py3 = (py3 - _y1) / _side_len
                    offset_px4 = (px4 - _x1) / _side_len
                    offset_py4 = (py4 - _y1) / _side_len
                    offset_px5 = (px5 - _x1) / _side_len
                    offset_py5 = (py5 - _y1) / _side_len

                    crop_box = [_x1, _y1, _x2, _y2]
                    face = image.crop(crop_box)
                    face_resize = face.resize((face_size, face_size))

                    iou = box_iou(np.array([crop_box]), np.array(boxes))[0]
                    if iou > 0.65:
                        pos_label_file.write(
                            "positive/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                pos_count, 1, offset_x1, offset_y1,
                                offset_x2, offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        face_resize.save(os.path.join(pos_image_dir, "{0}.jpg".format(pos_count)))
                        pos_count += 1
                    elif 0.6 > iou > 0.4:
                        part_label_file.write(
                            "part/{0}.jpg {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15}\n".format(
                                part_count, 2, offset_x1, offset_y1, offset_x2,
                                offset_y2, offset_px1, offset_py1, offset_px2, offset_py2, offset_px3,
                                offset_py3, offset_px4, offset_py4, offset_px5, offset_py5))
                        part_label_file.flush()
                        face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                        part_count += 1
                    elif iou < 0.2:
                        neg_label_file.write(
                            "negative/{0}.jpg {1} 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n".format(neg_count, 0))
                        neg_label_file.flush()
                        face_resize.save(os.path.join(neg_image_dir, "{0}.jpg".format(neg_count)))
                        neg_count += 1

                    count = pos_count + part_count + neg_count
                if count >= stop_value:
                    break
            except:
                traceback.print_exc()
    except:
        traceback.print_exc()


gen_sample(12, 75000)
gen_sample(24, 75000)
gen_sample(48, 75000)
