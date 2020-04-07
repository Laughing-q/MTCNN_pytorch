import os
from PIL import Image, ImageDraw
import random
import matplotlib.pyplot as plt
import numpy as np

box_path = r'D:\BaiduNetdiskDownload\Anno\list_bbox_celeba.txt'
mark_path = r'D:\BaiduNetdiskDownload\Anno\list_landmarks_celeba.txt'
img_path = r'D:\BaiduNetdiskDownload\img_celeba'

shuffle_index = list(range(0, 202599))
random.shuffle(shuffle_index)
shuffle_index = np.array(shuffle_index)

with open(box_path, 'r') as f:
    annos = f.readlines()
    annos = np.array(annos[2:])[shuffle_index]

with open(mark_path, 'r') as f:
    mark = f.readlines()
    mark = np.array(mark[2:])[shuffle_index]

for anno, mark in zip(annos, mark):
    # print(img)
    anno = anno.split()
    file_name = anno[0].strip()
    image = Image.open(os.path.join(img_path, file_name)).convert('RGB')
    box = [int(x.strip()) for x in anno[1:]]
    mark = mark.split()[1:]
    x = []
    y = []
    for i in range(0, 10, 2):
        # print(i)
        x.append(int(mark[i]))
        y.append(int(mark[i + 1]))
    plt.clf()
    ax = plt.gca()  # 获取到当前坐标轴信息
    ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
    ax.invert_yaxis()
    # print(x, y)
    plt.imshow(image)
    plt.scatter(x, y)
    ax.add_patch(plt.Rectangle((box[0], box[1]), box[2], box[3] * 0.9, fill=False, linewidth=2, color='red'))
    plt.pause(1)
