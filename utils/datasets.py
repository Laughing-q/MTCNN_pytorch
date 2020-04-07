import torch
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision import transforms


class FaceDatasets(Dataset):
    def __init__(self, path):
        super(FaceDatasets, self).__init__()
        self.path = path
        self.datasets = []
        self.datasets.extend(open(os.path.join(path, 'positive.txt')).readlines())
        self.datasets.extend(open(os.path.join(path, 'negative.txt')).readlines())
        self.datasets.extend(open(os.path.join(path, 'part.txt')).readlines())
        self.transfor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        data = self.datasets[item].split()
        img_path = data[0]
        img = Image.open(os.path.join(self.path, img_path)).convert('RGB')
        img = self.transfor(img)
        data[1:] = [float(x) for x in data[1:]]
        conf = torch.tensor(data[1:2], dtype=torch.float32)
        box = torch.tensor(data[2:6], dtype=torch.float32)
        landmark = torch.tensor(data[6:], dtype=torch.float32)
        return img, conf, box, landmark

    def __len__(self):
        return len(self.datasets)


if __name__ == '__main__':
    test = FaceDatasets('../data/12')
    print(test[0][0].shape)
    print(test[0][1])
    print(test[0][2])
    print(test[0][3])
    dataloader = DataLoader(dataset=test, batch_size=20, shuffle=True, num_workers=4)
    for i, (img, conf, box, landmark) in enumerate(dataloader):
        print(img.shape)
        print(conf.shape)
        print(box.shape)
        print(landmark.shape)
        break
