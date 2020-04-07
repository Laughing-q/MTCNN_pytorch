from torch import nn
import torch.optim as optim
from utils.datasets import *
import torch


class Trainer:
    def __init__(self, net, data_path, save_path, batch_size=512):
        super(Trainer, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.batch_size = batch_size
        self.net = net.to(self.device)
        self.datasets = FaceDatasets(data_path)
        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.net.parameters())
        self.conf_fn = nn.BCELoss()
        self.box_fn = nn.MSELoss()
        self.mark_fn = nn.MSELoss()
        self.save_path = save_path

        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))

    def __call__(self, stop_value, net=None):
        dataloader = DataLoader(dataset=self.datasets, batch_size=self.batch_size, shuffle=True, num_workers=2)
        loss = 0
        last_loss = 1000
        while True:
            for i, (img, conf, box, landmark) in enumerate(dataloader):
                img, conf, box, landmark = img.to(self.device), conf.to(self.device), box.to(self.device), landmark.to(
                    self.device)
                output = self.net(img)
                out_landmark = None
                if net == 'onet':
                    out_conf, out_box, out_landmark = output
                    out_landmark = out_landmark.view(-1, 10)
                else:
                    out_conf, out_box = output
                out_conf = out_conf.view(-1, 1)
                out_box = out_box.view(-1, 4)
                # 计算分类的损失
                # eq:等于，lt:小于，gt:大于，le:小于等于，ge:大于等于
                conf_mask = torch.lt(conf, 2)  # 得到分类标签小于2的布尔值，a<2,[0,1,2]-->[1,1,0]
                conf_ = torch.masked_select(conf, conf_mask)  # 通过掩码，得到符合条件的置信度标签值
                out_conf_ = torch.masked_select(out_conf, conf_mask)
                conf_loss = self.conf_fn(out_conf_, conf_)

                box_mask = torch.gt(conf, 0)
                box_ = torch.masked_select(box, box_mask)
                out_box_ = torch.masked_select(out_box, box_mask)
                box_loss = self.box_fn(out_box_, box_)

                loss = conf_loss + box_loss

                if net == 'onet':
                    mark_mask = box_mask
                    landmark_ = torch.masked_select(landmark, mark_mask)
                    out_landmark_ = torch.masked_select(out_landmark, mark_mask)
                    mark_loss = self.mark_fn(out_landmark_, landmark_)
                    loss = loss + mark_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                conf_loss = conf_loss.cpu().item()
                box_loss = box_loss.cpu().item()
                loss = loss.cpu().item()
                if net == 'onet':
                    mark_loss = mark_loss.cpu().item()
                    print(" loss:", loss, " conf_loss:", conf_loss, " box_loss", box_loss, 'landmark_loss', mark_loss)
                else:
                    print(" loss:", loss, " conf_loss:", conf_loss, " box_loss", box_loss)

                # if loss < last_loss:
            torch.save(self.net.state_dict(), self.save_path)
            print("save success")
            # last_loss = loss

            if loss < stop_value:
                break
