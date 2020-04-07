from modeling.layers.RNet import *
from trainer import *

if __name__ == '__main__':
    net = RNet()
    trainer = Trainer(net, data_path='data/24', save_path='weights/rnet.pth', batch_size=512)
    trainer(stop_value=0.001)
