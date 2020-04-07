from modeling.layers.ONet import *
from trainer import *

if __name__ == '__main__':
    net = ONet()
    trainer = Trainer(net, data_path='data/48', save_path='weights/onet.pth', batch_size=512)
    trainer(stop_value=0.001, net='onet')
