from modeling.layers.PNet import *
from trainer import *

if __name__ == '__main__':
    net = PNet()
    trainer = Trainer(net, data_path='data/12', save_path='weights/pnet.pth', batch_size=512)
    trainer(stop_value=0.01)
