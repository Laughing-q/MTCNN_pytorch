# MTCNN_pytorch
The difference from the paper is adding batch normalize after convolution.

## prerequisites

* Python 3.6+
* PyTorch 1.3+
* Torchvision 0.4.0+ (**need high version because Torchvision support nms now.**)
* requirements.txt 

## Datasets
CeleA.
### Generate data
```Shell
python gen_data.py 
# just change the dataset(CeleA) directory to your own in code then run it
```
## Training
```Shell
python train_pnet.py
python train_rnet.py
python train_onet.py
```
## Demo:
```Shell
python demo.py 
```
Output: 
![](https://github.com/Laughing-q/MTCNN_pytorch/blob/master/output/1.jpg) 
![](https://github.com/Laughing-q/MTCNN_pytorch/blob/master/output/multiface.jpg) 
![](https://github.com/Laughing-q/MTCNN_pytorch/blob/master/output/J2.jpg)  
![](https://github.com/Laughing-q/MTCNN_pytorch/blob/master/output/SPN.jpg)  
![](https://github.com/Laughing-q/MTCNN_pytorch/blob/master/output/original.jpg) 
![](https://github.com/Laughing-q/MTCNN_pytorch/blob/master/output/timg.jpg)
