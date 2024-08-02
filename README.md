# HFCNet: Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images

Welcome to the official repository for the paper "Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images", IEEE TGRS, 2024. 

### The Initialization Weights for Training
Download pre-trained classification weights of the [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [VGG](https://download.pytorch.org/models/vgg16-397923af.pth), and place the ` .pth ` files in ` ./pretrained ` directory. These weights are essential for initializing the model during training.

### Trained Weights of HFCNet for Testing

coming soon.

### Train
Please download the pre-trained model weights and dataset first. Next, generate the path lists of the training set and the test set, and change the dataset path in the code to the path of the dataset listing file (.txt) you specified.

~~~python
nohup python -u main.py --flag train --model_id HFCNet --config config/cod_mgl50_o.yaml --device cuda:0 > train_ORSSD.log &

nohup python -u main.py --flag train --model_id HFCNet --config config/cod_mgl50_e.yaml --device cuda:0 > train_EORSSD.log &

nohup python -u main.py --flag train --model_id HFCNet --config config/cod_mgl50_orsi.yaml --device cuda:0 > train_ORSI.log &
~~~

### Test
Download the HFCNet model weights, create the necessary directories to store these files, and be sure to update the corresponding paths in the code accordingly. 

~~~python
mkdir ./modelPTH-ORSSD
python -u main.py --flag test --model_id HFCNet --config config/cod_mgl50_o.yaml

mkdir ./modelPTH-EORSSD
python -u main.py --flag test --model_id HFCNet --config config/cod_mgl50_e.yaml 

mkdir ./modelPTH-ORSI
python -u main.py --flag test --model_id HFCNet --config config/cod_mgl50_orsi.yaml
~~~

### Citation

If it helps your research,  please use the information below to cite our work, thank you. 
