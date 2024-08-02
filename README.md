# HFCNet: Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images

Welcome to the official repository for the paper "Heterogeneous Feature Collaboration Network for Salient Object Detection in Optical Remote Sensing Images", IEEE TGRS, 2024. [**[IEEE link](https://eff.org)**]

### The Initialization Weights for Training
Download pre-trained classification weights of the [Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) and [VGG](https://download.pytorch.org/models/vgg16-397923af.pth), and put the ` pth ` file in ` ./pretrained `. These weights are needed for training to initialize the model.

### Trained Weights of HFCNet for Testing



### Train

~~~python
nohup python -u main.py --flag train --model_id HFCNet --config config/cod_mgl50_o.yaml --device cuda:0 > train_ORSSD.log &

nohup python -u main.py --flag train --model_id HFCNet --config config/cod_mgl50_e.yaml --device cuda:0 > train_EORSSD.log &

nohup python -u main.py --flag train --model_id HFCNet --config config/cod_mgl50_orsi.yaml --device cuda:0 > train_ORSI.log &
~~~

### Test

Please download the HFCNet model weights and dataset first. Next, create the necessary directories to store these files, and be sure to update the corresponding paths in the code accordingly. 

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
