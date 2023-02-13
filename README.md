# OmniFusion

This repository hosts the original implementation of CVPR 2022 (oral) paper "OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion
". [ArXiv link](https://arxiv.org/abs/2203.00838)


![Github_gif_prepare2](https://user-images.githubusercontent.com/13290379/170804206-4f31f70a-35a4-4c6e-ad4f-7052041f7dd5.gif)



## Citation
If you found our code helpful for your research, please cite our paper as:

```
@inproceedings{Li2022CVPR,
  Title      = {OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion},
  Author     = {Li, Yuyan and Guo, Yuliang and Yan, Zhixin and Huang, Xinyu and Ye, Duan and Ren, Liu},
  Booktitle  = {2022 Conference on Computer Vision and Pattern Recognition (CVPR)},
  Year       = {2022},
  Address    = {New Orleans, USA},
  Month      = jun,
}
```

## Pipeline

![Github_gif_prepare](https://user-images.githubusercontent.com/13290379/170779589-a9061c75-7c00-4e61-883a-c5083b620893.gif)


## Getting Started
#### Requirements
- Anaconda (tested on 4.9.2)
- Python (tested on 3.7.4)
- PyTorch (tested on 1.9.0)
- torchvision (tested on 0.10.0)
- CUDA (tested on 11.6)
- Other dependencies

```bash
git clone https://github.com/yuyanli0831/OmniFusion
cd OmniFusion
python3 -m venv omni-venv
source omni-venv/bin/activate
pip3 install -r requirements.txt
```

## Datasets
We trained and evaluated our method on three datasets [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html), 
[360D](https://vcl3d.github.io/3D60/), [Matterport3D](https://niessner.github.io/Matterport/).

FYI, a standard pre-process of the original Matterport3D dataset is applied for the development of 360-image-based depth estimation. Given the downloaded original dataset, the 360 images and depth maps are rendered by a matlab scrpt included in this [repo](https://github.com/alibaba/UniFuse-Unidirectional-Fusion/blob/main/UniFuse/Matterport3D).

## Pre-trained models

Our [pre-trained models](https://drive.google.com/drive/folders/1b6mZJhF3j914AZ6TOGXrqgtGcHzHUAOc?usp=sharing) are provided for direct evaluation.

##  Training
```
python train_erp_depth.py --fov 80 --patchsize (256, 256) --nrows 4
```
You can specify the patch fov, patch resolution, patch alignment(3, 4, 5, 6 rows).

## Evaluation
```
python test.py --fov 80 --patchsize (256, 256) --nrows 4
```

## Visualization
For training, visual results will be saved into tensorboard. In addition, depth maps and point clouds will be saved locally during training/evaluation.





