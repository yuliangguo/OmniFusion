# OmniFusion

This repository hosts the original implementation of CVPR 2022 (oral) paper "OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion
". [ArXiv link](https://arxiv.org/abs/2203.00838)


![Screen Shot 2022-05-17 at 11 08 12 PM](https://user-images.githubusercontent.com/13290379/168969292-ac4b09f1-23ff-4ea2-86c9-9378cf0f2833.png)

![Screen Shot 2022-05-17 at 11 09 34 PM](https://user-images.githubusercontent.com/13290379/168969309-60e38f9b-7881-4e8d-89f8-7994dca624d4.png)


## Citation
If you found our code helful for your own research, please cite our paper as:

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

## Getting Started
#### Requirements
- Anaconda (tested on 4.9.2)
- Python (tested on 3.7.4)
- PyTorch (tested on 1.9.0)
- torchvision (tested on 0.10.0)
- CUDA (tested on 11.6)
- Other dependencies

## Datasets
We trained and evaluated our method on three datasets [Stanford2D3D](http://buildingparser.stanford.edu/dataset.html), 
[360D](https://vcl3d.github.io/3D60/), [Matterport3D](https://niessner.github.io/Matterport/).

## Pre-trained models

Here we provide our [pre-trained models](https://drive.google.com/drive/folders/1b6mZJhF3j914AZ6TOGXrqgtGcHzHUAOc?usp=sharing) for direct evaluation.

##  Training
```
python train_erp_depth.py --fov 80 --patchsize (256, 256) --nrows 4
```
You can specify the patch fov, patch resolution, patch alignment(3, 4, 5, 6 rows).

## Evaluation
You can run the evaluation code to reproduce the results reported in our paper.
```
python test.py --fov 80 --patchsize (256, 256) --nrows 4
```

![Screen Shot 2022-05-17 at 11 14 50 PM](https://user-images.githubusercontent.com/13290379/168969991-afd0d9c5-cd18-4dda-8eaa-5ff597a6cbc2.png)


## Visualization
For training, visual results will be saved into tensorboard. In addition, depth maps and point clouds will be saved locally during training/evaluation. Some samples of the visual results can be seen below.

![qualitative](https://user-images.githubusercontent.com/13631958/159186337-b66d141c-71f5-40ec-a8ed-22353521f6d4.jpg)

