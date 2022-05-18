# OmniFusion

This repository hosts the original implementation of CVPR 2022 (oral) paper "OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion
". [ArXiv link](https://arxiv.org/abs/2203.00838)


![Screen Shot 2022-05-17 at 11 08 12 PM](https://user-images.githubusercontent.com/13290379/168969292-ac4b09f1-23ff-4ea2-86c9-9378cf0f2833.png)

![Screen Shot 2022-05-17 at 11 09 34 PM](https://user-images.githubusercontent.com/13290379/168969309-60e38f9b-7881-4e8d-89f8-7994dca624d4.png)


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
The evaluation code is expected reproduce the experimental results reported in our paper.

![Screen Shot 2022-05-17 at 11 14 50 PM](https://user-images.githubusercontent.com/13290379/168969991-afd0d9c5-cd18-4dda-8eaa-5ff597a6cbc2.png)


## Visualization
For training, visual results will be saved into tensorboard. In addition, depth maps and point clouds will be saved locally during training/evaluation. Some samples of the visual results can be seen below.

![Screen Shot 2022-05-17 at 11 31 02 PM](https://user-images.githubusercontent.com/13290379/168973637-ab76eeaf-dc5c-4c43-8037-ae80aedb3462.png)

The visual comparisons to the prior arts can be observed in the following two samples.
![Screen Shot 2022-05-17 at 11 49 21 PM](https://user-images.githubusercontent.com/13290379/168975739-0d6fd8e4-b88c-4e39-aae2-c5667cc3804c.png)
![Screen Shot 2022-05-17 at 11 49 46 PM](https://user-images.githubusercontent.com/13290379/168975755-080838a1-7ea9-490d-a765-624d857f2033.png)

As observed, our method recovers more accurate and structural depth maps, which appear sharper on the edges, and smoother within the surface.


