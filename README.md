# SNUNet-CD
The pytorch implementation for "SNUNet-CD: A Densely  Connected Siamese Network for Change Detection of VHR Images " (coming soon)

## Requirements

- Python 3.6

- Pytorch 1.4

- torchvision 0.5.0

```
# other packages needed
pip install opencv-python tqdm tensorboardX sklearn
```

## Dataset

- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) (Change Detection Dataset)
- paper: [Change detection in remote sensing images using conditional adversarial networks](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLII-2/565/2018/isprs-archives-XLII-2-565-2018.pdf)

## Train from scratch

    python train.py

## Evaluate model performance

    python eval.py

## Visualization

    python visualization.py

## Pre-trained models

The pre-trained models in CDD dataset are available. 

[google driver](https://drive.google.com/drive/folders/1_aoUvMC8zWy4Pv7vU_BHu5yzzjgIsNXv?usp=sharing) | [baidu disk](https://pan.baidu.com/s/1i8LeBdMyR5jTWPTTB1XZUA) (w90j)



------

The original version of "Siam-NestedUNet" is archived in Tag v1.1 .



## References

Appreciate the work from the following repository:

- [granularai / chip_segmentation_fabric](https://github.com/granularai/chip_segmentation_fabric)

