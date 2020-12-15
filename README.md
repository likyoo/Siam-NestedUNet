# SNUNet-CD
The pytorch implementation for "SNUNet-CD: A Densely  Connected Siamese Network for Change Detection of VHR Images " (coming soon)

## Requirements

- Python 3.7

- Pytorch 1.0+

- torchvision 0.2.1+

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

## visualization

    python visualization.py

## References

Appreciate the work from the following repository:

- [granularai / chip_segmentation_fabric](https://github.com/granularai/chip_segmentation_fabric)

