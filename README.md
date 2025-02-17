# DSMA-Net

This repository is DSMA-Net's attempt to classify thyroid nodules as benign or malignant.

he dataset contains 2898 ultrasound images, of which 1434 ultrasound images are malignant nodules and 1464 ultrasound images are benign nodules. Experiments can be performed using cross-validation.

We will upload the data as soon as the hospital dataset is publicly available.

# DSMA-Net Architecture
<div align="center">
  <img src="./picture/framework.png" width="600" height="350">
</div>
The architecture of our proposed DSMA-Net. It contains a shared encoder backbone for feature extraction and two independent decoders for classification and segmentation.

<div align="center">
  <img src="./picture/dense_se_block.png" width="300" height="300">
  <img src="./picture/aspp.png" width="250" height="250">
</div>
Dense SE Blocks and ASPP Structures in DSMA-Net architecture.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Usage
Use `python train.py` for training and then `python test.py` for testing.

If you want to compare, use `python train_compare.py` for training and then `python test_compare.py` for testing.

## Citing
If you find this work useful in your research, please consider citing the following papers:
```BibTex
@inproceedings{xing2024multi,
  title={A multi-task model for reliable classification of thyroid nodules in ultrasound images},
  author={Xing, Guangxin and Miao, Zhengqing and Zheng, Yelong and Zhao, Meirong},
  journal={Biomedical Engineering Letters},
  volume={14},
  number={2},
  pages={187--197},
  year={2024},
  doi={10.1007/s13534-023-00325-4},
  publisher={Springer}
}
```
