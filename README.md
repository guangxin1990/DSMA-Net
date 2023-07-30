# DSMA-Net

This repository is DSMA-Net's attempt to classify breast tumors as benign or malignant on a publicly available breast dataset.

We randomly divided the dataset into training, validation and test sets containing 504, 63 and 63 ultrasound images, respectively.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Usage
### 1
Download the  dataset [here](https://drive.google.com/file/d/11peQ9NXuPA-QNNA9pmzD8SbTXTfQz17_/view?usp=drive_link) and put it into `./data`.
### 2
Use `python train.py` for training and then `python test.py` for testing.
