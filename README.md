# Learning Hierarchical Semantic Image Manipulation through Structured Representations

This is official Pytorh implementation of NeurIPS 2018 paper [Learning Hierarchical Semantic Image Manipulation through Structured Representations](https://arxiv.org/abs/1808.07535) by Seunghoon Hong, Xinchen Yan, Thomas Huang, Honglak Lee.

<img src="https://aa501f67-a-62cb3a1a-s-sites.googlegroups.com/site/hierarchicalimagemanipulation/home/Figure_intro_horizontal.png" width="800px" height="300px"/>

Please follow the instructions to run the code.

## Requirements
MT-VAE requires or works with
* Mac OS X or Linux
* NVIDIA GPU (make sure your GPU has 12G+ memory)

## Installing Dependencies
* Install [Pytorch](https://pytorch.org/)
  * Note: This implementation has been tested with [Pytorch 0.3.1](https://pytorch.org/get-started/previous-versions/).
  ```
  conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
  ```
* Install [TensorFlow](https://www.tensorflow.org/)
  * Note: This implementation has been tested with [TensorFlow 1.5](https://www.tensorflow.org/versions).
  ```
  pip install tensorflow-gpu==1.5
  ```
* Install [Python Dominate Library](https://pypi.org/project/dominate/)
  ```
  pip install dominate
  ```

## Data Preprocessing
TBD

## Training (Box-to-Layout Generator)
* If you want to train the box-to-layout generator on Cityscape dataset, please run the following script (usually it takes a few hours using one GPU).
  ```
  bash scripts/train_box2mask_city.sh
  ```
* Alternatively, you can download the pre-trained box-to-layout model, please run the following script.

## Training (Layout-to-Image Generator)
* If you want to train the layout-to-image generator on Cityscape dataset, please run the following script (usually it takes one day using one GPU).
  ```
  bash scripts/train_mask2image_city.sh
  ```
* Alternatively, you can download the pre-trained box-to-layout model, please run the following script.

## Inference (Box-to-Layout Generator)
TBD

## Inference (Layout-to-Image Generator)
TBD

## Joint Inference
TBD
