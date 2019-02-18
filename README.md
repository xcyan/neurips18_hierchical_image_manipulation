# Learning Hierarchical Semantic Image Manipulation through Structured Representations

This is official Pytorh implementation of NeurIPS 2018 paper [Learning Hierarchical Semantic Image Manipulation through Structured Representations](https://arxiv.org/abs/1808.07535) by Seunghoon Hong, Xinchen Yan, Thomas Huang, Honglak Lee.

<img src="https://aa501f67-a-62cb3a1a-s-sites.googlegroups.com/site/hierarchicalimagemanipulation/home/Figure_intro_horizontal.png" width="800px" height="300px"/>

Please follow the instructions to run the code.

## Prerequisites
* Mac OS X or Linux
* NVIDIA GPU (make sure your GPU has 12G+ memory) + CUDA cuDNN

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
* Please run the following script that creates two folders ```checkpoints/``` and ```datasets/```.
  ```
  bash setup.sh
  ```
* Please download the Cityscapes dataset from the [official website](https://www.cityscapes-dataset.com/) (registration required). After downloading, please put it under the ```datasets``` folder and run the following script.
  ```
  python preprocess_city.py
  ```

TBD

## Inference using a Pre-trained Box-to-Layout Generator
* You can download the pre-trained box-to-layout models, please run the following scripts.
  ```
  bash scripts/download_pretrained_box2mask_city.sh
  bash scripts/download_pretrained_box2mask_ade.sh
  ```
* Now, let us generate the manipulated layout from the pre-trained models. Please check the synthesized layouts under ```checkpoints/```.
  ```
  bash scripts/test_pretrained_box2mask_city.sh
  ```

## Inference using a Pre-trained Layout-to-Image Generator
* You can download the pre-trained layout-to-image models, please run the following scripts.
  ```
  bash scripts/download_pretrained_mask2image_city.sh
  bash scripts/download_pretrained_mask2image_ade.sh
  ```
* Now, let us generate the manipulated image from the pre-trained models. Please check the synthesized images under ```checkpoints/```.
  ```
  bash scripts/test_pretrained_mask2image_city.sh
  ```

## Joint Inference
<img src="https://aa501f67-a-62cb3a1a-s-sites.googlegroups.com/site/hierarchicalimagemanipulation/home/Figure_adaptive_generation_selected_7cols_bbox.png" width="800px" height="250px"/>

TBD


## Training Box-to-Layout Generator
* If you want to train the box-to-layout generator on Cityscape dataset, please run the following script (usually it takes a few hours using one GPU).
  ```
  bash scripts/train_box2mask_city.sh
  ```

## Training Layout-to-Image Generator
* If you want to train the layout-to-image generator on Cityscape dataset, please run the following script (usually it takes one day using one GPU).
  ```
  bash scripts/train_mask2image_city.sh
  ```
  
## Citation
If you find this useful, please cite our work as follows:
```
@inproceedings{hong2018learning,
  title={Learning hierarchical semantic image manipulation through structured representations},
  author={Hong, Seunghoon and Yan, Xinchen and Huang, Thomas S and Lee, Honglak},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2713--2723},
  year={2018}
}
```

## Acknowledgements
We would like to thank the amazing developers and the open-sourcing community. Our implementation has especially been benefited from the following excellent repositories:
TBD
