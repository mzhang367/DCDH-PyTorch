# DCDH: Class-wise labels-based deep supervised hashing method
This repository contains source codes of our paper: **Deep Center-Based Dual-Constrained Hashing for Discriminative Face Image Retrieval** implemented by PyTorch.

# Citation
If you find the codes are helpful to your research, please consider citing our PR paper:
```
@article{zhang2021deep,
  title={Deep Center-Based Dual-Constrained Hashing for Discriminative Face Image Retrieval},
  author={Ming Zhang and Xuefei Zhe and Shifeng Chen and Hong Yan},
  journal={Pattern Recognition},
  pages={to appear},
  year={2021},
  publisher={Elsevier}
}
```
# Developed by
**Ming Zhang<sup>1</sup> and Xuefei Zhe<sup>2</sup>**

<sup>1</sup> Department of Electrical Engineering, City University of Hong Kong 

<sup>2</sup> Tencent AI Lab

# Overview
![method illustration](/images/dcdh_framework.png)
<br>
Illustration of DCDH framework. ***θ***, ***M***, ***B***, and ***W*** represent the network parameters, class centers, binary hashing codes, and regression matrix, respectively. The proposed dual constraint on class centers aims to pull intra-class samples to the corresponding class center while pushing pairwise centers as far as possible. By ***W***, ***B*** is mutually determined by the hashing layer output and labels information. The arrows between every two modules show the information forward/backward-propagation.

# Prerequisites
- Python >=3.6
- PyTorch >=1.2
# Data
We directly use the same data of FaceScrub and YouTube Faces datasets as released by [DDH](https://github.com/xjcvip007/DDH). For your convenience:
- YouTube Faces

 Training: https://drive.google.com/open?id=1Of-hUKhQhk3OtCUAZc15pZqEvKHWNUpN
 
 Test: https://drive.google.com/open?id=10RuxZuIMfvPN6ziPglJTgS6IWFBzW3pW
 - FaceScrub
 
 Training: https://drive.google.com/open?id=1STytImOvad3PgPN2NwUSVga-fOU5i2Tm
 
 Test: https://drive.google.com/open?id=1saSeHjaJBSAUNlyC-Ismdh2VbEiyZPuU
 
 - Subset of VGGFace2:
 
 The processed data of the subset of VGGFace2 is also provided for your download: (total size ~2.4GB)
 
 https://drive.google.com/open?id=1PaXIfSrw1SC1uiv4Affgm882-Q-w_icl
 
 # Trained Model
 We release the trained models on two datasets: FaceScrub and YouTube Faces under four different code lengths {12, 24, 36, 48}, respectively. The models are named following the rule of "dcdh_\[*dataset*]\[*length*].tar" and can be downloaded [here](https://drive.google.com/open?id=152TYljUGI4tDdJJhtjDr9bVLYRxuDK5n).
 
 # Usage
 ## Training
 Assuming the dataset path `./FaceScrub` is under the same directory as the `dcdh_train.py`, to train the model by yourself, simply run the following command:
 ```
 python dcdh_train.py --dataset facescrub --save your_model_name.tar --len 48
 ```
This will train the 48-bit hashing model on the dataset FaceScrub with the default configurations of hyper-parameters, which are the same as written in the paper. The model will be saved under the folder `./checkpoint` in default. To see help for the training script:
 ```
 python dcdh_train.py -h
 ```
 
 ## Evaluation
 We provide various of commonly-used evaulation metrics including mean average precision (mAP), precision and recall under Hamming distance 2 and the precision curve w.r.t. Hamming ranking. Suppose your model for evaluation is under the default folder `./checkpoint`, simply execute:
 ```
 python evaluation.py --load model_name_on_youtube_12bit.tar --dataset youtube --len 12
 ```
 
 # Related Projects
 - Feature Learning based Deep Supervised Hashing with Pairwise Labels [(DPSH)](https://github.com/jiangqy/DPSH-pytorch)
 - Discriminative Deep Hashing for Face Image Retrieval [(DDH)](https://github.com/xjcvip007/DDH)
 - Deep Supervised Discrete Hashing [(DSDH)](https://github.com/liqi-casia/DSDH-HashingCode)
 - Discriminative Deep Attention-Aware Hashing for Face Image Retrieval [(DDAH)](https://github.com/deephashface/DDAH)
