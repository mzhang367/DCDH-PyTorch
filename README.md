# Deep Center-Based Dual-Constrained Hashing for Discriminative Face Image Retrieval (DCDH)
This repository contains the PyTorch implementation of our paper: [Deep Center-Based Dual-Constrained Hashing for Discriminative Face Image Retrieval](https://www.sciencedirect.com/science/article/pii/S0031320321001631) by Ming Zhang<sup>a</sup>, Xuefei Zhe<sup>b</sup>, Shifeng Chen<sup>c</sup> and Hong Yan<sup>a</sup>.

<sup>a</sup> Department of Electrical Engineering, City University of Hong Kong 

<sup>b</sup> Tencent AI Lab

<sup>c</sup> Shenzhen Institutes of Advanced Technology, CAS

# Citation
If you find the codes are useful to your research, please consider citing our PR paper:
```
@article{zhang2021deep,
title = {Deep center-based dual-constrained hashing for discriminative face image retrieval},
author = {Ming Zhang and Xuefei Zhe and Shifeng Chen and Hong Yan},
journal = {Pattern Recognition},
volume = {117},
pages = {107976},
year = {2021},
}
```
# Introduction
We propose a novel center-based framework integrating end-to-end hashing learning and class centers learning simultaneously. Unlike most existing works that are either based on pairwise/triplet labels \[1, 2, 3\] or softmax classification loss \[4, 5\], we apply a class-wise similarity as the supervision. With a normalized Gaussian-based loss function, the first constraint of DCDH minimizes the intra-class variance by clustering intra-class samples into a learnable class center. And the second constraint serves as a regularization term to enlarge the Hamming distance between pairwise class centers for inter-class separability. Furthermore, we introduce a regression matrix to encourage intra-class samples to generate the same binary codes for hashing codes compactness.

# Overview
<img src="/images/dcdh_framework.png" alt="drawing" width="75%"/>
<p></p>

Illustration of the proposed framework. ***θ***, ***M***, ***B***, and ***W*** represent the network parameters, class centers, binary hashing codes, and regression matrix, respectively. The proposed dual constraint on class centers aims to pull intra-class samples to the corresponding class center while pushing pairwise centers as far as possible. By ***W***, ***B*** is mutually determined by the hashing layer output and labels information. The arrows between every two modules show the information forward/backward-propagation.

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
 
 The processed data of the subset of VGGFace2 is also provided for your [download](https://drive.google.com/file/d/1OLtVgQQ59alSAsyQdqWZe0t_KiNbSUMm/view?usp=sharing) (total size ~2.4GB).
 
 # Models
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
 We provide various of commonly-used evaulation metrics including mean average precision (mAP), precision and recall under two Hamming distances and the precision curve w.r.t. Hamming ranking. Suppose your model is under the default folder `./checkpoint`, to evaluate, simply execute:
 ```
 python evaluation.py --load model_name_on_youtube_12bit.tar --dataset youtube --len 12
 ```
 
 # Related Projects
 \[1\] Feature Learning based Deep Supervised Hashing with Pairwise Labels [(DPSH)](https://github.com/jiangqy/DPSH-pytorch)
 
 \[2\] Deep Supervised Hashing with Triplet Labels [(DTSH)](https://github.com/Minione/DTSH)
 
 \[3\] Deep Supervised Discrete Hashing [(DSDH)](https://github.com/liqi-casia/DSDH-HashingCode)
 
 \[4\] Discriminative Deep Hashing for Face Image Retrieval [(DDH)](https://github.com/xjcvip007/DDH)
 
 \[5\] Discriminative Deep Attention-Aware Hashing for Face Image Retrieval [(DDAH)](https://github.com/deephashface/DDAH)
 
 We also thank the above authors for releasing their codes.
