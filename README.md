# DCDH: Deep Hashing for Semantics-preserving Face Image Retrieval
This repository contains the source codes of our paper submitted to ECCV20: **Deep Center-Based Dual-Constrained Hashing for Semantics-Preserving Face Image Retrieval** implemented by PyTorch. 
# Introduction
We propose a novel center-based deep hashing framework, which ensures the intra-class samples to be closer to the corresponding class center than to other centers in Hamming space. The dual-constraint of the framework jointly minimizes the Hamming distance from intra-class samples to the corresponding class center while maximizing the Hamming distance between pairwise centers. Besides, we apply a regression term connecting labels and binary codes, which further contributes to discriminative hashing learning. Experiments on three large-scale datasets show that the proposed method outperforms state-of-the-art methods under various compared code lengths and several commonly-used evaluation metrics. 
The illustration of proposed DCDH framework is shown as following:
![method illustration](/images/figure1_1.png)
# Prerequisites
- Python >=3.6
- Pytorch >=1.2
# Data
We directly use the same data of FaceScrub and YouTube Faces datasets as released by [DDH](https://github.com/xjcvip007/DDH). For your convenience:
- YouTube Faces

 Training: https://drive.google.com/open?id=1Of-hUKhQhk3OtCUAZc15pZqEvKHWNUpN
 
 Test: https://drive.google.com/open?id=10RuxZuIMfvPN6ziPglJTgS6IWFBzW3pW
 - FaceScrub
 
 Training: https://drive.google.com/open?id=1STytImOvad3PgPN2NwUSVga-fOU5i2Tm
 
 Test: https://drive.google.com/open?id=1saSeHjaJBSAUNlyC-Ismdh2VbEiyZPuU
 
 - Subset of VGGFace2:
 
 The processed data of the subset of VGGFace2 is also provided here for download. 
 
 # Trained Model
 We release the trained model of two datasets: FaceScrub and YouTube Faces under four different code lengths {12, 24, 36, 48}, respectively. The model is named as dcdh_\[*dataset*]\[*length*]. 
