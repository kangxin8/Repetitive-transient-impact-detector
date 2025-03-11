These codes correspond to Paper :Repetitive transient impact detection and its application in cross-machine bearing fault detection, which is Published in mssp.   
The code supports training and testing using our processed dataset, as well as the visualization of convolutional kernels and processed samples. For now, we will upload the processed dataset.  

Before running, you need to download the data first. Here is the link to the data on the cloud drive：[processed_data](链接：https://pan.baidu.com/s/1c-foOawQlX3r90Z_-sXLAA?pwd=gi11 
提取码：gi11). After that, modify the data path option in the main script's argument parser, and then simply run the main function to train the model.
We saved the trained model parameters, ready for direct use and visualization analysis as file of 16-1.0000-best_model.pth

This code references
@misc{Zhao2019,
author = {Zhibin Zhao and Qiyang Zhang and Xiaolei Yu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
title = {Unsupervised Deep Transfer Learning for Intelligent Fault Diagnosis},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ZhaoZhibin/UDTL}},
}.


