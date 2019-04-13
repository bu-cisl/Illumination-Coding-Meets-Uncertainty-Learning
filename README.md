# Illumination-Coding-Meets-Uncertainty-Learning
Python (Keras) implementation of paper: **Reliable deep learning-based phase imaging with uncertainty quantification**. We provide model, pre-trained weight, test data and a quick demo.


### Citation
If you find this project useful in your research, please consider citing our paper:

[**Yujia Xue, Shiyi Cheng, Yunzhe Li, and Lei Tian, "Reliable deep learning-based phase imaging with uncertainty quantification", ariXv preprint ](https://arxiv.org/abs/1901.02038)


### Abstract
Emerging deep learning (DL) based techniques have significant potential to revolutionize biomedical imaging. However, one outstanding challenge is the lack of reliability assessment in the DL predictions, whose errors are commonly revealed only in hindsight. Here, we propose a new Bayesian convolutional neural network (BNN) based framework that overcomes this issue by quantifying DL prediction uncertainty. Foremost, we show that BNN predicted uncertainty maps provide surrogate estimates of the true error from the network model and measurement itself. The uncertainty maps characterize imperfections often unknown in realworld applications, such as noise, model error, incomplete training data, and out-of-distribution testing data. Quantifying this uncertainty provides a per-pixel estimate of the DL prediction’s confidence level as well as the quality of the model and dataset. We demonstrate this framework in the application of large space-bandwidth product phase imaging using a physics-guided coded illumination scheme. From only five multiplexed illumination measurements, our BNN predicts gigapixel phase images in both static and dynamic biological samples with quantitative credibility assessment. Furthermore, we show that low-certainty regions can identify spatially and temporally rare biological phenomena. We believe our uncertainty learning framework is widely applicable to many DL-based biomedical imaging techniques for assessing the reliability
of DL predictions.



### Requirements
python 3.6

keras 2.1.2

tensorflow 1.4.0

numpy 1.14.3

h5py 2.7.1

matplotlib 2.1.2


### AI-augmented Phase Imaging and Uncertainty Learning Framework
<p align="center">
  <img src="/figs/overview.png">
</p>

### Network Structure
<p align="center">
  <img src="/figs/network.png">
</p>


### How to run the demo
Make sure you have all dependent packages installed correctly and run [demo.py](demo.py).


### Results
## On SEEN cell type
<p align="center">
  <img src="/figs/seen_cell_type.png">
</p>

## On UNSEEN cell type
<p align="center">
  <img src="/figs/unseen_cell_type.png">
</p>

## Uncertainty learning framework identifies spatially and temporally rare biological phenomena.
<p align="center">
  <img src="/figs/video.png">
</p>


## License
This project is licensed under the terms of the BSD-3-Clause license. see the [LICENSE](LICENSE) file for details
