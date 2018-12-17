# Illumination-Coding-Meets-Uncertainty-Learning
Python implementation of paper: **Illumination coding meets uncertainty learning: toward reliable AI-augmented phase imaging**. We provide model, pre-trained weights(download link available below), test data and a quick demo.


### Citation
If you find this project useful in your research, please consider citing our paper:

[**paper citation link**](link tbd)


### Abstract
Traditional phase imaging techniques are limited by the throughput due to the linear trade-off between field-of-view (FOV), resolution, and acquisition speed. Here, we show that a much expanded nonlinear trade space can be obtained by augmenting physical encoding with deep learning (DL).We propose a physics-assisted DL framework in which the coded patterns are designed based on physical principles and are well suited for efficient DL inference. This allows our imaging pipeline to be broadly applicable and is robust to setup variations and unknown experimental errors. Further, we demonstrate uncertainty learning (UL) that provides quantification of the intrinsic variabilities of the DL predictions. Typical DL algorithms tend to output overly confident predictions, whose errors are only discovered in hindsight. In contrast, UL outputs two types of predictive uncertainties to assess the confidence level of the predictions. The model uncertainty characterizes the intrinsic variability of the data acquisition and DL inference procedure. The data uncertainty evaluates the confidence level of the prediction subject to data noise and experimental errors. We experimentally demonstrate this framework reliably reconstructs phase with 5X resolution enhancement across a 4X FOV using only five multiplexed measurements – more than 10X data reduction over the state-of-the-art. We provide a comprehensive data analysis procedure to quantify the reliability of the DL predications. Our approach demonstrates a highly scalable AI-augmented high-SBP phase imaging technique with dependable predictions. The UL framework is widely applicable to scientific and biomedical imaging where critical assessment of results is essential.

<p align="center">
  <img src="/images/">
</p>


### Requirements
python 3.6

keras 2.1.2

tensorflow 1.4.0

numpy 1.14.3

h5py 2.7.1

matplotlib 2.1.2


### Uncertainty Learning Framework
<p align="center">
  <img src="/images/">
</p>


### Download pre-trained weights
You can download pre-trained weights from [link](tbd)


### How to use
After download the pre-trained weights file, put it under the root directory and run [demo.py](demo.py).


### Results
<p align="center">
  <img src="/images/">
</p>


## License
This project is licensed under the terms of the BSD-3-Clause license. see the [LICENSE](LICENSE) file for details
