# Illumination-Coding-Meets-Uncertainty-Learning
Python (Keras) implementation of paper: **Illumination coding meets uncertainty learning: toward reliable AI-augmented phase imaging**. We provide model, pre-trained weight, test data and a quick demo.


### Citation
If you find this project useful in your research, please consider citing our paper:

[**Yujia Xue, Shiyi Cheng, Yunzhe Li, and Lei Tian, "Illumination coding meets uncertainty learning: toward reliable AI-augmented phase imaging"](https://arxiv.org/abs/1901.02038)


### Abstract
We develop a new Bayesian convolutional neural network (BNN) based technique for achieving large space-bandwidth product phase imaging that is both {\it scalable} and {\it reliable}.The scalability of our technique is enabled by a novel coded illumination scheme designed by the physical principles of asymmetric illumination-based phase contrast and synthetic aperture imaging. The system takes highly multiplexed intensity measurements to encode phase and high resolution information across a wide field-of-view (FOV).The inversion requires solving a highly ill-posed phase retrieval problem, which we show can be overcome by a deep learning (DL) algorithm. The reliability of our technique is quantitatively assessed by a novel uncertainty learning framework. Differing from existing DL-based reconstruction algorithms whose prediction errors can only be  discovered  in hindsight, our BNN framework allows uncertainty quantification of the DL predictions. Specifically, we show that the BNN predicted uncertainty maps can be used as  surrogates to the true error, which is typically unknown in many real-world applications. Furthermore, we complement the BNN with a statistical data analysis procedure that relate the network outputs to credibility quantification metrics.We apply our technique to both static and dynamic biological samples, and show that the illumination scheme allows achieving 5$\times$ resolution enhancement across a 4$\times$ FOV using only five multiplexed measurements.In addition, we show that the uncertainty quantification procedure allows evaluating the effects of several common experimental imperfections, including noise, model errors, incomplete training data, and out-of-distribution testing data. Finally, we illustrate the utility of the predicted uncertainty maps as a possible way to identify spatially and temporally rare biological phenomena.




### Requirements
python 3.6

keras 2.1.2

tensorflow 1.4.0

numpy 1.14.3

h5py 2.7.1

matplotlib 2.1.2


### Uncertainty Learning Framework
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
  <img src="/figs/usseen_cell_type.png">
</p>
## Uncertainty learning framework identifies spatially and temporally rare biological phenomena.
<p align="center">
  <img src="/figs/video.png">
</p>


## License
This project is licensed under the terms of the BSD-3-Clause license. see the [LICENSE](LICENSE) file for details
