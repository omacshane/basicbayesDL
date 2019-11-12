# basicbayesDL
**Basic Bayesian Deep Learning**

These scripts present a basic implemention of a Bayesian Convolutional Deep Network in Keras based upon
"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2015)
by Yarin Gal and Zoubin Ghahramani [1].

The code is based on Yarin Gal's code examples:
https://github.com/yaringal/DropoutUncertaintyExps/blob/master/bostonHousing/net/net/net.py

These scripts can be used as a starting point for implementing deep learning models using dropout Byesian
approximation.

## Get Started

The jupyter [notebook](/notebooks/run_bbdl_experiment.ipynb) has provides a quick setup for running some experiments on the CIFAR-10 dataset, 
comparing MC Dropout to a standard CNN model.

![PRCurve](/notebooks/class8.png)


[1] https://arxiv.org/abs/1506.02142
