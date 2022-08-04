# EgPDE-net
The official implementation of the paper "EgPDE-Net Building Continuous Neural Networks for Time Series Prediction with Exogenous Variables",[ArXiv](https://arxiv.org/abs/2208.01913)

While exogenous variables have a major impact on performance improvement in time series analysis, inter-series correlation and time dependence among them are rarely considered in the present continuous methods. The dynamical systems of multivariate time series could be modelled with complex unknown partial differential equations (PDEs) which play a prominent role in many disciplines of science and engineering. In this paper, we propose a continuous-time model for arbitrary-step prediction to learn an unknown PDE system in multivariate time series whose governing equations are parameterised by self-attention and gated recurrent neural networks. The proposed model, \underline{E}xogenous-\underline{g}uided \underline{P}artial \underline{D}ifferential \underline{E}quation Network (EgPDE-Net), takes account of the relationships among the exogenous variables and their effects on the target series. Importantly, the model can be reduced into a regularised ordinary differential equation (ODE) problem with special designed regularisation guidance, which makes the PDE problem tractable to obtain numerical solutions and feasible to predict multiple future values of the target series at arbitrary time points.

## 1. Requirements
PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 10.2;
torchvision;
Install torchdiffeq from https://github.com/rtqichen/torchdiffeq.

## 2. Running for different datasets
python two_ode_selfatt.py -d datasetname


## Citation
If you find the project is helpful, please cite us as follows.
```
@article{,
  title={EgPDE-Net Building Continuous Neural Networks for Time Series Prediction with Exogenous Variables},
  author={},
  journal={},
  year={2022}
}
```
