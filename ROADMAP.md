# ROADMAP

Provides general information on notebook content.

## CONTENT

* dc_step_explanation.ipynb: Illustrates why the additional step introduced in Fedor's paper 
is neccessary alongside the EGM algorithm in the presense of discrete choices in the model.
The need for the DC step is illustrated based on a model parametrisation that include basically no
(minimal) taste and income shocks. This is important since the presense of shocks leads to 
smoothing out of the value function, such that the DC step often time is not needed any more.
* retirement_model.ipynb: Exemplifies the implementation of the DC step line-by-line, based on the
same minimal shocks parametrisation and the same period that is used in the notebook above.
* matlab folder: contains the Matlab code from Fedor's repository + one important added file,
retirement_minimal_shocks.m. This file represents the corresponding line-by-line implementation
of retirement_model.ipynb in MatLab. Looking at both at the same time is extremely useful for 
understanding the purpose of each line, and verification that not only final, but also
intermediate results are the same.

## GOALS

* Reproduce MatLab solution for model parametrisation m5 and model with minimal shocks,
m0.
* Simulate based on both m5 and m0 and plot.