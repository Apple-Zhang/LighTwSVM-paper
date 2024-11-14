# LighTwSVM-paper
This repo contains the official code of *J. Zhang, Z. Lai, "LighTwSVM: Efficient Linear Nonparallel Classifier for Millions of Data"* (under review).

An efficient implementation of the linear TwSVM.

The requirement is Matlab.
If you are not 64-bit Windows user or your Matlab version is old (cannot use .mexw64 file),
run `mex onetwsvm.cpp` first to compile the solver.

Then you can run `demo.m` in your Matlab. 
