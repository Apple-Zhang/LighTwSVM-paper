function [accu, pred] = lightwsvm_predict(X, Y, model)
%LIGHTWSVM_PREDICT 
%   Input:
%       X: n*m testing sample matrix.
%       Y: n*1 label matrix. y_i \in 1, ..., numClass.
%   model: The model outputted by lightwsvm_train.
%
%   Output:
%       accu: accuracy in percentage
%       pred: the prediction label
%
%   By Junhong Zhang in 2024.10.15

n = size(X,1);
X = [X, ones(n,1)];

pred = X * model.w;
[~, pred] = min(abs(pred), [], 2);
pred = model.labelName(pred);

accu = mean(pred(:) == Y(:)) * 100;

end