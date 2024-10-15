clear;
close;

% fix random seed for reproducibility
rng("default");

% load data
load data\dna.scale.mat;
nTotal = data_infm.nr_sample;
nTrain = round(0.6 * nTotal);

% randomly select the training sample
train_ind = false(nTotal, 1);
train_ind(randperm(nTotal, nTrain)) = true;
Xtrain = data(train_ind,:);
Ytrain = label(train_ind);
Xtest = data(~train_ind,:);
Ytest = label(~train_ind);

% normalize
Xm = mean(Xtrain);
Xs = std(Xtrain);
Xtrain = (Xtrain - Xm) ./ (Xs + eps);
Xtest  = (Xtest - Xm) ./ (Xs + eps);

% start training
params = struct("c", 0.25, "gamma", 64, ...
                "twIter", 100, "eps", eps, "shrink", true, "verbose", true);
tic;
model = lightwsvm_train(Xtrain, Ytrain, params);
accu = lightwsvm_predict(Xtest, Ytest, model);
toc;
fprintf("Accuracy: %.4f%%\n", accu);