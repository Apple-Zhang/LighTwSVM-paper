function model = lightwsvm_train(X, Y, param)
%LIGHTWSVM_TRAIN 
%   Input:
%       X: n*m sample matrix. n is the number of training sample, m is dim.
%       Y: n*1 label matrix. y_i \in 1, ..., numClass.
%   param: struct(
%                 "c",      double_scalar,  % penalty factor, REQUIRED
%                 "gamma",  double_scalar,  % regularization, REQUIRED
%                 "twIter", int_scalar,     % max iterations, default: 100
%                 "eps",    double_scalar,  % tolerance, default: 2.2e-16
%                 "shrink", logical_scalar, % do shrink, default: true
%                 "verbose",logical_scalar, % output intermediate stuff.
%                                             default: false
%                )
%
%   Output:
%       model: including weight and Lagrange multipliers
% 
%   By Junhong Zhang in 2024.10.15

[n, m] = size(X);
X = [X, ones(n,1)];
m = m+1;

uY = unique(Y);
numClass = numel(uY);
model.labelName = uY;

model.w     = zeros(m,numClass);
model.alpha = zeros(n,numClass);

if ~isfield(param, "twIter")
    param.twIter = 100;
end

if ~isfield(param, "eps")
    param.eps = eps;
end

if ~isfield(param, "verbose")
    param.verbose = false;
end

if ~isfield(param, "shrink")
    param.shrink = true;
end

for ic = 1:numClass
    [w, alpha] = onetwsvm(full(X'), Y, ic, param.gamma, param.c, param.twIter, param.eps, param.verbose, param.shrink);

    % check KKT condition and do conjugate gradient descent
    mask = (model.labelName(ic) == Y);
    Xp = X(mask,:);

    kkterr = norm(Xp*w + alpha(mask), "inf");
    if param.verbose
        fprintf("classID: %d, KKT error: %.6f\n", ic, kkterr);
    end

    if kkterr > 1e-3
        wA = Xp'*alpha(mask) / param.gamma;
        wB = w - wA;
        CA = (Xp'*Xp) / param.gamma;
    
        [alpha(mask), wA, kkterr] = conjgrad(Xp, CA, w, wA, alpha(mask), param);
        w = wA + wB;
        if param.verbose
            fprintf("After ConjGrad, KKT error: %.6f\n", kkterr);
        end
    end

    model.w(:, ic) = w;
    model.alpha(:, ic) = alpha;
end

end

function [tau, wA, r2] = conjgrad(XA, CA, w, wA, tau, param)

EPSILON = param.eps;
if ~isfield(param, 'maxIterCG')
    param.maxIterCG = 100;
end

nA = size(XA,1);

if isempty(tau)
    tau = zeros(nA,1);
end
tau0 = tau;
tau0_2 = sum(tau0.^2,"all");

% Initialize aux variables
u = -w;
v = u;
c = -1;
d = c;

sum_sp_u = zeros(size(u,1),1);
sum_sp_c = 0;

% initialize r and p
% p = XA*u + c*tau0;
% r = p;
% r2 = sum(r.^2,"all");

dot_ab = @(u,v,c,d) u'*CA*v + (d*u+c*v)'*wA + c*d*tau0_2;
r2 = dot_ab(u, u, c, c);
if r2 <= EPSILON * nA
    return;
end

for lp = 1:param.maxIterCG
    v_ = CA*u + c*wA + u;
    d_ = c;
    
    pTAp = dot_ab(u, v_, c, d_);
    s = r2 ./ pTAp;
    sum_sp_u = sum_sp_u + s*u;
    sum_sp_c = sum_sp_c + s*c;

    v = v - s*v_;
    d = d - s*d_;

    r2old = r2;
    r2 = dot_ab(v, v, d, d);
    if r2 <= EPSILON * nA
        tau = XA*sum_sp_u + (sum_sp_c + 1)*tau0;
        wA  = CA*sum_sp_u + (sum_sp_c + 1)*wA;
        return;
    end
    beta = r2/r2old;
    
    u = v + beta*u;
    c = d + beta*c;
end
tau = XA*sum_sp_u + (sum_sp_c+1)*tau0;
wA  = CA*sum_sp_u + (sum_sp_c+1)*wA;

if param.debug
    fprintf("Warning: not convergence after %d iterations.\n", maxIter);
end
end