#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <list>

#include "mex.h"
#include "blas.h"

#define INF HUGE_VAL

#define info(...) mexPrintf(__VA_ARGS__)

inline double dot(const double *a, const double *b, size_t n) {
    ptrdiff_t nn = n;
    ptrdiff_t inc = 1;
    return ddot(&nn, a, &inc, b, &inc);
}

inline void axpy(const double* a, const double* b, const double mul, double* c, size_t n) {
    ptrdiff_t nn = n;
    ptrdiff_t inc = 1;
    daxpy(&nn, &mul, b, &inc, c, &inc);
}

inline double norm(const double *a, size_t n) {
    ptrdiff_t nn = n;
    ptrdiff_t inc = 1;
    return dnrm2(&nn, a, &inc);
}

double oneTWSVM(
           double* w, double* alpha, // output
           const double *X, const double *y, const double cls_id, const size_t m, const size_t n,  // input matrices
           const double gamma, const double c, const size_t max_iter = 100, const double eps = 2.23e-16, const bool verbose = false, const bool do_shrink = true) // hyperparameters
{
    double  pgrad_max_new;
	double  pgrad_max_old =  INF;
    double  pgrad_min_new;
    double  pgrad_min_old = -INF;
    double  tol;
    double  r_neg;
    double  r_pos;
    double* xi2 = new double[n];
    size_t* idx = new size_t[n];
    size_t  active_size = n;
    size_t  lp;
    std::random_device rd;
    std::mt19937 random_gen(rd());

    size_t n_pos = 0;
    size_t n_neg = 0;

    for (size_t i = 0; i < n; i++) {
        xi2[i] = dot(X+i*m, X+i*m, m) / gamma;
    }
    std::iota(idx, idx + n, 0);

    std::fill(w, w + m, 0.0);
    std::fill(alpha, alpha + n, 0.0);
    tol = eps * norm(xi2, n);
    for (lp = 0; lp < max_iter; lp++) {
        // set the projection gradient
        pgrad_max_new = -INF;
        pgrad_min_new =  INF;
        r_neg = 0.0;
        r_pos = 0.0;
        
        std::shuffle(idx, idx + active_size, random_gen); // randperm(nB)

        for (size_t i_raw = 0; i_raw < active_size; i_raw++) {
            size_t i = idx[i_raw];
            if (y[i] == cls_id) {
                double grad = dot(X + i*m, w, m) + alpha[i];
                if (std::fabs(grad) > tol*10) {
                    double delta = grad / (xi2[i] + 1);
                    alpha[i] -= delta;
                    axpy(w, X + i*m, -delta / gamma, w, m);
                }
                r_pos = std::max(r_pos, std::fabs(grad));
            }
            else {
                double pgrad;
                double lambda_new;
                double grad;

                grad = dot(X + i*m, w, m) - 1.0;
                pgrad = 0.0;
                if (alpha[i] == 0) {
                    if (do_shrink) {
                        if (grad > pgrad_max_old) {
                            std::swap(idx[i], idx[--active_size]);
                            i_raw--;
                            continue;
                        }
                        else if (grad < 0) {
                            pgrad = grad;
                        }
                    }
                    else {
                        pgrad = std::min(0.0, grad);
                    }
                }
                else if (alpha[i] == c) {
                    if (do_shrink) {
                        if (grad < pgrad_min_old) {
                            std::swap(idx[i], idx[--active_size]);
                            i_raw--;
                            continue;
                        }
                        else if (grad > 0) {
                            pgrad = grad;
                        }
                    }
                    else {
                        pgrad = std::max(0.0, grad);
                    }
                }
                else {
                    pgrad = grad;
                }
                
                pgrad_max_new = std::max(pgrad_max_new, pgrad);
                pgrad_min_new = std::min(pgrad_min_new, pgrad);

                if (std::fabs(pgrad) > tol) {
                    lambda_new = std::min(c, std::max(alpha[i] - grad / (xi2[i]), 0.0));
                    axpy(w, X + i*m, (lambda_new - alpha[i])/gamma, w, m);
                    alpha[i] = lambda_new;
                }
            }
        }

        if (verbose && lp % 10 == 0) {
            info(".");
            if (lp > 0 && lp % 200 == 0) {
                info("\n");
            }
        }

        if (std::fabs(pgrad_max_new) < tol && std::fabs(pgrad_min_new) < tol && pgrad_max_new - pgrad_min_new < tol) 
        {
            if (active_size == n) {
                break;
            }
            else {
                active_size = n;
                pgrad_max_old =  INF;
                pgrad_min_old = -INF;
                if (verbose) {info("*");}
                continue;
            }
        }

        pgrad_max_old = pgrad_max_new;
        pgrad_min_old = pgrad_min_new;

        if (pgrad_max_old <= 0) {
            pgrad_max_old = INF;
        }
        if (pgrad_min_old >= 0) {
            pgrad_min_old = -INF;
        }
    }
    if (verbose) {
        info("\noptimization finished, #iter = %d\n", lp);
    }

    delete[] xi2;
    delete[] idx;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 9) {
        mexErrMsgTxt("Nine input arguments required.");
    }
    if (nlhs != 2) {
        mexErrMsgTxt("A output arguments required.");
    }

    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]) || !mxIsDouble(prhs[4]) || !mxIsDouble(prhs[5]) || !mxIsDouble(prhs[6]) || !mxIsLogical(prhs[7]) || !mxIsLogical(prhs[8])) {
        mexErrMsgTxt("Input arguments must be double.");
    }

    if (mxGetN(prhs[1]) != 1) {
        mexErrMsgTxt("y must be column vector.");
    }

    if (mxGetM(prhs[1]) != mxGetN(prhs[0])) {
        mexErrMsgTxt("The number of the cols of X must be equal to the length of y.");
    }

    double *X = mxGetPr(prhs[0]);
    double *y = mxGetPr(prhs[1]);
    double cls = mxGetScalar(prhs[2]);
    double gamma = mxGetScalar(prhs[3]);
    double c = mxGetScalar(prhs[4]);
    double max_iter = mxGetScalar(prhs[5]);
    double eps = mxGetScalar(prhs[6]);
    bool *verbose = mxGetLogicals(prhs[7]);
    bool *do_shrink = mxGetLogicals(prhs[8]);

    size_t m = mxGetM(prhs[0]);
    size_t n = mxGetN(prhs[0]);

    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *w_out = mxGetPr(plhs[0]);
    double *alpha_out = mxGetPr(plhs[1]);

    oneTWSVM(w_out, alpha_out, X, y, cls, m, n, gamma, c, max_iter, eps, *verbose, *do_shrink);
}