#ifndef LBFGS_DIRECT_H_INCLUDED
#define LBFGS_DIRECT_H_INCLUDED

int lbfgs_initial(int k, int n);
int lbfgs_free();
int lbfgs_direct(double* kg, double* kx, double* p);

#endif // LBFGS_DIRECT_H_INCLUDED
