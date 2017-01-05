#include <stdlib.h>
#include <mkl_cblas.h>
#include <mkl_vml_functions.h>
#include <omp.h>
#include <string.h>

double** lbfgs_s;
double** lbfgs_y;

double* lbfgs_rho;
double* lbfgs_alpha;
double* lbfgs_beta;

double* lbfgs_x;
double* lbfgs_g;

int lbfgs_K;
int lbfgs_N;
int lbfgs_head;
int lbfgs_tail;

int lbfgs_initial(int k, int n)
{
	lbfgs_K = k;
	lbfgs_N = n;
	lbfgs_head = lbfgs_K - 1;
	lbfgs_tail = 0;
	lbfgs_s = (double**)malloc(k * sizeof(double*));
	lbfgs_y = (double**)malloc(k * sizeof(double*));
	lbfgs_rho = (double*)malloc(k * sizeof(double));
	lbfgs_alpha = (double*)malloc(k * sizeof(double));
	lbfgs_beta = (double*)malloc(k * sizeof(double));
	lbfgs_x = (double*)malloc(n * sizeof(double));
	lbfgs_g = (double*)malloc(n * sizeof(double));
	double* tmp = (double*)malloc(k * n * sizeof(double));
	int i;
#pragma omp parallel for private(i)
	for (i = 0; i < k; i++)
		lbfgs_s[i] = tmp + i * n;

	tmp = (double*)malloc(k * n * sizeof(double));
#pragma omp parallel for private(i)
	for (i = 0; i < k; i++)
		lbfgs_y[i] = tmp + i * n;

	memset((char*)lbfgs_s[0], '\0', lbfgs_N * lbfgs_K * sizeof(double));
	memset((char*)lbfgs_y[0], '\0', lbfgs_N * lbfgs_K * sizeof(double));
	memset((char*)lbfgs_rho, '\0', lbfgs_K * sizeof(double));
	memset((char*)lbfgs_alpha, '\0', lbfgs_K * sizeof(double));
	memset((char*)lbfgs_beta, '\0', lbfgs_K * sizeof(double));
	memset((char*)lbfgs_x, '\0', lbfgs_N * sizeof(double));
	memset((char*)lbfgs_g, '\0', lbfgs_N * sizeof(double));


	return 0;
}

int lbfgs_free()
{
	free(&lbfgs_s[0][0]);
	free(&lbfgs_y[0][0]);
	free(lbfgs_s);
	free(lbfgs_y);
	free(lbfgs_rho);
	free(lbfgs_alpha);
	free(lbfgs_beta);
	free(lbfgs_x);
	free(lbfgs_g);

	return 0;
}

int lbfgs_direct(double* kg, double* kx, double *p)
{
	int i, id;

	cblas_dcopy(lbfgs_N, kg, 1, p, 1);

	for (i = 0; i < lbfgs_K; i++)
	{
		id = (lbfgs_head - i + lbfgs_K) % lbfgs_K;
		lbfgs_alpha[id] = lbfgs_rho[id] * cblas_ddot(lbfgs_N, lbfgs_s[id], 1, p, 1);
		cblas_daxpy(lbfgs_N, -lbfgs_alpha[id], lbfgs_y[id], 1, p, 1);
	}

	for (i = 0; i < lbfgs_K; i++)
	{
		id = (lbfgs_tail + i) % lbfgs_K;
		lbfgs_beta[id] = lbfgs_rho[id] * cblas_ddot(lbfgs_N, lbfgs_y[id], 1, p, 1);
		cblas_daxpy(lbfgs_N, lbfgs_alpha[id] - lbfgs_beta[id], lbfgs_s[id], 1, p, 1);
	}

	lbfgs_head = (lbfgs_head + 1) % lbfgs_K;
	lbfgs_tail = (lbfgs_tail + 1) % lbfgs_K;

	i = lbfgs_head;
	vdSub(lbfgs_N, kx, lbfgs_x, lbfgs_s[i]);
	vdSub(lbfgs_N, kg, lbfgs_g, lbfgs_y[i]);
	lbfgs_rho[i] = 1.0 / cblas_ddot(lbfgs_N, lbfgs_y[i], 1, lbfgs_s[i], 1);
	cblas_dcopy(lbfgs_N, kx, 1, lbfgs_x, 1);
	cblas_dcopy(lbfgs_N, kg, 1, lbfgs_g, 1);

	return 0;
}

