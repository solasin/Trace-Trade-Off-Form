#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mkl_cblas.h>
#include <mkl_vml_functions.h>
#include <mkl_vsl.h>

#include "MF_sparse_matrix.h"
#include "lbfgs_direct.h"
#include "FVL.h"

int localOptTmpMemAlloc(FVL_INFO info, FVL_TMP* tmp)
{
	int i;
	tmp->dY[0] = (double*)malloc(2 * info.p * info.N * sizeof(double));
#pragma omp parallel private(i)
	for (i = 0; i < info.N; i++)
	{
		tmp->dY[i] = tmp->dY[0] + i * info.p;
	}
	tmp->pz = tmp->dY[0] + info.p * info.N;

	return 0;
}

int localOptTmpMemFree(FVL_TMP* tmp)
{
	free(tmp->dY[0]);

	return 0;
}

double localOptCostFunction(FVL_INFO info, FVL_TMP tmp)
{
	double cost;
	int p = info.p;
	int N = info.N;
	int i, j, ThreadID;
	double *vU, *vV, tp;

#pragma omp parallel private(i, j, vU, vV, tp, ThreadID)
	{
		ThreadID = omp_get_thread_num();
		for (i = ThreadID; i < info.num_tr; i += NUM_OF_THREADS)
		{
			vU = info.U[info.trIdx[i][0]];
			vV = info.V[info.trIdx[i][1]];
			tp = 0.0;
			for (j = 0; j < p; j++)
				tp += vU[j] * vV[j];
			tmp.Ue[i] = tmp.Ur[i] - tp;
			tmp.Ve[tmp.UmapV[i]] = tmp.Ue[i];
		}
	}

	cost = tmp.lambda * cblas_ddot(p * N, tmp.Y[0], 1, tmp.Y[0], 1) / 2;
	cost += cblas_ddot(info.num_tr, tmp.Ve, 1, tmp.Ve, 1) / 2;

	return cost;
}

int localOptComputeGradient(FVL_INFO info, FVL_TMP tmp)
{
	symetric_matrix_matrix_scale_diag(tmp.Uorder, tmp.Ue, tmp.Vorder, tmp.Ve, -tmp.lambda, info.U, info.V, tmp.dY, tmp.dY + info.nU, tmp.Unum, tmp.Vnum, info.nU, info.nV, info.p);
	cblas_dscal(info.N * info.p, -1.0, tmp.dY[0], 1);
	return 0;
}

int localOptComputeNumericalGradient(FVL_INFO info, FVL_TMP tmp)
{
	int i;
	int base = info.N * info.p;
	double epsilon = tmp.epsilon;
	double cost1, cost2;
	double* y = tmp.Y[0];
	double* dy = tmp.pz;

	for (i = 0; i < base; i++)
	{
		y[i] += epsilon;
		cost1 = localOptCostFunction(info, tmp);
		y[i] -= 2 * epsilon;
		cost2 = localOptCostFunction(info, tmp);
		y[i] += epsilon;
		dy[i] = (cost1 - cost2) / (2 * epsilon);
		printf("%d/%d: %g, %g, %g, %g, %g\n", i, base, y[i] + epsilon, y[i] - epsilon, cost1, cost2, dy[i]);
	}

	return 0;
}

double localOptUpdateModel(FVL_INFO* info, FVL_TMP* tmp, double oc)
{
	int i = 0;
	double stepSize = 100.0;
	double tp = cblas_ddot(info->N * info->p, tmp->pz, 1, tmp->pz, 1);
	cblas_dscal(info->N * info->p, 1.0 / sqrt(tp), tmp->pz, 1);
	cblas_daxpy(info->N * info->p, -stepSize, tmp->pz, 1, tmp->Y[0], 1);
	double cost = localOptCostFunction(*info, *tmp);
	//	printf("%d, cost: %g (%g), stepSize: %g\n", i, cost, oc, stepSize);

	while (i < 20 && (isnan(cost) || isinf(cost) || cost >= oc))
	{
		i++;
		stepSize /= 2.0;
		cblas_daxpy(info->N * info->p, stepSize, tmp->pz, 1, tmp->Y[0], 1);
		cost = localOptCostFunction(*info, *tmp);
		//		printf("%d, cost: %g (%g), stepSize: %g\n", i, cost, oc, stepSize);
	}
	//	exit(0);

	return cost;
}

// root of one variable 3 order equation

int fun(double a, double b, double c, double d,
	double *real_y1, double *real_y2, double *real_y3,
	double *imag_y1, double *imag_y2, double *imag_y3)
{
	double p, q, r, u, v, g, h, fai;
	p = (3.0*a*c - b*b) / (3 * a*a);
	q = (2.0*pow(b, 3.0) - 9 * a*b*c + 27.0*a*a*d) / (27.0*pow(a, 3.0));
	r = b / (3.0*a);
	h = pow(q / 2.0, 2.0) + pow(p / 3.0, 3.0);
	g = sqrt(h);
	if (h >= 0)
	{
		if (-q / 2.0 + g<0)
			u = -pow(fabs(-q / 2.0 + g), 1.0 / 3.0);
		else
			u = pow((-q / 2.0 + g), 1.0 / 3.0);
		if (-q / 2.0 - g<0)
			v = -pow(fabs(-q / 2.0 - g), 1.0 / 3.0);
		else
			v = -pow((-q / 2.0 - g), 1.0 / 3.0);
		if (h == 0)
		{
			*real_y1 = u + v - r;            *imag_y1 = 0;
			*real_y2 = -(u + v) / 2 - r;       *imag_y2 = 0;
			*real_y3 = -(u + v) / 2 - r;       *imag_y3 = 0;
		}
		else
		{
			*real_y1 = u + v - r;       *imag_y1 = 0;
			*real_y2 = -(u + v) / 2;    *imag_y2 = sqrt(3.0)*(u - v) / 2;
			*real_y3 = -(u + v) / 2;    *imag_y3 = -sqrt(3.0)*(u - v) / 2;
		}
	}
	else
	{
		fai = acos((-q / 2) / (sqrt(pow(fabs(p), 3) / 27)));
		*real_y1 = 2 * sqrt(fabs(p) / 3.0)*cos(fai / 3.0) - r;
		*real_y2 = -2 * sqrt(fabs(p) / 3.0)*cos((fai + 3.1415926) / 3.0) - r;
		*real_y3 = -2 * sqrt(fabs(p) / 3.0)*cos((fai - 3.1415926) / 3.0) - r;
		*imag_y1 = 0;   *imag_y2 = 0;    *imag_y3 = 0;
	}

	return 0;
}

double localOptUpdateModel_steepest_lbfgs(FVL_INFO* info, FVL_TMP* tmp, double oc)
{
	double tp = cblas_ddot(info->N * info->p, tmp->pz, 1, tmp->pz, 1);
	cblas_dscal(info->N * info->p, 1.0 / sqrt(tp), tmp->pz, 1);
	double a, b, c, d;
	double t1, t2;
	double tp1 = 0.0, tp2 = 0.0;
	int i, k;

#pragma omp parallel private(i, k, t1, t2), reduction(+: tp1, tp2)
	for (i = 0; i < info->num_tr; i++)
	{
		t1 = 0.0; t2 = 0.0;
		for (k = 0; k < info->p; k++)
		{
			t1 += *(tmp->pz + info->trIdx[i][0] * info->p + k) * *(tmp->pz + (info->nU + info->trIdx[i][1]) * info->p + k);
			t2 += info->U[info->trIdx[i][0]][k] * *(tmp->pz + (info->nU + info->trIdx[i][1]) * info->p + k);
		}
		tp1 -= tmp->Ue[i] * t1;
		tp2 -= tmp->Ue[i] * t2;
		tmp->Ue[i] = t1;
		tmp->Ve[i] = t2;
	}
	tp1 += tmp->lambda * cblas_ddot(info->N * info->p, tmp->pz, 1, tmp->pz, 1) / 2;
	tp2 += tmp->lambda * cblas_ddot(info->N * info->p, tmp->Y[0], 1, tmp->pz, 1) / 2;
	a = cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ue, 1);
	b = -3 * cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ve, 1);
	c = tp1 + 2 * cblas_ddot(info->num_tr, tmp->Ve, 1, tmp->Ve, 1);
	d = -tp2;

	double real_x1, real_x2, real_x3;
	double imag_x1, imag_x2, imag_x3;
	fun(a, b, c, d, &real_x1, &real_x2, &real_x3, &imag_x1, &imag_x2, &imag_x3);

	double stepSize = 100.0;
	if (real_x1 > 0)
		stepSize = real_x1;
	if (real_x2 > 0 && real_x2 < stepSize)
		stepSize = real_x2;
	if (real_x3 > 0 && real_x3 < stepSize)
		stepSize = real_x3;

	if (isnan(stepSize) || stepSize > 100.0)
		stepSize = 100.0;
	cblas_daxpy(info->N * info->p, -stepSize, tmp->pz, 1, tmp->Y[0], 1);
	double cost = localOptCostFunction(*info, *tmp);
	i = 0;
	while (i < 200 && (isnan(cost) || isinf(cost) || cost >= oc))
	{
		i++;
		stepSize /= 2.0;
		cblas_daxpy(info->N * info->p, stepSize, tmp->pz, 1, tmp->Y[0], 1);
		cost = localOptCostFunction(*info, *tmp);
	}
	stepSize /= 2.0;
	cblas_daxpy(info->N * info->p, stepSize, tmp->pz, 1, tmp->Y[0], 1);
	cost = localOptCostFunction(*info, *tmp);

	return cost;

}

double localOpt(FVL_INFO* info, FVL_TMP* tmp)
{
	int i = 0;

	lbfgs_initial(5, info->N * info->p);
	double cost = localOptCostFunction(*info, *tmp);
	localOptComputeGradient(*info, *tmp);
	double err = sqrt(cblas_ddot(info->p * info->N, tmp->dY[0], 1, tmp->dY[0], 1) / (info->p * info->N));
	//	printf("iter: %d, cost: %g, error: %g\n", i, cost, err);

	/*
	//check gradient
	localOptComputeNumericalGradient(*info, *tmp);
	int base = info->N * info->p;
	FILE* fp = fopen("dY.txt", "w");
	double norm1 = 0.0;
	double norm2 = 0.0;
	for(i = 0; i < base; i++)
	{
	norm1 += (tmp->dY[0][i] - tmp->pz[i]) * (tmp->dY[0][i] - tmp->pz[i]);
	norm2 += (tmp->dY[0][i] + tmp->pz[i]) * (tmp->dY[0][i] + tmp->pz[i]);
	fprintf(fp, "%g\t%g\n", tmp->dY[0][i], tmp->pz[i]);
	}
	norm1 = sqrt(norm1);
	norm2 = sqrt(norm2);
	printf("%g\n", norm1 / norm2);
	fclose(fp);
	exit(0);
	*/
	while (i < 100000 && err > tmp->epsilon)
	{
		lbfgs_direct(tmp->dY[0], tmp->Y[0], tmp->pz);
		//cost = localOptUpdateModel(info, tmp, cost);
		cost = localOptUpdateModel_steepest_lbfgs(info, tmp, cost);
		localOptComputeGradient(*info, *tmp);
		err = cblas_ddot(info->p * info->N, tmp->dY[0], 1, tmp->dY[0], 1) / (info->p * info->N);
		i++;
		//printf("iter: %d, cost: %g, error: %g\n", i, cost, err);
	}

	lbfgs_free();

	return cost;
}

int checkOpt(VSLStreamStatePtr stream, FVL_INFO* info, FVL_TMP* tmp)
{
	int mark = 0;
	int iter;
	double err;

	double tp = sqrt(cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ue, 1)) / tmp->avr;
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, info->N, tmp->pz, 0.0, 0.01);
	tmp->minEig = -symetric_matrix_scale_diag_max_eig(tmp->Uorder, tmp->Ue, tmp->Vorder, tmp->Ve, -tmp->lambda, tmp->pz, tmp->pz + info->nU, tmp->eigV, tmp->eigV + info->nU, tmp->Unum, tmp->Vnum, info->nU, info->nV, tmp->M * tp + tmp->lambda, 100000, 1.0E-6, &iter, &err);

	if (tmp->minEig / info->N > -tmp->epsilon)
		mark = 1;

	return mark;
}

int addDimension(FVL_INFO* info, FVL_TMP* tmp)
{
	double* u = tmp->eigV;
	double* v = tmp->eigV + info->nU;
	double* y = tmp->Y[0];
	int i = 0;

	info->p++;
	tmp->Y[0] = (double*)malloc(info->p * (info->nU + info->nV) * sizeof(double));
#pragma omp parallel private(i)
	for (i = 0; i < info->N; i++)
		tmp->Y[i] = tmp->Y[0] + i * info->p;
	for (i = 0; i < info->p - 1; i++)
		cblas_dcopy(info->N, y + i, info->p - 1, tmp->Y[0] + i, info->p);
	free(y);
	cblas_dcopy(info->N, tmp->eigV, 1, tmp->Y[0] + info->p - 1, info->p);
#pragma omp parallel private(i)
	for (i = 0; i < info->num_tr; i++)
		tmp->Ve[i] = u[info->trIdx[i][0]] * v[info->trIdx[i][1]];
	double alpha = -tmp->minEig / (2 * cblas_ddot(info->num_tr, tmp->Ve, 1, tmp->Ve, 1));
	cblas_dscal(info->N, sqrt(alpha), tmp->Y[0] + info->p - 1, info->p);
	//	printf("alpha: %g\n", alpha);

	return 0;
}

int initialRankOne(VSLStreamStatePtr stream, FVL_INFO* info, FVL_TMP* tmp)
{
	int i;
	double* u = tmp->eigV;
	double* v = tmp->eigV + info->nU;
	int iter;
	double err;

	double tp = sqrt(cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ue, 1)) / tmp->avr;
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, info->N, tmp->pz, 0.0, 0.01);
	tmp->minEig = -symetric_matrix_scale_diag_max_eig(tmp->Uorder, tmp->Ue, tmp->Vorder, tmp->Ve, -tmp->lambda, tmp->pz, tmp->pz + info->nU, tmp->eigV, tmp->eigV + info->nU, tmp->Unum, tmp->Vnum, info->nU, info->nV, tmp->M * tp + tmp->lambda, 10000, 1.0E-6, &iter, &err);


	info->p = 1;
	tmp->Y[0] = (double*)malloc(info->N * sizeof(double));
#pragma omp parallel private(i)
	for (i = 0; i < info->N; i++)
		tmp->Y[i] = tmp->Y[0] + i;
	cblas_dcopy(info->N, tmp->eigV, 1, tmp->Y[0], 1);
#pragma omp parallel private(i)
	for (i = 0; i < info->num_tr; i++)
		tmp->Ve[i] = u[info->trIdx[i][0]] * v[info->trIdx[i][1]];
	//	double alpha = -cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ve, 1) / cblas_ddot(info->num_tr, tmp->Ve, 1, tmp->Ve, 1);
	double alpha = 1.0;
	if (tmp->minEig < 0.0)
		alpha = -tmp->minEig / (2 * cblas_ddot(info->num_tr, tmp->Ve, 1, tmp->Ve, 1));
	cblas_dscal(info->N, sqrt(alpha), tmp->Y[0], 1);
	//	printf("alpha: %g\n", alpha);

	return 0;
}

int pred(FVL_INFO info, FVL_TMP tmp)
{
	int i, k;
	double* u;
	double* v;

#pragma omp parallel private(i, k)
	for (i = 0; i < info.num_tr; i++)
	{
		info.ptrR[i] = 0.0;
		u = info.U[info.trIdx[i][0]];
		v = info.V[info.trIdx[i][1]];
		for (k = 0; k < info.p; k++)
			info.ptrR[i] += u[k] * v[k];
		info.ptrR[i] += tmp.Meanr;
		info.ptrR[i] = (info.ptrR[i] >= 0.0) ? info.ptrR[i] : 0.0;
		info.ptrR[i] = (info.ptrR[i] <= 1.0) ? info.ptrR[i] : 1.0;
		info.ptrR[i] = tmp.MinR + info.ptrR[i] * (tmp.MaxR - tmp.MinR);
	}

#pragma omp parallel private(i, k)
	for (i = 0; i < info.num_ts; i++)
	{
		info.ptsR[i] = 0.0;
		u = info.U[info.tsIdx[i][0]];
		v = info.V[info.tsIdx[i][1]];
		for (k = 0; k < info.p; k++)
			info.ptsR[i] += u[k] * v[k];
		info.ptsR[i] += tmp.Meanr;
		info.ptsR[i] = (info.ptsR[i] >= 0.0) ? info.ptsR[i] : 0.0;
		info.ptsR[i] = (info.ptsR[i] <= 1.0) ? info.ptsR[i] : 1.0;
		info.ptsR[i] = tmp.MinR + info.ptsR[i] * (tmp.MaxR - tmp.MinR);
	}

	return 0;
}

double RMSE(double* R1, double* R2, int num)
{
	double res = 0.0;
	int i;

#pragma omp parallel for private(i), reduction(+:res)
	for (i = 0; i < num; i++)
		res += (R1[i] - R2[i]) * (R1[i] - R2[i]);

	res = sqrt(res / num);

	return res;
}

double globalOpt(FVL_INFO* info, FVL_TMP* tmp)
{
	int i = 0;
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MCG31, 1);
	info->p = 1;
	localOptTmpMemAlloc(*info, tmp);
	initialRankOne(stream, info, tmp);
	double cost = localOpt(info, tmp);
	//	exit(0);
	int mark = checkOpt(stream, info, tmp);
	localOptTmpMemFree(tmp);
	pred(*info, *tmp);
	double trRMSE = RMSE(info->ptrR, info->trR, info->num_tr);
	double tsRMSE = RMSE(info->ptsR, info->tsR, info->num_ts);
	printf("#Iter: %d, Rank: %d, MinEig: %g, Cost: %g, trRMSE: %g, tsRMSE: %g\n", i, info->p, tmp->minEig, cost, trRMSE, tsRMSE);
	while (!mark)
	{
		addDimension(info, tmp);
		localOptTmpMemAlloc(*info, tmp);
		cost = localOpt(info, tmp);
		mark = checkOpt(stream, info, tmp);
		localOptTmpMemFree(tmp);
		pred(*info, *tmp);
		trRMSE = RMSE(info->ptrR, info->trR, info->num_tr);
		tsRMSE = RMSE(info->ptsR, info->tsR, info->num_ts);
		i++;
		printf("#Iter: %d, Rank: %d, MinEig: %g, Cost: %g, trRMSE: %g, tsRMSE: %g\n", i, info->p, tmp->minEig, cost, trRMSE, tsRMSE);
	}

	info->sigma = (double*)malloc(info->p * sizeof(double));
	for (i = 0; i < info->p; i++)
		info->sigma[i] = sqrt(cblas_ddot(info->N, tmp->Y[0] + i, info->p, tmp->Y[0] + i, info->p) / info->N);

	return cost;
}

double globalOpt2(FVL_INFO* info, FVL_TMP* tmp)
{
	int i = 0;
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MCG31, 1);
	info->p = 1;
	localOptTmpMemAlloc(*info, tmp);
	initialRankOne(stream, info, tmp);
	double cost = localOpt(info, tmp);
	//	exit(0);
	int mark = checkOpt(stream, info, tmp);
	localOptTmpMemFree(tmp);
	pred(*info, *tmp);
	double trRMSE = RMSE(info->ptrR, info->trR, info->num_tr);
	printf("#Iter: %d, Rank: %d, MinEig: %g, Cost: %g, trRMSE: %g\n", i, info->p, tmp->minEig, cost, trRMSE);
	while (!mark)
	{
		addDimension(info, tmp);
		localOptTmpMemAlloc(*info, tmp);
		cost = localOpt(info, tmp);
		mark = checkOpt(stream, info, tmp);
		localOptTmpMemFree(tmp);
		pred(*info, *tmp);
		trRMSE = RMSE(info->ptrR, info->trR, info->num_tr);
		i++;
		printf("#Iter: %d, Rank: %d, MinEig: %g, Cost: %g, trRMSE: %g\n", i, info->p, tmp->minEig, cost, trRMSE);
	}

	info->sigma = (double*)malloc(info->p * sizeof(double));
	for (i = 0; i < info->p; i++)
		info->sigma[i] = sqrt(cblas_ddot(info->N, tmp->Y[0] + i, info->p, tmp->Y[0] + i, info->p) / info->N);

	return cost;
}

