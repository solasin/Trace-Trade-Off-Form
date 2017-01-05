#ifndef FVL_H
#define FVL_H

struct FVL_info
{
	int nU;
	int nV;
	int N;

	int p;

	double* sigma;

	double** U;
	double** V;

	int num_tr;
	int** trIdx;
	double* trR;
	double* ptrR;

	int num_ts;
	int** tsIdx;
	double* tsR;
	double* ptsR;
};

struct FVL_tmp
{
	double MaxR;
	double MinR;

	double Meanr;
	double avr;

	double M;
	double minEig;
	double* eigV;

	double epsilon;
	double lambda;

	int* Uorder;
	int* Vorder;
	double* Ur;
	double* Ue;
	double* Ve;

	int* UmapV;

	int** Unum;
	int** Vnum;

	double** Y;
	double** dY;

	double* pz;
};

typedef struct FVL_info FVL_INFO;
typedef struct FVL_tmp FVL_TMP;

double globalOpt(FVL_INFO* info, FVL_TMP* tmp);  //print both the RMSE of the training set and the RMSE of the testing set
double globalOpt2(FVL_INFO* info, FVL_TMP* tmp); //print only the RMSE of the training set
#endif

