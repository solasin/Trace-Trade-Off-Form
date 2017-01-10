#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "MF_sparse_matrix.h"
#include "FVL.h"
#include"Global_Config.h"
double LAMBDA;

int readData(FVL_INFO* info, FVL_TMP* tmp, char* path) //include ratings both in the testing set and the training set.
{
	FILE* fp;
	char ch[100];

	int nU;
	int nV;
	int num_tr;
	int num_ts;

	int i, tp, tpp;

	sprintf(ch, "%s/data.statistics", path);
	fp = fopen(ch, "r");
	fscanf(fp, "%d%d", &nU, &nV);
	fscanf(fp, "%d%d", &num_tr, &num_ts);
	fclose(fp);
	info->nU = nU;
	info->nV = nV;
	info->N = nU + nV;
	info->num_tr = num_tr;
	info->num_ts = num_ts;

	info->U = (double**)malloc(info->N * sizeof(double*));
	info->V = info->U + nU;
	info->trIdx = (int**)malloc((num_tr + num_ts) * sizeof(int*));
	info->tsIdx = info->trIdx + num_tr;
	info->trIdx[0] = (int*)malloc(2 * (num_tr + num_ts) * sizeof(int));
	for (i = 1; i < num_tr + num_ts; i++)
		info->trIdx[i] = info->trIdx[0] + i * 2;
	info->trR = (double*)malloc(2 * (num_tr + num_ts) * sizeof(double));
	info->ptrR = info->trR + num_tr;
	info->tsR = info->ptrR + num_tr;
	info->ptsR = info->tsR + num_ts;

	tmp->eigV = (double*)malloc((info->N + 3 * info->num_tr) * sizeof(double));
	tmp->Ur = tmp->eigV + info->N;
	tmp->Ue = tmp->Ur + info->num_tr;
	tmp->Ve = tmp->Ue + info->num_tr;
	tmp->Uorder = (int*)malloc((3 * info->num_tr + 2 * info->N)* sizeof(int));
	tmp->Vorder = tmp->Uorder + info->num_tr;
	tmp->UmapV = tmp->Vorder + info->num_tr;

	tmp->Unum = (int**)malloc(info->N * sizeof(int*));
	tmp->Vnum = tmp->Unum + info->nU;
	for (i = 0; i < info->N; i++)
		tmp->Unum[i] = tmp->UmapV + info->num_tr + i * 2;

	tmp->Y = info->U;
	tmp->dY = (double**)malloc(info->N * sizeof(double*));

	tmp->MaxR = 0.0;
	tmp->MinR = 100.0;
	tmp->avr = 0.0;

	sprintf(ch, "%s/Uorder_data.train", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_tr; i++)
	{
		fscanf(fp, "%d%d%lf", &(info->trIdx[i][0]), &(info->trIdx[i][1]), &(info->trR[i]));
		info->trIdx[i][0]--;
		info->trIdx[i][1]--;
		tmp->Uorder[i] = info->trIdx[i][1];
		tmp->Ur[i] = info->trR[i];
		tmp->avr += info->trR[i] * info->trR[i];
		if (tmp->MaxR < tmp->Ur[i])
			tmp->MaxR = tmp->Ur[i];
		if (tmp->MinR > tmp->Ur[i])
			tmp->MinR = tmp->Ur[i];
	}
	fclose(fp);
	tmp->avr = sqrt(tmp->avr);

	sprintf(ch, "%s/data.test", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_ts; i++)
	{
		fscanf(fp, "%d%d%lf", &(info->tsIdx[i][0]), &(info->tsIdx[i][1]), &(info->tsR[i]));
		info->tsIdx[i][0]--;
		info->tsIdx[i][1]--;
	}
	fclose(fp);

	sprintf(ch, "%s/Item_Map_User.train", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_tr; i++)
	{
		double tpp_dbl;
		fscanf(fp, "%d%d%lf", &tp, &(tmp->Vorder[i]), &tpp_dbl);
		tmp->Vorder[i]--;
	}
	fclose(fp);

	sprintf(ch, "%s/User_Map_Item.train", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_tr; i++)
	{
		fscanf(fp, "%d%d%d", &tp, &tpp, &(tmp->UmapV[i]));
		tmp->UmapV[i]--;
	}
	fclose(fp);

	sprintf(ch, "%s/User_num.txt", path);
	fp = fopen(ch, "r");
	for (i = 0; i < nU; i++)
	{
		fscanf(fp, "%d%d", &(tmp->Unum[i][0]), &(tmp->Unum[i][1]));
		tmp->Unum[i][0]--;
	}
	fclose(fp);

	sprintf(ch, "%s/Item_num.txt", path);
	fp = fopen(ch, "r");
	for (i = 0; i < nV; i++)
	{
		fscanf(fp, "%d%d", &(tmp->Vnum[i][0]), &(tmp->Vnum[i][1]));
		tmp->Vnum[i][0]--;
	}
	fclose(fp);

	tmp->Meanr = 0.0;
	for (i = 0; i < num_tr; i++)
	{
		tmp->Ur[i] -= tmp->MinR;
		tmp->Ur[i] /= (tmp->MaxR - tmp->MinR);
		tmp->Meanr += tmp->Ur[i];
	}
	tmp->Meanr /= num_tr;
	//	tmp->Meanr = 0.669434;
	for (i = 0; i < num_tr; i++)
	{
		tmp->Ur[i] -= tmp->Meanr;
		tmp->Ue[i] = tmp->Ur[i];
		tmp->Ve[tmp->UmapV[i]] = tmp->Ue[i];
	}

	return 0;
}

int readData2(FVL_INFO* info, FVL_TMP* tmp, char* path) // only include ratings in the training set
{
	FILE* fp;
	char ch[100];

	int nU;
	int nV;
	int num_tr;

	int i, tp, tpp;

	sprintf(ch, "%s/data2.statistics", path);
	fp = fopen(ch, "r");
	fscanf(fp, "%d%d", &nU, &nV);
	fscanf(fp, "%d", &num_tr);
	fclose(fp);
	info->nU = nU;
	info->nV = nV;
	info->N = nU + nV;
	info->num_tr = num_tr;
	info->num_ts = 0;

	info->U = (double**)malloc(info->N * sizeof(double*));
	info->V = info->U + nU;
	info->trIdx = (int**)malloc(num_tr * sizeof(int*));
	info->tsIdx = NULL;
	info->trIdx[0] = (int*)malloc(2 * num_tr * sizeof(int));
	for (i = 1; i < num_tr; i++)
		info->trIdx[i] = info->trIdx[0] + i * 2;
	info->trR = (double*)malloc(2 * num_tr * sizeof(double));
	info->ptrR = info->trR + num_tr;
	info->tsR = NULL;
	info->ptsR = NULL;

	tmp->eigV = (double*)malloc((info->N + 3 * info->num_tr) * sizeof(double));
	tmp->Ur = tmp->eigV + info->N;
	tmp->Ue = tmp->Ur + info->num_tr;
	tmp->Ve = tmp->Ue + info->num_tr;
	tmp->Uorder = (int*)malloc((3 * info->num_tr + 2 * info->N)* sizeof(int));
	tmp->Vorder = tmp->Uorder + info->num_tr;
	tmp->UmapV = tmp->Vorder + info->num_tr;

	tmp->Unum = (int**)malloc(info->N * sizeof(int*));
	tmp->Vnum = tmp->Unum + info->nU;
	for (i = 0; i < info->N; i++)
		tmp->Unum[i] = tmp->UmapV + info->num_tr + i * 2;

	tmp->Y = info->U;
	tmp->dY = (double**)malloc(info->N * sizeof(double*));

	tmp->MaxR = 0.0;
	tmp->MinR = 100.0;
	tmp->avr = 0.0;

	sprintf(ch, "%s/Uorder_data.train", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_tr; i++)
	{
		fscanf(fp, "%d%d%lf", &(info->trIdx[i][0]), &(info->trIdx[i][1]), &(info->trR[i]));
		info->trIdx[i][0]--;
		info->trIdx[i][1]--;
		tmp->Uorder[i] = info->trIdx[i][1];
		tmp->Ur[i] = info->trR[i];
		tmp->avr += info->trR[i] * info->trR[i];
		if (tmp->MaxR < tmp->Ur[i])
			tmp->MaxR = tmp->Ur[i];
		if (tmp->MinR > tmp->Ur[i])
			tmp->MinR = tmp->Ur[i];
	}
	fclose(fp);
	tmp->avr = sqrt(tmp->avr);

	sprintf(ch, "%s/Item_Map_User.train", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_tr; i++)
	{
		fscanf(fp, "%d%d%d", &tp, &(tmp->Vorder[i]), &tpp);
		tmp->Vorder[i]--;
	}
	fclose(fp);

	sprintf(ch, "%s/User_Map_Item.train", path);
	fp = fopen(ch, "r");
	for (i = 0; i < num_tr; i++)
	{
		fscanf(fp, "%d%d%d", &tp, &tpp, &(tmp->UmapV[i]));
		tmp->UmapV[i]--;
	}
	fclose(fp);

	sprintf(ch, "%s/User_num.txt", path);
	fp = fopen(ch, "r");
	for (i = 0; i < nU; i++)
	{
		fscanf(fp, "%d%d", &(tmp->Unum[i][0]), &(tmp->Unum[i][1]));
		tmp->Unum[i][0]--;
	}
	fclose(fp);

	sprintf(ch, "%s/Item_num.txt", path);
	fp = fopen(ch, "r");
	for (i = 0; i < nV; i++)
	{
		fscanf(fp, "%d%d", &(tmp->Vnum[i][0]), &(tmp->Vnum[i][1]));
		tmp->Vnum[i][0]--;
	}
	fclose(fp);

	tmp->Meanr = 0.0;
	for (i = 0; i < num_tr; i++)
	{
		tmp->Ur[i] -= tmp->MinR;
		tmp->Ur[i] /= (tmp->MaxR - tmp->MinR);
		tmp->Meanr += tmp->Ur[i];
	}
	tmp->Meanr /= num_tr;
	//	tmp->Meanr = 0.669434;
	for (i = 0; i < num_tr; i++)
	{
		tmp->Ur[i] -= tmp->Meanr;
		tmp->Ue[i] = tmp->Ur[i];
		tmp->Ve[tmp->UmapV[i]] = tmp->Ue[i];
	}

	return 0;
}

int saveSigma(FVL_INFO* info, char* path,double cur_lambda)
{
	char ch[100];
	FILE* fp;
	int i;

	sprintf(ch, "%s/%.6lfsigma.txt", cur_lambda,path);
	fp = fopen(ch, "w");
	fprintf(fp, "%d\n", info->p);
	for (i = 0; i < info->p; i++)
		fprintf(fp, "%g\n", info->sigma[i]);
	fclose(fp);
	freopen("CON", "w", stdout);
	return 0;
}

double para_tune(double cur_lambda, double cur_epsilon, int cur_ty, FVL_INFO *cur_info, FVL_TMP *cur_tmp, char *cur_path){
	 cur_tmp->epsilon= cur_epsilon;
	 cur_tmp->lambda = cur_lambda;
	 if (cur_ty == 0)
		 readData(cur_info, cur_tmp, cur_path);
	 else
		 readData2(cur_info, cur_tmp, cur_path);
	 cur_tmp->M = cur_info->N / 20.0;
	 printf("MaxR: %g\n", cur_tmp->MaxR);
	 printf("MinR: %g\n", cur_tmp->MinR);
	 printf("Meanr: %g\n", cur_tmp->Meanr);
	 printf("M: %g\n", cur_tmp->M);
	 printf("epsilon: %g\n", cur_tmp->epsilon);
	 printf("lambda: %g\n", cur_tmp->lambda);
	 printf("path: %s\n",cur_path);
	 if (cur_ty == 0)
		 globalOpt(cur_info, cur_tmp);
	 else
		 globalOpt2(cur_info, cur_tmp);
	 saveSigma(cur_info, cur_path,cur_lambda);
	 return (RMSE(cur_info->ptsR, cur_info->tsR, cur_info->num_ts));

}

int main()
{
	omp_set_num_threads(NUM_OF_THREADS);

	FVL_INFO info;
	FVL_TMP tmp;
	char path[100];
	int i, ty;
	
	LAMBDA = 1.0;
	for (int i = 0; i < 5; i++){
		sprintf(path, "%s%d/", DATA_PATH, i);
		double lambda_li = 0.0;
		double lambda_ri = 1.0;
		while (lambda_ri - lambda_li>0.001){
			double lambda_lm = lambda_li + (lambda_ri - lambda_li) / 3.0;
			double lambda_rm = lambda_ri - (lambda_ri - lambda_li) / 3.0;
			double tst_rmse_li = para_tune(lambda_lm, EPSILON, 0, &info, &tmp, path);
			double tst_rmse_ri = para_tune(lambda_rm, EPSILON, 0, &info, &tmp, path);
			if (tst_rmse_li > tst_rmse_ri)
				lambda_li = lambda_lm;
			else
				lambda_ri = lambda_rm;
		}
	}
	
	/*sprintf(path, DATA_PATH);
	tmp.epsilon = EPSILON;
	tmp.lambda = LAMBDA;
	ty = PROCESS_TYPE;
	
	if (ty == 0)
		readData(&info, &tmp, path);
	else
		readData2(&info, &tmp, path);
	tmp.M = info.N / 20.0;
	printf("MaxR: %g\n", tmp.MaxR);
	printf("MinR: %g\n", tmp.MinR);
	printf("Meanr: %g\n", tmp.Meanr);
	printf("M: %g\n", tmp.M);
	printf("epsilon: %g\n", tmp.epsilon);
	printf("lambda: %g\n", tmp.lambda);
	printf("path: %s\n", path);
	if (ty == 0)
		globalOpt(&info, &tmp);
	else
		globalOpt2(&info, &tmp);

	saveSigma(info, path);*/

	return 0;
}

