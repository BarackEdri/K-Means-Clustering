#include <stdlib.h>
#include <stdio.h> 
#include <math.h>
#include <string.h>

/* ----------- SYM ----------- */
double **sym_c(double **X, int N, int d);
double distance(double *i, double *j, int d);
/* ----------- SYM ----------- */

/* ----------- DDG ----------- */
double **ddg_c(double **X, int N, int d);
/* ----------- DDG ----------- */

/* ----------- NORM ----------- */
double **matrix_product(double **X1, double **X2, int N, int K, int M);
double **norm_c(double **X, int N, int d);
/* ----------- NORM ----------- */

/* ----------- SYMNMF ----------- */
double **matrix_transpose(double **X, int N, int K);
double **symnmf_c(double **H_0, double **W, int N, int K, int max_iter, double epsilon);
/* ----------- SYMNMF ----------- */