#include "symnmf.h"

int main(int argc, char *argv[])
{
    char *goal, *file_name;
    FILE *fp;
    int N = 0 , d = 0, flag = 1, i, j;
    double *p;
    double **X;
    double n;
    char c;
    double **M = NULL;

    /* Get the arguments from CMD */
    if (argc == 3)
    {
        goal = argv[1];
        file_name = argv[2];
    }
    else
    {
        printf("%s", "An Error Has Occurred");
        return 1;
    }

    /* Read the file */
    fp = fopen(file_name, "r");
    while (fscanf(fp, "%lf%c", &n, &c) == 2)
    {
        if(flag)
        {
            d = d + 1;
        }    
        if (c == '\n')
        {
            N = N + 1;
            flag = 0;
        }
    }
    if(d == 0 || N == 0)
    {
        printf("%s", "An Error Has Occurred");
        return 1;
    }
    rewind(fp);

    /* Create the DataPoints X */
    p = calloc(N * d, sizeof(double));
    X = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        X[i] = p + i * d;
    }
    
    for (i = 0; i < N; i++)
    {  
        for(j = 0; j < d ; j++)
        {
            if (fscanf(fp, "%lf%c", &n, &c) == 2){
                X[i][j] = n;
            }
        }
    }
    fclose(fp);

    /* Create the wanted matrix */
    if(strcmp(goal, "sym") == 0){
        M = sym_c(X, N, d);
    }

    if(strcmp(goal, "ddg") == 0){
        M = ddg_c(X, N, d);
    }

    if(strcmp(goal, "norm") == 0){
        M = norm_c(X, N, d);
    }
    
    for(i = 0; i < N; i++)
    {
        for (j = 0; j < N - 1; j++)
        {
            printf("%.4f%s",M[i][j],",");
        }
        printf("%.4f%s",M[i][N - 1],"\n");
    }

    free(M[0]);
    free(M);
    free(X[0]);
    free(X);

    return 0;
}

/* ----------- SYM ----------- */
double distance(double *i, double *j, int d)
{
    int k;
    double sum = 0;

    for(k = 0; k < d; k++)
    {
        sum = sum + pow((i[k] - j[k]), 2.0);
    }
    return sum;
}

double **sym_c(double **X, int N, int d)
{
    double **A;
    double *p;
    int i, j;
    p = calloc(N * N, sizeof(double));
    A = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        A[i] = p + i * N;
    }

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            if(i == j){
                A[i][j] = 0.0;
            }
            else{
                A[i][j] = exp(-(distance(X[i], X[j], d)/2));
            }
        }
    }
    return A;
}
/* ----------- SYM ----------- */

/* ----------- DDG ----------- */
double **ddg_c(double **X, int N, int d)
{
    double **A, **D;
    double *p;
    int i, j;
    double sum = 0.0;
    p = calloc(N * N, sizeof(double));
    D = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        D[i] = p + i * N;
    }

    A = sym_c(X, N, d);
    for(i = 0; i < N; i++){
        sum = 0.0;
        for(j = 0; j < N; j++){
            sum = sum + A[i][j];
        }
        D[i][i] = sum;
    }
    free(A[0]);
    free(A);
    return D;
}
/* ----------- DDG ----------- */

/* ----------- NORM ----------- */
double **matrix_product(double **X1, double **X2, int N, int K, int M)
{
    /*   X1 - is a matrix (N * K)  X2 - is a matrix (K * M)   */
    double **R;
    double *p;
    int i, j, t;
    double sum = 0.0;
    p = calloc(N * M, sizeof(double));
    R = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        R[i] = p + i * M;
    }
    for(i = 0; i < N; i++){
        for(j = 0; j < M; j++){
            for(t = 0; t < K; t++){
                sum = sum + X1[i][t] * X2[t][j];
            }
            R[i][j] = sum;
            sum = 0.0;
        }
    }
    return R;
}

double **norm_c(double **X, int N, int d)
{
    double **A, **D, **W, **W1;
    int i;
    A = sym_c(X, N, d);
    D = ddg_c(X, N, d);
    for(i = 0; i < N; i++){
        D[i][i] = pow(D[i][i], -0.5);
    }
    
    W1 = matrix_product(D, A, N, N, N);
    W = matrix_product(W1, D, N, N, N);

    free(A[0]);
    free(A);
    free(D[0]);
    free(D);
    free(W1[0]);
    free(W1);
    return(W);
}
/* ----------- NORM ----------- */

/* ----------- SYMNMF ----------- */
double **matrix_transpose(double **X, int N, int K)
{
    /*   X - is a matrix (N * K)   */
    double **R;
    double *p;
    int i, j;

    p = calloc(N * K, sizeof(double));
    R = calloc(K, sizeof(double *));
    for(i = 0; i < K; i++){
        R[i] = p + i * N;
    }

    for(i = 0; i < K; i++){
        for(j = 0; j < N; j++){
            R[i][j] = X[j][i];
        }
    }

    return R;
}

double **symnmf_c(double **H_0, double **W, int N, int K, int max_iter, double epsilon)
{
    double **H_i, **H_i1, **H_i_T, **P1, **P2, **P3, **P4, **H_temp;
    double *p, *q, *z;
    int i, j;
    double b = 0.5, norm_f = 0.0;
    int iteration_number = 0;

    /* Intilize H_i to be H_0 */
    z = calloc(N * K, sizeof(double));
    H_i = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        H_i[i] = z + i * K;
    }
    for(i = 0; i < N; i++){
        for(j = 0; j < K; j++){
            H_i[i][j] = H_0[i][j];
        }
    }

    /* Intilize H_i+1 to be Matrix of zeros */
    p = calloc(N * K, sizeof(double));
    H_i1 = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        H_i1[i] = p + i * K;
    }

    /* Intilize P4 for |H_i+1 - H_i| */
    q = calloc(N * K, sizeof(double));
    P4 = calloc(N, sizeof(double *));
    for(i = 0; i < N; i++){
        P4[i] = q + i * K;
    }

    while(1)
    {
        iteration_number = iteration_number + 1;
        H_i_T = matrix_transpose(H_i, N, K);
        P1 = matrix_product(W, H_i, N, N, K);
        P2 = matrix_product(H_i, H_i_T, N, K, N);
        P3 = matrix_product(P2, H_i, N, N, K);
        
        for(i = 0; i < N; i++){
            for(j = 0; j < K; j++){
                H_i1[i][j] = H_i[i][j] * ((1 - b) + (b * (P1[i][j] / P3[i][j])));
            }
        }

        free(H_i_T[0]);
        free(H_i_T);
        free(P1[0]);
        free(P1);
        free(P2[0]);
        free(P2);
        free(P3[0]);
        free(P3);

        for(i = 0; i < N; i++){
            for(j = 0; j < K; j++){
                P4[i][j] = H_i1[i][j] - H_i[i][j];
            }
        }

        norm_f = 0.0;
        for(i = 0; i < N; i++){
            for(j = 0; j < K; j++){
                norm_f = norm_f + pow(P4[i][j], 2);
            }
        }

        H_temp = H_i;
        H_i = H_i1;
        H_i1 = H_temp;
        
        if((iteration_number == max_iter) || (norm_f < epsilon)){
            break;
        }
    }
    free(P4[0]);
    free(P4);
    free(H_i1[0]);
    free(H_i1);
    
    return H_i;
}
/* ----------- SYMNMF ----------- */

