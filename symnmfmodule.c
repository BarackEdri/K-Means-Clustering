#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* ----------- SYM ----------- */
static PyObject *sym(PyObject *self, PyObject *args)
{
    /* Initalize */
    int N, d, i, j;
    double num;
    PyObject *X, *item, *point, *python_A, *python_A_datapoint, *python_A_point;
    double **new_X, **A;
    double *p;

    /* Parsing */
    if(!PyArg_ParseTuple(args, "O", &X))
    {
        return NULL;
    }

    /* Get N and d */
    N = PyObject_Length(X);
    if(N < 0) {
        return NULL;
    }
    item = PyList_GetItem(X, 0);  
    d = PyObject_Length(item);
    if(d < 0) {
        return NULL;
    }

    /* Create new_X to parse X from pyhton to C */
    p = calloc(N * d, sizeof(double));
    new_X = calloc(N, sizeof(double));
    for(i = 0; i < N; i++){
        new_X[i] = p + (i * d);
    }

    /* Populate new_X */
    for (i = 0; i < N; i++) {
        item = PyList_GetItem(X, i);
        for(j = 0; j < d; j++){
            point = PyList_GetItem(item, j);
            num = PyFloat_AsDouble(point);
            new_X[i][j] = num;
        }
    }

    /* Get the A matrix */
    A = sym_c(new_X, N, d);

    /* Convert A in C to A in python*/
    python_A = PyList_New(N);

    for(i = 0; i < N; i++){
        python_A_datapoint = PyList_New(N);
        for(j = 0; j < N; j++){
            python_A_point = Py_BuildValue("d", A[i][j]);
            PyList_SetItem(python_A_datapoint, j, python_A_point);
        }
        PyList_SetItem(python_A, i, python_A_datapoint);
    }

    free(A[0]);
    free(A);
    free(new_X[0]);
    free(new_X);

    return python_A;
}
/* ----------- SYM ----------- */


/* ----------- DDG ----------- */
static PyObject *ddg(PyObject *self, PyObject *args)
{
    /* Initalize */
    int N, d, i, j;
    double num;
    PyObject *X, *item, *point, *python_D, *python_D_datapoint, *python_D_point;
    double **new_X, **D;
    double *p;

    /* Parsing */
    if(!PyArg_ParseTuple(args, "O", &X))
    {
        return NULL;
    }

    /* Get N and d */
    N = PyObject_Length(X);
    if(N < 0) {
        return NULL;
    }
    item = PyList_GetItem(X, 0);  
    d = PyObject_Length(item);
    if(d < 0) {
        return NULL;
    }

    /* Create new_X to parse X from pyhton to C */
    p = calloc(N * d, sizeof(double));
    new_X = calloc(N, sizeof(double));
    for(i = 0; i < N; i++){
        new_X[i] = p + (i * d);
    }

    /* Populate new_X */
    for (i = 0; i < N; i++) {
        item = PyList_GetItem(X, i);
        for(j = 0; j < d; j++){
            point = PyList_GetItem(item, j);
            num = PyFloat_AsDouble(point);
            new_X[i][j] = num;
        }
    }

    /* Get the D matrix */
    D = ddg_c(new_X, N, d);

    /* Convert D in C to D in python*/
    python_D = PyList_New(N);

    for(i = 0; i < N; i++){
        python_D_datapoint = PyList_New(N);
        for(j = 0; j < N; j++){
            python_D_point = Py_BuildValue("d", D[i][j]);
            PyList_SetItem(python_D_datapoint, j, python_D_point);
        }
        PyList_SetItem(python_D, i, python_D_datapoint);
    }

    free(D[0]);
    free(D);
    free(new_X[0]);
    free(new_X);
    return python_D;
}
/* ----------- DDG ----------- */


/* ----------- NORM ----------- */
static PyObject *norm(PyObject *self, PyObject *args)
{
    /* Initalize */
    int N, d, i, j;
    double num;
    PyObject *X, *item, *point, *python_W, *python_W_datapoint, *python_W_point;
    double **new_X, **W;
    double *p;

    /* Parsing */
    if(!PyArg_ParseTuple(args, "O", &X))
    {
        return NULL;
    }

    /* Get N and d */
    N = PyObject_Length(X);
    if(N < 0) {
        return NULL;
    }
    item = PyList_GetItem(X, 0);  
    d = PyObject_Length(item);
    if(d < 0) {
        return NULL;
    }

    /* Create new_X to parse X from pyhton to C */
    p = calloc(N * d, sizeof(double));
    new_X = calloc(N, sizeof(double));
    for(i = 0; i < N; i++){
        new_X[i] = p + (i * d);
    }

    /* Populate new_X */
    for (i = 0; i < N; i++) {
        item = PyList_GetItem(X, i);
        for(j = 0; j < d; j++){
            point = PyList_GetItem(item, j);
            num = PyFloat_AsDouble(point);
            new_X[i][j] = num;
        }
    }

    /* Get the W matrix */
    W = norm_c(new_X, N, d);

    /* Convert W in C to W in python*/
    python_W = PyList_New(N);

    for(i = 0; i < N; i++){
        python_W_datapoint = PyList_New(N);
        for(j = 0; j < N; j++){
            python_W_point = Py_BuildValue("d", W[i][j]);
            PyList_SetItem(python_W_datapoint, j, python_W_point);
        }
        PyList_SetItem(python_W, i, python_W_datapoint);
    }

    free(W[0]);
    free(W);
    free(new_X[0]);
    free(new_X);
    return python_W;
}
/* ----------- NORM ----------- */


/* ----------- SYMNMF ----------- */
static PyObject *symnmf(PyObject *self, PyObject *args)
{
    /* Initalize */
    int N, K, i, j, max_iter;
    double num, epsilon;
    PyObject *W, *H_0, *item, *point, *python_H_i, *python_H_i_datapoint, *python_H_i_point;
    double **new_W, **new_H_0, **H_i;
    double *p, *q;

    /* Parsing */
    if(!PyArg_ParseTuple(args, "OOiiid", &H_0, &W, &N, &K, &max_iter, &epsilon))
    {
        return NULL;
    }

    /* Create new_W to parse W from pyhton to C */
    p = calloc(N * N, sizeof(double));
    new_W = calloc(N, sizeof(double));
    for(i = 0; i < N; i++){
        new_W[i] = p + (i * N);
    }

    /* Create new_H_0 to parse H_0 from pyhton to C */
    q = calloc(N * K, sizeof(double));
    new_H_0 = calloc(N, sizeof(double));
    for(i = 0; i < N; i++){
        new_H_0[i] = q + (i * K);
    }

    /* Populate new_W */
    for (i = 0; i < N; i++) {
        item = PyList_GetItem(W, i);
        for(j = 0; j < N; j++){
            point = PyList_GetItem(item, j);
            num = PyFloat_AsDouble(point);
            new_W[i][j] = num;
        }
    }

    /* Populate new_H_0 */
    for (i = 0; i < N; i++) {
        item = PyList_GetItem(H_0, i);
        for(j = 0; j < K; j++){
            point = PyList_GetItem(item, j);
            num = PyFloat_AsDouble(point);
            new_H_0[i][j] = num;
        }
    }

    /* Get the W matrix */
    H_i = symnmf_c(new_H_0, new_W, N, K, max_iter, epsilon);

    /* Convert H_i in C to H_i in python*/
    python_H_i = PyList_New(N);

    for(i = 0; i < N; i++){
        python_H_i_datapoint = PyList_New(K);
        for(j = 0; j < K; j++){
            python_H_i_point = Py_BuildValue("d", H_i[i][j]);
            PyList_SetItem(python_H_i_datapoint, j, python_H_i_point);
        }
        PyList_SetItem(python_H_i, i, python_H_i_datapoint);
    }

    free(H_i[0]);
    free(H_i);
    free(new_H_0[0]);
    free(new_H_0);
    free(new_W[0]);
    free(new_W);

    return python_H_i;
}
/* ----------- SYMNMF ----------- */


/* ----------- MATRIXS ----------- */
static PyMethodDef matrixsMethods[] = {
    {"sym", 
    (PyCFunction) sym,
    METH_VARARGS,
    PyDoc_STR("The method expects to get : X - The list of the data points.")},

    {"ddg", 
    (PyCFunction) ddg,
    METH_VARARGS,
    PyDoc_STR("The method expects to get : X - The list of the data points.")},

    {"norm", 
    (PyCFunction) norm,
    METH_VARARGS,
    PyDoc_STR("The method expects to get : X - The list of the data points.")},

    {"symnmf", 
    (PyCFunction) symnmf,
    METH_VARARGS,
    PyDoc_STR("The method expects to get : H_0 - intialize matrix, W - norm matrix, N - number of datapoints, K - user argument, max_iter - max iterition, epsilon - as is.")},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = 
{
    PyModuleDef_HEAD_INIT, 
    "myMatrixs", 
    NULL, 
    -1, 
    matrixsMethods 
};

PyMODINIT_FUNC PyInit_myMatrixs(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) 
    {
        return NULL;
    }
    return m;
}
/* ----------- MATRIXS ----------- */