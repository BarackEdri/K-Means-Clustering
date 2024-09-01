import sys
import pandas as pd
import numpy as np
import math
import myMatrixs

# X - DataPoints
# N - number of DataPoints
# d - number of points in each DataPoints
# A - similarity matrix
# D - diagonal matrix
# W - norm matrix
# m - the average of all entries of W
# H_0 - initalize symnmf matrix
# final_H - the symnmf matrix

def print_Matrix(matrix, N, M):
    for i in range(N):
        for j in range(M - 1):
            print('{:.4f}'.format(matrix[i][j]), end = ",")
        print('{:.4f}'.format(matrix[i][M - 1]))

try:
    # Intalize
    if len(sys.argv) == 4:
        K = int(sys.argv[1])
        goal = sys.argv[2]
        file_name = sys.argv[3]
        epsilon = 0.0001
        max_iter = 300
    else:
        sys.exit("An Error Has Occurred")

    # Convert the points from the file to a Numpy table called X
    X = pd.read_csv(file_name, header = None)
    X = X.to_numpy()
    N = X.shape[0]
    d = X.shape[1]
    np.random.seed(0)

    # Check for the right goal
    if goal == "symnmf":
        W = myMatrixs.norm(X.tolist())
        W = np.array(W)
        m = np.mean(W)
        H_0 = np.random.uniform(0, 2 * math.sqrt(m / K), (N, K))
        final_H = myMatrixs.symnmf(H_0.tolist(), W.tolist(), N, K, max_iter, epsilon)
        print_Matrix(final_H, N, K)

    if goal == "sym":
        A = myMatrixs.sym(X.tolist())
        A = np.array(A)
        print_Matrix(A, N, N)
       
    if goal == "ddg":
        D = myMatrixs.ddg(X.tolist())
        D = np.array(D)
        print_Matrix(D, N, N)
    
    if goal == "norm":
        W = myMatrixs.norm(X.tolist())
        W = np.array(W)
        print_Matrix(W, N, N)

except:
    sys.exit("An Error Has Occurred")
