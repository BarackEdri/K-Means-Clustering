import sys
import pandas as pd
import numpy as np
import math
import myMatrixs
import sklearn.metrics

# ---------- Analysis ----------
def isequal_datapoints(X, Y, D):
    for i in range(D):
        if X[i] != Y[i]:
            return False
    return True
# ---------- Analysis ----------

# ---------- Symnmf ----------
def using_symnmf(K, X, N):
    try:
        # Intalize
        epsilon = 0.0001
        max_iter = 300

        # Check for the right goal
        W = myMatrixs.norm(X.tolist())
        W = np.array(W)
        m = np.mean(W)
        H_0 = np.random.uniform(0, 2 * math.sqrt(m / K), (N, K))
        final_H = myMatrixs.symnmf(H_0.tolist(), W.tolist(), N, K, max_iter, epsilon)

        return final_H

    except:
        sys.exit("An Error Has Occurred")
# ---------- Symnmf ----------

# ---------- Kmeans ----------
def distance(p, q):
    # Calculate the distance between 2 points
    sumi = 0
    d = len(p)
    for i in range(d):
        sumi = sumi + pow((p[i]-q[i]), 2)
    return pow(sumi, 0.5)

def argmin(lst_u, x):
    # Calculate the distance between the first two points for the min value
    mini = distance(lst_u[0], x)
    index_u = 0
    K = len(lst_u)

    # Run over all u_k and find which one is the closet to x
    for i in range(1, K):
        dis = distance(lst_u[i], x)
        if dis < mini:
            mini = dis
            index_u = i

    return index_u
      
def updateCentroids(lst_k):
    u = []
    d = len(lst_k[0])

    # Run over each corainate in the vector
    for i in range(d):
        sumi = 0
        # Run over all vectors in the same cordinate
        for j in range(len(lst_k)):
            sumi = sumi + lst_k[j][i]
        u.append(sumi/len(lst_k))

    return u

def using_kmeans(K, input_data):
    try:    
        # Intalize parameters
        iteri = 300
        text = open(input_data, "r")
        lst_x = []
        lst_u_prev = []
        N = 0

        # Convert the points from the file to a list
        for line in text:
            data_point_string = line.split(",")
            data_point = []
            for i in range(len(data_point_string)):
                data_point.append(float(data_point_string[i]))
            lst_x.append(data_point)
            N = N + 1
        text.close()    

        # Create the u_k list
        for i in range(K):
            lst_u_prev.append(lst_x[i])

        # The algoritem
        condition = True
        iteration_number = 0
        while condition:

            # Intalize
            lst_u_new = []
            part1 = False
            part2 = False
            iteration_number = iteration_number + 1
            clusters = []
            for i in range(K):
                clusters.append([])
                    
            # Checking first condition
            if iteration_number < iteri:
                part1 = True

            # Insert the x_i to the right place in clusters 
            for i in range(N):
                index = argmin(lst_u_prev, lst_x[i])
                clusters[index].append(lst_x[i])

            # Checking second condition and update the centroid for u_k
            for i in range(K):
                lst_u_new.append(updateCentroids(clusters[i]))
                if distance(lst_u_new[i], lst_u_prev[i]) >= 0.0001:
                    part2 = True
                
            lst_u_prev = lst_u_new
            condition = part1 and part2

        # Returnring Clusters
        return clusters

    except:
        sys.exit("An Error Has Occurred")
# ---------- Kmeans ----------      

# ---------- Main ----------
try:
    # Intalize
    np.random.seed(0)
    if len(sys.argv) == 3:
        K = int(sys.argv[1])
        file_name = sys.argv[2]

    else:
        sys.exit("An Error Has Occurred")

    # Convert the points from the file to a Numpy table called X
    X = pd.read_csv(file_name, header = None)
    X = X.to_numpy()
    N = X.shape[0]
    D = X.shape[1]

    # Creating Symnmf_label
    H_Symnmf = using_symnmf(K, X, N)
    Symnmf_label = np.zeros(N, dtype=int)
    for i in range(N):
        max_col = -1
        max_index = -1
        for j in range(K):
            num = H_Symnmf[i][j]
            if num > max_col:
                max_col = num
                max_index = j
        Symnmf_label[i] = max_index

    # Creating Kmeans_label
    lst_Kmeans = using_kmeans(K, file_name)
    Kmeans_label = np.zeros(N, dtype=int)
    for i in range(N):
        for j in range(len(lst_Kmeans)):
            for k in range(len(lst_Kmeans[j])):
                if isequal_datapoints(X[i], lst_Kmeans[j][k], D):
                    Kmeans_label[i] = j


    # Calculating Silhouette_Score
    score_Symnmf = sklearn.metrics.silhouette_score(X, Symnmf_label)
    score_Kmeans = sklearn.metrics.silhouette_score(X, Kmeans_label)
    print("nmf: " + '{:.4f}'.format(score_Symnmf))
    print("kmeans: " + '{:.4f}'.format(score_Kmeans))

except:
    sys.exit("An Error Has Occurred")
# ---------- Main ----------     
    
    
