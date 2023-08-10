from dynamic_signed import learn_a_dynamic_signed_graph
import pandas as pd
import numpy as np
from time import time

if __name__ == "__main__":
    df = pd.read_csv("X_d_10_30.csv", header=None)
    T = 10
    X_noisy = np.array(df)
    DIM, NUM = X_noisy.shape
    # print(DIM,NUM)
    X = []
    NUM = NUM // T
    # print(DIM, NUM)
    for i in range(T):
        X.append(X_noisy[:, i * NUM:(i + 1) * NUM])
        # print(X[i].shape)

    _, params = learn_a_dynamic_signed_graph(X, 0.12, 0.85)

    tic = time()
    w, params = learn_a_dynamic_signed_graph(X, 0.12, 0.85, **params)
    toc = time() - tic
    print(toc)
    w = np.array(w['+']) - np.array(w['-'])
    # print(w[0].shape)
    # print(w.shape)
    W_i = np.zeros((DIM, DIM))
    W = np.zeros((DIM, DIM * T))
    upper = np.triu_indices_from(W_i, k=1)
    for i, w_i in enumerate(w):
        W_i[upper] = np.squeeze(w_i)
        W[:, i * DIM:(i + 1) * DIM] = W_i + W_i.T

    df = pd.DataFrame(W)
    df.to_csv('W_dynSGL_10_30.csv', index=False, header=False)
