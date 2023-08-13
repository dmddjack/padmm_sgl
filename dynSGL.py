from dynamic_signed import learn_a_dynamic_signed_graph
import pandas as pd
import numpy as np
from time import time
import sys, getopt

if __name__ == "__main__":
    seed = None
    time_slots = None
    opts, args = getopt.getopt(sys.argv[1:], "s:t:")
    for opt, arg in opts:
        if opt == '-s':
            seed = int(arg)
        if opt == '-t':
            time_slots = int(arg)
    if seed is None or time_slots is None:
        raise IOError

    df = pd.read_csv("X_d_{}_{}.csv".format(time_slots, seed), header=None)
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

    _, params = learn_a_dynamic_signed_graph(X, 0.10, 0.75)

    tic = time()
    w, params = learn_a_dynamic_signed_graph(X, 0.10, 0.75, **params)
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
    df.to_csv("W_dynSGL_{}_{}.csv".format(time_slots, seed), index=False, header=False)
