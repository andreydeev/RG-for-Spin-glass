import numpy as np
import random
from numba import njit, prange
from numba import int32, float32, float64
import itertools
from gmpy2 import mpfr, fsum
import gmpy2
import time
import math

import os
import matplotlib.pyplot as plt
import json

# NUMBA for fast iter implementation (prob shoud have done in cython like multilayer trace calc cause found numba unrealiable, but oh well)
@njit
def iter_V(V, h, v, Nvisible, Nhidden, Ninput, factor, Ninternal):
    dV = np.zeros((Nvisible, Nhidden))
    for i in range(Ninput):
        for l in range(Nhidden):
            h_l = 0
            for j in range(Nvisible):
                h_l += v[i,j]*V[j,l]
            eh2=np.exp(2.*h_l)
            for j in range(Nvisible):
                dV[j][l]+=(1./Ninput)*v[i][j]*(eh2-1.)/(eh2+1.)
            for k in range(factor):
                if random.random() > 1./(eh2+1.):
                    h[i*factor+k][l]=1
                else:
                    h[i*factor+k][l]=-1
        for k in range(factor):
            for j in range(Nvisible):
                v_j = 0
                for l in range(Nhidden):
                    v_j+=h[i*factor+k][l]*V[j][l]
                ev2 = np.exp(2.*v_j)
                for l in range(Nhidden):
                    dV[j][l]-=(1./Ninternal)*h[i*factor+k][l]*(ev2-1.)/(ev2+1.)
    s = np.sum(np.square(dV))
    V+=dV
    return s/(Nvisible*Nhidden)

class RBM:
    def __init__(self, Nvisible, Nhidden, factor):
        self.V = np.random.uniform(size=(Nvisible, Nhidden))
        self.Nvisible = Nvisible
        self.Nhidden = Nhidden
        self.factor = factor
        self.h = []
        self.Ninternal = 1
    def train(self, v, n_iter, verbose=False):
        self.Ninternal = self.factor * len(v)
        self.h = np.zeros((self.Ninternal, self.Nhidden))
        for i in range(n_iter):
            x = iter_V(self.V, self.h, v, self.Nvisible, self.Nhidden, len(v), self.factor, self.Ninternal)
            if x < 1e-7:
                break
            if verbose:
                print(i, x)
        
class ML_RBM:
    def __init__(self, configuration, factor=10):
        self.Nvisible = configuration[0]
        self.hidden_layers = configuration[1:]
        self.matrices = []
        self.train_sets = []
        self.factor = factor

    def train(self, v_init, n_iter, verbose=False):
        Ninput = len(v_init)
        self.matrices = []
        self.train_sets = []

        for layer_ind in range(len(self.hidden_layers)):
            Nhidden = self.hidden_layers[layer_ind]
            if layer_ind > 0:
                visible = self.hidden_layers[layer_ind - 1]
                res_v = np.zeros((Ninput, visible))
                # generate new train set for inner layers
                for i in range(Ninput):
                    for k in range(visible):
                        hz = 0
                        for l in range(len(v[i])):
                            hz += v[i][l] * self.matrices[-1][l, k]
                        res_v[i, k] = 1 if random.random() > 1 / (1 + np.exp(2 * hz)) else -1
                v = res_v
            else:
                visible = self.Nvisible
                v = np.copy(v_init)

            # create RBM model
            model = RBM(visible, Nhidden, self.factor)
            model.train(v, n_iter, True)

            self.matrices.append(model.V)
            self.train_sets.append(v)

def get_confs(N):
    arr = []
    for i in range(2 ** (N)):
        arr.append([1 if int(x) > 0 else -1 for x in format(i, "0{}b".format(N))])
    return arr

# probability function
def P(J, v, beta, Z=1):
    return np.exp(-np.sum(J * np.outer(v, v)) * beta) / Z


def calc_Z(J, v, beta):
    z = 0
    v_seen = set()
    for item in v:
        h = "".join([str(x) for x in item])
        if h not in v_seen:
            v_seen.add(h)
            z += P(J, item, beta)
    return z


# implement gibbs distribution
def gibbs_sample(x1, J, beta):
    x = np.copy(x1)
    for i in range(len(x)):
        x[i] = -1
        e1 = P(J, x, beta)
        x[i] = 1
        e2 = P(J, x, beta)
        rand = np.random.uniform(0, e1 + e2)
        if rand < e1:
            x[i] = -1
    return x


def generate_J(N=4, sq=True):
    J_location = "./Js/J_{}.npy".format(N)
    # generate J if needed
    if sq:
        N = N ** 2
    if not os.path.isfile(J_location):
        J = np.random.normal(0, 1.0 / N, N ** 2)
        J = np.reshape(J, (-1, N))
        np.fill_diagonal(J, 0)
        for i in range(len(J)):
            for j in range(len(J[i])):
                J[i, j] = J[j, i]
        # save J
        np.save(J_location, J)
    else:
        J = np.load(J_location)
    return J


def generate_J_NN(N=4, sq=True):
    J_location = "./Js/J_{}_NN.npy".format(N)
    # generate J if needed
    if sq:
        N = N ** 2
    if not os.path.isfile(J_location):
        J = np.random.normal(0, 1.0 / N, N ** 2)
        J = np.reshape(J, (-1, N))
        np.fill_diagonal(J, 0)
        for i in range(len(J)):
            for j in range(len(J[i])):
                J[i, j] = J[j, i]

        realJ = np.zeros((N, N))
        N = int(np.sqrt(N))
        for i in range(len(J)):
            x = i % N
            y = i // N
            print(i, x, y)
            # up
            coordY = 0
            if y == 0:
                coordY = (N - 1)
            else:
                coordY = y - 1
            coord = x + coordY * N
            realJ[i, coord] = realJ[coord, i] = J[i, coord]

            # down
            coordY = 0
            if y == N - 1:
                coordY = 0
            else:
                coordY = y + 1
            coord = x + coordY * N
            realJ[i, coord] = realJ[coord, i] = J[i, coord]

            # left
            coordX = 0
            if x == 0:
                coordX = N - 1
            else:
                coordX = x - 1
            coord = coordX + y * N
            realJ[i, coord] = realJ[coord, i] = J[i, coord]

            # right
            coordX = 0
            if x == N - 1:
                coordX = 0
            else:
                coordX = x + 1
            coord = coordX + y * N
            realJ[i, coord] = realJ[coord, i] = J[i, coord]

        J = realJ
        # save J
        np.save(J_location, J)
    else:
        J = np.load(J_location)
    return J


def generate_train(J, N, training_size, beta, override=False, sq=True):
    train_X = []
    if sq:
        N = N ** 2
    # for caching purposes
    train_location = "./train_sets/train_{}_{}.json".format(N, beta)
    if not os.path.isfile(train_location) or override:
        for i in range(training_size):
            x1 = np.random.rand(N)
            x1[x1 < 0.5] = -1
            x1[x1 > 0.5] = 1
            # do 2 gibbs sampling just to be sure
            x1 = gibbs_sample(x1, J, beta)
            x1 = gibbs_sample(x1, J, beta)
            train_X.append(x1)

        # save training data
        to_save = [x.tolist() for x in train_X]
        with open(train_location, "w") as fp:
            json.dump(to_save, fp)
    else:
        with open(train_location) as fp:
            train_X = json.load(fp)
    train_X = np.array(train_X)
    return train_X

