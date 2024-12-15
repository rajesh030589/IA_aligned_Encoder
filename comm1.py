import itertools
import math
import numpy as np
from math import pi
import pickle
import math
from itertools import product


# from sympy import *
def dec2binary_array(n):
    X = np.zeros((2**n, n))
    for i in range(2**n):
        X[i, :] = np.array(
            list("".join("{:0>" + str(n) + "}").format(str(bin(i))[2:]))
        ).astype(int)
    return X


def binrep(n, r):
    return "{0:0{1}b}".format(n, r)


def rate_function(ber):
    ber = np.array(ber)
    return 1 + ber * np.log2(np.maximum(1e-9, ber)) + (1 - ber) * np.log2(1 - ber)


def generateGrayarr(n):
    arr = list()
    X = np.zeros(2**n)
    # start with one-bit pattern
    arr.append("0")
    arr.append("1")

    i = 2
    j = 0
    while True:
        if i >= 1 << n:
            break
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])
        for j in range(i):
            arr[j] = "0" + arr[j]
        for j in range(i, 2 * i):
            arr[j] = "1" + arr[j]
        i = i << 1
    for i in range(len(arr)):
        X[i] = int(arr[i], 2)
    return X


def PSKMod(msg_array, m, n):
    mod_seq = generateGrayarr(m)
    sym_seq1 = [np.math.cos(pi / 4 + ((2 * pi) / (2**m)) * i) for i in range(2**m)]
    sym_seq2 = [np.math.sin(pi / 4 + ((2 * pi) / (2**m)) * i) for i in range(2**m)]

    X = np.zeros((len(msg_array), n))
    for i in range(len(msg_array)):
        X[i, 0] = sym_seq1[np.where(mod_seq == msg_array[i])[0][0]]
        X[i, 1] = sym_seq2[np.where(mod_seq == msg_array[i])[0][0]]
    return X


def PSKDemod(RxSig, m, n):
    mod_seq = generateGrayarr(m)
    sym_seq1 = [np.math.cos(pi / 4 + ((2 * pi) / (2**m)) * i) for i in range(2**m)]
    sym_seq2 = [np.math.sin(pi / 4 + ((2 * pi) / (2**m)) * i) for i in range(2**m)]
    msg_det = np.zeros(len(RxSig))
    for i in range(len(RxSig)):
        dist = np.zeros(2**m)
        for j in range(2**m):
            dist[j] = (RxSig[i, 0] - sym_seq1[j]) ** 2 + (
                RxSig[i, 1] - sym_seq2[j]
            ) ** 2
        msg_det[i] = mod_seq[np.argmin(dist)]
    return msg_det


def PAMMod(msg_array, m, n):
    if m != 1 and m != 2 and m != 4:
        mod_seq = np.arange(2**m)
        l = round(2**m)
    else:
        mod_seq = generateGrayarr(int(m / n))
        l = 2 ** (int(m / n))
    mod_seq1 = np.zeros(len(mod_seq), dtype=int)
    for i in range(len(mod_seq1)):
        mod_seq1[i] = np.where(mod_seq == i)[0][0]
    odd_seq = np.linspace(l - 1, -l + 1, l)
    A = np.sqrt(3 / ((l) ** 2 - 1))
    L = len(msg_array)

    X2 = np.zeros((L, 1))
    X2[:, 0] = odd_seq[mod_seq1[msg_array]] * A
    return X2


def PAMMod_new(L, msg_array=None):
    M = []
    for i in range(L):
        M.append(-1 + (2 / (L - 1)) * i)

    # Variance computation
    var = 0
    for i in range(L):
        var += M[i] ** 2
    var = var / L

    M = np.array(M)
    M = M / np.sqrt(var)

    if msg_array is None:
        return M
    else:
        S = len(msg_array)
        X2 = np.zeros((S, 1))
        X2[:, 0] = M[msg_array]
        return X2


def PAMDemod(RxSig, m, n):
    if not (int(m / n) - m / n == 0):
        raise AttributeError("m should be greater than equal to n for baseline")

    mod_seq = generateGrayarr(int(m / n))
    l = int(m / n)
    odd_seq = np.linspace(2**l - 1, -(2**l) + 1, 2**l)
    A = np.sqrt(3 / ((2**l) ** 2 - 1)) / np.sqrt(n)
    odd_seq1 = odd_seq * A

    L = len(RxSig)
    msg_det = np.zeros(L)
    if n == 1:
        ODD_seq = np.matmul(np.ones((L, 1)), odd_seq1.reshape(1, len(odd_seq)))
        MSG_seq = np.matmul(RxSig.reshape(L, 1), np.ones((1, len(odd_seq1))))
        DIST_seq = np.abs(ODD_seq - MSG_seq)

        msg_det1 = mod_seq[np.argmin(DIST_seq, axis=1)]

        return msg_det1.astype(int)
    for i in range(L):
        num = 0
        for j in range(n):
            a = mod_seq[np.argmin(abs(odd_seq * A - RxSig[i, j]))]
            num = num + 2 ** (l * (n - j - 1)) * a
        msg_det[i] = num
    return msg_det.astype(int)


def convert_channel(args, A, theta):
    K, n, SNR = args.K, args.n, args.SNR

    H = np.zeros((K, K, n, n))
    T = np.zeros((K, K, n, n))

    assert n % 2 == 0

    n1 = int(n / 2)

    for i in range(K):
        for j in range(K):
            for i1 in range(n1):
                for j1 in range(n1):
                    T[i, j, i1 * 2 : (i1 + 1) * 2, j1 * 2 : (j1 + 1) * 2] = np.array(
                        [
                            [
                                math.cos(theta[i, j, i1, j1]),
                                math.sin(theta[i, j, i1, j1]),
                            ],
                            [
                                -math.sin(theta[i, j, i1, j1]),
                                math.cos(theta[i, j, i1, j1]),
                            ],
                        ]
                    )
    for i in range(K):
        for j in range(K):
            for i1 in range(n1):
                for j1 in range(n1):
                    alpha0 = A[i, j, i1, j1]
                    SIR = (SNR) * (1 - alpha0)
                    sir = 10 ** (SIR / 10)
                    H[i, j, i1 * 2 : (i1 + 1) * 2, j1 * 2 : (j1 + 1) * 2] = (
                        np.sqrt(1 / sir)
                        * T[i, j, i1 * 2 : (i1 + 1) * 2, j1 * 2 : (j1 + 1) * 2]
                    )
    return H


def get_channel(args):
    K, n = args.K, args.n

    assert n % 2 == 0
    n1 = int(n / 2)
    A = np.zeros((K, K, n1, n1))
    for i in range(K):
        for j in range(K):
            if i != j:
                A[i, j, :, :] = 0.9 * np.ones((n1, n1))
            else:
                A[i, j, :, :] = np.ones((n1, n1))

    rng = np.random.RandomState(args.seed)
    Theta = np.zeros((K, K, n1, n1))
    for i in range(K):
        for j in range(K):
            if i != j:
                Theta[i, j, :, :] = (2 * rng.rand(n1, n1) - 1) * (math.pi / 2)

    H = convert_channel(args, A, Theta)
    return H


def get_theta_channel(args):
    K, n = args.K, args.n

    assert n % 2 == 0
    n1 = int(n / 2)
    A = np.zeros((K, K, n1, n1))
    for i in range(K):
        for j in range(K):
            if i != j:
                A[i, j, :, :] = 0.9 * np.ones((n1, n1))
            else:
                A[i, j, :, :] = np.ones((n1, n1))

    rng = np.random.RandomState(args.seed)
    Theta = np.zeros((K, K, n1, n1))
    for i in range(K):
        for j in range(K):
            if i != j:
                Theta[i, j, :, :] = args.theta

    H = convert_channel(args, A, Theta)
    return H


def qfunc(x):
    return 0.5 - 0.5 * math.erf(x / np.sqrt(2))


def transmit_sequence(K, m):
    A = []
    for i in range(K - 1):
        a = np.arange(2**m)
        A.append(a)

    M = list(itertools.product(*A))
    M = np.array(M)

    X_des = []
    for k in range(K):
        X_des1 = []
        for i in range(2**m):
            M_new = np.insert(M, k, values=i * np.ones(2 ** (m * (K - 1))), axis=1)
            b_s = 2 ** (m * (K - 1))
            X = np.zeros((2**m, b_s, K))
            for u, v in product(range(b_s), range(K)):
                X[M_new[u, v], u, v] = 1
            X_des1.append(X)
        X_des.append(X_des1)
    return X_des
    # X_des = []
    # for i in range()

    # X_des = np.zeros(2**m,M.shape[0],K ):


# X_des = []
# for i in range(K):
#     X_des1 = []
#     for j in range(2**m):
# save_channels(5, "test_5_0")
# save_channels(5, "test_5_1")


def distance2D(X, a):
    return np.sqrt((X[:, 0] - a[0]) ** 2 + (X[:, 1] - a[1]) ** 2)
