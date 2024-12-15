import math
from scipy import special
from itertools import product
from sys import _clear_type_cache
import numpy as np
import pickle
from comm1 import get_channel, get_theta_channel, PAMMod, PAMDemod, PAMMod_new
from get_default_args import get_args, get_filename
from tqdm import tqdm


def compute_rate_trans(tp, tar, m):
    tp = np.expand_dims(tp, axis=0)
    tp = np.repeat(tp, m, axis=0)
    tp = np.multiply(tp, tar)

    # Computes the probability matrix P(What|W)
    PWhat_g_W = np.divide(np.sum(tp, axis=2), np.sum(tar, axis=2))
    PWWhat = (1 / (m)) * PWhat_g_W
    PWhat = np.sum(PWWhat, axis=0) + 1e-20
    PWhat_temp = np.matmul(np.ones((m, 1), dtype=np.float64), np.reshape(PWhat, (1, m)))
    PW_g_What = np.divide(PWWhat, PWhat_temp)
    PW_g_What += 1e-40
    R = np.log2(m) + np.sum(np.multiply(PWWhat, np.log2(PW_g_What)))
    return R


def minimum_distance(K, m, input, Des, SNR):
    snr = 10 ** (SNR / 10)
    noise_std = 1 / np.sqrt(snr)

    X = list(product(range(m), repeat=K - 1))
    L = len(X)

    P = []
    P2 = 0
    for i in range(m):
        P1 = 0
        for l in range(L):
            R = (input - Des[i][l, :]) ** 2
            R = np.sum(R, axis=1)
            R = R / (2 * noise_std**2)
            P1 += (1 / (m ** (K))) * (1 / (2 * math.pi * noise_std**2)) * np.exp(-R)
        P2 += P1
        P.append(P1)
    P3 = []
    for i in range(m):
        P3.append(P[i] / P2)

    return P3


def compute_linear_rate(args, SignalIn, SignalOut, Target):
    K, m, SNR = args.K, args.m, args.SNR
    Rate = []
    MsgOut = []
    for k in range(K):
        X = SignalIn[k]
        M = PAMDemod(X, np.log2(m), 1).astype(int)

        Y = SignalOut[k]
        N = PAMDemod(Y, np.log2(m), 1).astype(int)

        P = np.zeros((m, N.shape[0]))
        P[N, np.arange(N.shape[0])] = 1
        T = Target[k]

        R = compute_rate_trans(P, T, m)
        Rate.append(R)
        MsgOut.append(N)
    return Rate, MsgOut


def compute_nl_rate(args, SignalOut, Desired, Target):
    K, m, SNR = args.K, args.m, args.SNR
    Rate = []
    MsgOut = []
    for k in range(K):
        Y = SignalOut[k]
        D = Desired[k]
        P = minimum_distance(K, m, Y, D, SNR)
        P = np.array(P)
        N = np.argmax(P, axis=0)
        if args.test_loss == "hard_decode":
            P = np.where(P == np.max(P, axis=0), 1, 0)
            # change more than one maxima to 0
            for i in range(P.shape[1]):
                if np.sum(P[:, i]) > 1:
                    found = 0
                    for j in range(P.shape[0]):
                        if P[j, i] == 1:
                            if found == 0:
                                found = 1
                            elif found == 1:
                                P[j, i] = 0

        T = Target[k]
        R = compute_rate_trans(P, T, m)
        Rate.append(R)
        MsgOut.append(N)
    return Rate, MsgOut


def get_desired(args, V, H):
    K, m, n = args.K, args.m, args.n

    X = list(product(range(m), repeat=K - 1))
    L = len(X)

    X1 = np.array(X)
    X1 = PAMMod(X1.flatten(), np.log2(m), 1)
    X1 = np.reshape(X1, (L, -1))

    Xs = PAMMod(np.arange(m), np.log2(m), 1)

    Desired = []
    for k in range(K):
        D1 = []
        for j in range((m)):
            D2 = np.zeros((L, n))
            for l in range(L):
                t = 0
                D3 = 0
                for k1 in range(K):
                    if k != k1:
                        D3 += H[k][k1] @ (X1[l, t] * V[:, k1 : k1 + 1])
                        t = t + 1
                    else:
                        D3 += H[k][k] @ (Xs[j] * V[:, k : k + 1])
                D2[l, :] = D3.T
            D1.append(D2)
        Desired.append(D1)

    return Desired


def get_values(args):
    args.test_size = 1
    args.mx_file = "Models/" + get_filename(args, "MX") + ".pkl"
    H = get_channel(args)

    Beam = pickle.load(open(args.mx_file, "rb"))
    V = Beam["V"]

    TxOut, HOut, _, _, RxIn, RxInN = maxSINR_comm(args, H, V)

    return TxOut, HOut, RxIn, RxInN


def prepare_data(args, V):
    snr = 10 ** (args.SNR / 10)
    K, m, b_s = args.K, args.m, args.test_size

    I = np.asarray(list(product(np.arange(m), repeat=K)))
    M = I
    for _ in range(b_s - 1):
        M = np.concatenate((M, I), axis=0)
    L = M.shape[0]
    # Transmitter
    Message = []
    SignalIn = []
    TxOut = []
    Target = []
    for k in range(K):
        X = PAMMod(M[:, k], np.log2(m), 1)
        # X1 = PAMMod_new(M[:, k], m)

        VX1 = X
        VX = np.matmul(VX1, np.transpose(V[:, k : k + 1]))
        Message.append(M[:, k])
        SignalIn.append(X)
        TxOut.append(VX)
        T = np.zeros((m, m, L))
        for i in range(L):
            T[M[i, k], :, i] = 1

        Target.append(T)

    # Channel
    noise_std = 1 / np.sqrt(snr)
    Noise = []
    for k in range(K):
        noise = noise_std * np.random.randn(TxOut[k].shape[0], TxOut[k].shape[1])
        Noise.append(noise)

    return SignalIn, Target, TxOut, Noise


def maxSINR_comm(args, H, V, U=None):
    K = args.K
    # Transmitter

    SignalIn, Target, TxOut, Noise = prepare_data(args, V)
    TxPower = []
    for k in range(K):
        TxPower.append(sum(np.var(TxOut[k], axis=0)))
    Des = get_desired(args, V, H)

    # Channel
    RxIn = []
    RxInN = []
    HOut = []
    SignalOut = []
    for k in range(K):
        Y = 0
        h = []
        for k1 in range(K):
            Y1 = np.matmul(TxOut[k1], H[k][k1].T)
            h.append(Y1)
            Y += Y1
        HOut.append(h)
        RxIn.append(Y)
        Y = Y + Noise[k]
        RxInN.append(Y)
        y = np.matmul(Y, U[:, k : k + 1])
        SignalOut.append(y)

    return (SignalIn, TxOut, HOut, Target, Des, RxIn, RxInN, SignalOut)


def testing(args):
    args.test_size = int(90000 / args.m**args.K)
    if args.ch_type == "random":
        H = get_channel(args)
        args.mx_file = "Models/" + get_filename(args, "MX") + ".pkl"
    elif args.ch_type == "symmetric":
        args.mx_file = (
            "Models/" + get_filename(args, "MX") + "_" + args.theta_str + ".pkl"
        )
        H = get_theta_channel(args)

    Beam = pickle.load(open(args.mx_file, "rb"))
    V = Beam["V"]
    U = Beam["U"]
    SignalIn, _, _, Target, Des, _, RxInN, RxInNF = maxSINR_comm(args, H, V, U)
    if args.linear_rx:
        Rate, _ = compute_linear_rate(args, SignalIn, RxInNF, Target)
    else:
        Rate, _ = compute_nl_rate(args, RxInN, Des, Target)

    if args.out_type == "user_rate":
        return np.round(Rate, 3)
    else:
        return np.sum(Rate)


def testing_theta(args):
    args.mx_file = "Models/" + get_filename(args, "MX") + "_" + args.theta_str + ".pkl"
    H = get_theta_channel(args)

    Beam = pickle.load(open(args.mx_file, "rb"))
    V = Beam["V"]
    args.test_size = int(90000 / args.m**args.K)
    _, _, Target, Des, _, RxInN = maxSINR_comm(args, H, V)
    args.test_size = int(90000 / (args.m**args.K))
    Rate, _ = compute_nl_rate(args, RxInN, Des, Target)
    return np.sum(Rate)


def main_function():
    args = get_args()
    args.K, args.m, args.n = 3, 7, 2
    SNR_list = [18]
    RATE = []
    args.linear_rx = True
    for SNR in SNR_list:
        for seed in range(1):
            args.SNR = SNR
            args.seed = 3
            Rate = testing(args)
            RATE.append(Rate)

    print(RATE)
    # plt.figure()
    # plt.plot(SNR_list, RATE)
    # plt.grid(True)
    # plt.savefig


def testing_single_case():
    args = get_args()
    args.K, args.m, args.n = 3, 8, 2
    args.SNR = 12
    args.linear_rx = True
    args.seed = 0
    Rate = testing(args)
    print(np.sum(Rate))


if __name__ == "__main__":
    # main_function()
    testing_single_case()
