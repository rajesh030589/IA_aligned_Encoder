from cmath import pi
from numpy.core.fromnumeric import transpose
from scipy.io import loadmat
from scipy.linalg import null_space
import numpy as np
from tqdm import tqdm

import scipy.io as sio
import math


def random_beam(K, n, type="random"):
    # Initial random power allocation and beam vector selection

    V = []
    if type == "random":
        for _ in range(K):
            Vtemp = np.zeros((n, 1))
            v = np.random.rand(n)
            e = np.linalg.norm(v)
            Vtemp[:, 0] = v / e
            V.append(Vtemp)
    elif type == "uniform":
        theta = 2 * pi / (K)
        rot_mat = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        v = np.random.rand(n).reshape(n, 1)
        for _ in range(K):
            Vtemp = np.zeros((n, 1))
            v = np.matmul(rot_mat, v)
            e = np.linalg.norm(v)
            Vtemp[:, 0] = np.squeeze(v / e)
            V.append(Vtemp)

    elif type == "fixed_random":
        for k in range(K):
            Vtemp = np.zeros((n, 1))
            if k == 1:
                v = np.concatenate(([1], np.zeros(n - 1)))
            else:
                v = np.random.rand(n)
            e = np.linalg.norm(v)
            Vtemp[:, 0] = v / e
            V.append(Vtemp)
    return V


def compute_B_Matrix(K, n, P, H, V):
    B = []
    for rx_k in range(K):
        B1 = 0
        for tx_k in range(K):
            if rx_k != tx_k:
                A = H[rx_k][tx_k] @ V[tx_k]
                B1 += A @ (A.T)
        B1 += (1 / P) * np.eye(n)
        B.append(B1)
    return B


def compute_recieve_beamvector(K, B, H, V):
    U = []
    for rx_k in range(K):
        Ur = np.linalg.inv(B[rx_k]) @ H[rx_k][rx_k] @ V[rx_k]
        Ur = Ur / np.linalg.norm(Ur)
        U.append(Ur)
    return U
    """
    This function consolidates the received beamformers for all users
    """


def sum_rate(K, U, H, V, B):
    C = []
    for rx_k in range(K):
        A1 = (U[rx_k].T) @ H[rx_k][rx_k] @ V[rx_k]
        A2 = A1 @ (A1.T)
        A3 = (U[rx_k].T) @ B[rx_k] @ U[rx_k]
        SINR = A2 / A3
        C.append(np.log2(1 + SINR))
        return sum(C)


def Distributed_maxSINR(args, H):
    K, n = args.K, args.n
    P = 10 ** (args.SNR / 10)

    C_max = 0
    if args.MX_convergence:
        Cap = []
    for i in range(args.MX_iterations):
        err_th = 1e-6
        err = 1

        # SNR
        C_old = 0
        count = 0
        V1 = random_beam(K, n)
        while err > err_th and count < 100:
            count = count + 1

            # Compute the receive vectors

            B2 = compute_B_Matrix(K, n, P, H, V1)
            U2 = compute_recieve_beamvector(K, B2, H, V1)
            C = sum_rate(K, U2, H, V1, B2)

            err = abs(C - C_old)
            C_old = C

            # Reciprocal Channel
            Hr = np.transpose(H, [1, 0, 3, 2])

            # Compute the receive vectors
            B1 = compute_B_Matrix(K, n, P, Hr, U2)
            V1 = compute_recieve_beamvector(K, B1, Hr, U2)

            # Compute the beam vectors for the reciprocal channel
            if args.MX_convergence:
                Cap.append(np.squeeze(C))
        if C > C_max:
            C_max = C
            V_max = V1.copy()
            U_max = U2.copy()
    if args.MX_convergence:
        import matplotlib.pyplot as plt

        plt.plot(Cap)
        plt.savefig(f"Figures/Converence_MX_{args.K}_{args.n}_{args.SNR}.png")
    return V_max, U_max, C_max
