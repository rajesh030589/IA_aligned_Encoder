import numpy as np
import tensorflow as tf
import pickle
import math
from MAXSINR_testing import (
    get_values as get_values_MX,
    testing as testing_MX,
    testing_theta as testing_MX_theta,
)
from get_default_args import get_args, get_filename
from NEURAL_MIND_PRE_testing import (
    get_constellation as get_constellation_NN,
    load_model as load_model_NN,
    get_values as get_values_NN,
    testing as testing_NN,
    testing_ber as testing_ber_NN1,
    testing_no_change as testing_NN_no_change,
    testing_with_constellation as testing_NN_with_constellation,
)

import matplotlib.pyplot as plt
from get_default_args import get_args
from tqdm import tqdm
from scipy.signal import savgol_filter

from matplotlib.lines import Line2D


def cdfplot(R):
    count, bins_count = np.histogram(R, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)

    return cdf, bins_count[1:]


def plot_rate():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 4, 2
    args.SNR = 12
    # # for m = 1
    # args.test_size = 10000

    # for m = 1
    args.test_size = 20
    RATE_MX = []
    RATE_NN = []
    for seed in tqdm(range(10)):
        args.seed = seed
        RATE_MX.append(testing_MX(args))
        RATE_NN.append(testing_NN(args))
    cdf_mx, bin_count_mx = cdfplot(RATE_MX)
    cdf_nn, bin_count_nn = cdfplot(RATE_NN)
    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    fig, axes = plt.subplots()
    axes.plot(bin_count_mx, cdf_mx, label="MX")
    axes.plot(bin_count_nn, cdf_nn, label="NN")
    axes.grid(True)
    plt.savefig(f"Figures/Rate_CDF_{args.SNR}_{args.K}_{args.m}_{args.n}.png")


def plot_constellation():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    args.SNR = 24
    args.seed = 0
    args.train_loss = "BCE"

    TxOut_MX, RXH_MX, _, RXSig_MX = get_values_MX(args)
    TxOut_NN, RXH_NN, _, RXSig_NN = get_values_NN(args)

    plt.rcParams.update({"figure.figsize": (12, 12), "figure.dpi": 100})
    fig, axes = plt.subplots(
        3,
        args.K,
        sharey=True,
    )
    for k in range(args.K):
        for k1 in range(args.K):
            axes[0][k].scatter(
                RXH_MX[k][k1][:, 0], RXH_MX[k][k1][:, 1], label="User " + str(k1)
            )
            axes[0][k].grid(True)
            axes[0][k].set_title("User " + str(k))
            axes[0][k].set_xlim([-1.5, 1.5])
    for k in range(args.K):
        for k1 in range(args.K):
            axes[1][k].scatter(
                RXH_NN[k][k1][:, 0], RXH_NN[k][k1][:, 1], label="User " + str(k1)
            )
            axes[1][k].grid(True)
            axes[1][k].set_title("User " + str(k))
            axes[1][k].set_xlim([-1.5, 1.5])
    for k in range(args.K):
        for k1 in range(args.K):
            axes[2][k].scatter(
                RXH_NN1[k][k1][:, 0], RXH_NN1[k][k1][:, 1], label="User " + str(k1)
            )
            axes[2][k].grid(True)
            axes[2][k].set_title("User " + str(k))
            axes[2][k].set_xlim([-1.5, 1.5])
    plt.savefig(f"Figures/RXH_BER_{args.SNR}_{args.K}_{args.m}_{args.n}.png")


def plot_training_loss():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    SNR_list = [28]
    for SNR in SNR_list:
        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        args.SNR = SNR
        for seed in range(1):
            args.seed = seed

            file_str = "NN_temp_scale"
            args.train_loss = "temp_scale"
            args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"

            # Load model
            _, Loss = load_model_NN(args)
            Loss = savgol_filter(Loss, 20, 3)  # window size 51, polynomial order 3

            axes.plot(Loss, label="MX")
            axes.grid(True)
        plt.savefig(
            f"Figures/Loss_{file_str}_{args.SNR}_{args.K}_{args.m}_{args.n}.png"
        )

    return Loss


def plot_NN_loss_constellation():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    args.SNR = 28
    args.seed = 0
    args.train_loss = "BCE"
    TxOut_NN, RXH_NN, _, RXSig_NN = get_values_NN(args)

    args.train_loss = "temp_scale"
    TxOut_NN1, RXH_NN1, _, RXSig_NN1 = get_values_NN(args)

    args.train_loss = "soft_decode"
    TxOut_NN2, RXH_NN2, _, RXSig_NN2 = get_values_NN(args)

    plt.rcParams.update({"figure.figsize": (12, 12), "figure.dpi": 100})
    fig, axes = plt.subplots(
        3,
        args.K,
        sharey=True,
    )
    for k in range(args.K):
        for k1 in range(args.K):
            axes[0][k].scatter(
                RXH_NN[k][k1][:, 0], RXH_NN[k][k1][:, 1], label="User " + str(k1)
            )
            axes[0][k].grid(True)
            axes[0][k].set_title("User " + str(k))
            axes[0][k].set_xlim([-1.5, 1.5])
    for k in range(args.K):
        for k1 in range(args.K):
            axes[1][k].scatter(
                RXH_NN1[k][k1][:, 0], RXH_NN1[k][k1][:, 1], label="User " + str(k1)
            )
            axes[1][k].grid(True)
            axes[1][k].set_title("User " + str(k))
            axes[1][k].set_xlim([-1.5, 1.5])
    for k in range(args.K):
        for k1 in range(args.K):
            axes[2][k].scatter(
                RXH_NN2[k][k1][:, 0], RXH_NN2[k][k1][:, 1], label="User " + str(k1)
            )
            axes[2][k].grid(True)
            axes[2][k].set_title("User " + str(k))
            axes[2][k].set_xlim([-1.5, 1.5])
    plt.savefig(f"Figures/Const_loss_{args.SNR}_{args.K}_{args.m}_{args.n}.png")


def get_loss_from_model(args):
    # Load model
    _, Loss = load_model_NN(args)
    Loss = savgol_filter(Loss, 20, 3)  # window size 51, polynomial order 3
    return Loss


def plot_training_loss_multiple():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    SNR_list = [4, 8, 12, 16, 20, 24]
    for SNR in SNR_list:
        NN_SD = []
        NN_BCE = []
        NN_TS = []
        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        args.SNR = SNR
        args.seed = 0

        file_str = "NN"
        args.train_loss = "soft_decode"
        args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"
        NN_SD = get_loss_from_model(args)

        file_str = "NN_BCE"
        args.train_loss = "BCE"
        args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"
        NN_BCE = get_loss_from_model(args)

        file_str = "NN_temp_scale"
        args.train_loss = "temp_scale"
        args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"
        NN_TS = get_loss_from_model(args)

        axes.plot(NN_SD, label="NN soft decode")
        axes.plot(NN_BCE, label="NN BCE")
        axes.plot(NN_TS, label="NN temp scale")

        axes.grid(True)
        plt.savefig(f"Figures/Loss_all_NN_{args.SNR}_{args.K}_{args.m}_{args.n}.png")


def plot_rate_SNR():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 2, 2
    args.seed = 0
    m_list = [2, 4, 8, 16]

    args.train_loss = "temp_scale"
    args.file_str = "NN_temp_scale"
    args.test_loss = "hard_decode"

    for m in m_list:
        RATE_MX = []
        RATE_NN = []

        args.m = m

        if m == 2:
            SNR_list = [4, 6, 8, 10, 12]
        elif m == 4:
            SNR_list = [8, 10, 12, 14, 16]
        elif m == 8:
            SNR_list = [12, 14, 16, 18, 20]
        elif m == 16:
            SNR_list = [14, 16, 18, 20, 22]
        for SNR in tqdm(SNR_list):
            args.SNR = SNR
            RATE_MX.append(testing_MX(args))
            RATE_NN.append(testing_NN(args))
        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        axes.plot(SNR_list, RATE_MX, label="MX")
        axes.plot(SNR_list, RATE_NN, label="NN")
        axes.grid(True)
        plt.savefig(
            f"Figures/Rate_SNR_{args.K}_{args.m}_{args.n}_{args.seed}_{args.file_str}.png"
        )


def plot_rate_all_SNR():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    SNR_List = [4, 8, 12, 16, 20, 24]
    m_list = [16]

    RATE_NN_soft_decode = []
    RATE_NN_hard_decode = []
    RATE_NN_BCE_soft_decode = []
    RATE_NN_BCE_hard_decode = []
    RATE_NN_TS_soft_decode = []
    RATE_NN_TS_hard_decode = []
    for m in m_list:
        args.m = m
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            args.file_str = "NN"
            args.train_loss = "soft_decode"
            args.test_loss = "hard_decode"
            RATE_NN_hard_decode.append(testing_NN(args))
            args.test_loss = "soft_decode"
            RATE_NN_soft_decode.append(testing_NN(args))
            args.train_loss = "BCE"
            args.file_str = "NN_BCE"
            args.test_loss = "hard_decode"
            RATE_NN_BCE_hard_decode.append(testing_NN(args))
            args.test_loss = "soft_decode"
            RATE_NN_BCE_soft_decode.append(testing_NN(args))
            args.train_loss = "temp_scale"
            args.file_str = "NN_temp_scale"
            args.test_loss = "hard_decode"
            RATE_NN_TS_hard_decode.append(testing_NN(args))
            args.test_loss = "soft_decode"
            RATE_NN_TS_soft_decode.append(testing_NN(args))

        plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 100})
        fig, axes = plt.subplots()
        axes.plot(
            SNR_List, RATE_NN_hard_decode, label="train: soft decode test: hard decode"
        )
        axes.plot(
            SNR_List, RATE_NN_soft_decode, label="train: soft decode test: soft decode"
        )
        axes.plot(
            SNR_List, RATE_NN_BCE_hard_decode, label="train: BCE test: hard decode"
        )
        axes.plot(
            SNR_List, RATE_NN_BCE_soft_decode, label="train: BCE test: soft decode"
        )
        axes.plot(
            SNR_List, RATE_NN_TS_hard_decode, label="train: TEMP test: hard decode"
        )
        axes.plot(
            SNR_List, RATE_NN_TS_soft_decode, label="train: TEMP test: soft decode"
        )
        axes.grid(True)
        plt.legend()
        plt.savefig(
            f"Figures/Rate1_SNR_all_NN_{args.K}_{args.m}_{args.n}_{args.seed}.png"
        )


def plot_rate_SNR_multiple_M():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    SNR_List = [4, 8, 12, 16, 20, 24]
    args.test_size = 20

    fig, axes = plt.subplots()
    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    RATE_NN = []
    RATE_TDMA = []
    for SNR in tqdm(SNR_List):
        args.SNR = SNR
        snr = 10 ** (SNR / 10)
        RATE_NN.append(testing_NN(args))
        RATE_TDMA.append(0.5 * np.log2(1 + args.K * snr))
    axes.plot(SNR_List, RATE_NN, label="NN")
    axes.plot(SNR_List, RATE_TDMA, label="TDMA")

    m_list = [3, 4, 5, 6, 8, 10]
    # for m = 1
    for m in m_list:
        args.m = m
        RATE_MX = []
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            RATE_MX.append(testing_MX(args))
        axes.plot(SNR_List, RATE_MX, label="MX " + str(m))
    axes.grid(True)
    axes.legend()
    plt.savefig(f"Figures/Rate_SNR_{args.K}_all_m_{args.n}_{args.seed}.png")


def plot_ber_SNR():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    SNR_List = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    # # for m = 1
    # args.test_size = 10000

    # for m = 1
    args.test_size = 20
    RATE_MX = []
    RATE_NN = []
    # RATE_NN1 = []
    for SNR in tqdm(SNR_List):
        args.SNR = SNR
        RATE_MX.append(testing_ber_MX(args))
        RATE_NN.append(testing_ber_NN(args))
        # RATE_NN1.append(testing_ber_NN1(args))
    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    fig, axes = plt.subplots()
    axes.semilogy(SNR_List, RATE_MX, label="MX")
    axes.semilogy(SNR_List, RATE_NN, label="NN BER")
    # axes.semilogy(SNR_List, RATE_NN1, label="NN RATE")
    axes.grid(True)
    plt.legend()
    plt.savefig(f"Figures/BER_SNR_{args.K}_{args.m}_{args.n}_{args.seed}.png")
    plt.close()


def plot_rate_SNR_multiple_M_constellation():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    fig, axes = plt.subplots()
    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    SNR_List = [4, 8, 12, 16, 20, 24]
    RATE_TDMA = []
    for SNR in tqdm(SNR_List):
        args.SNR = SNR
        snr = 10 ** (SNR / 10)
        RATE_TDMA.append(0.5 * np.log2(1 + args.K * snr))
    axes.plot(SNR_List, RATE_TDMA, label="TDMA")
    m_list = [4, 8, 16]
    for m in m_list:
        args.m = m
        RATE_NN = []
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            snr = 10 ** (SNR / 10)
            RATE_NN.append(testing_NN_constellation(args))
        axes.plot(SNR_List, RATE_NN, label="NN" + str(m))

    m_list = [3, 4, 5, 6, 8, 10, 12, 14, 16]
    # for m = 1
    for m in m_list:
        args.m = m
        RATE_MX = []
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            RATE_MX.append(testing_MX_constellation(args))
        axes.plot(SNR_List, RATE_MX, label="MX " + str(m))
    axes.grid(True)
    axes.legend()
    plt.savefig(f"Figures/Rate_SNR1_{args.K}_all_m_{args.n}_{args.seed}.png")
    plt.close()


def plot_ber_TP():
    args = get_args()

    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    fig, axes = plt.subplots()
    # plot main functions
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    SNR_List = [4, 8, 12, 16, 20, 24]

    for m in [4, 6, 8, 10, 12, 14, 16]:
        TP_MX = []
        for SNR in tqdm(SNR_List):
            args.m = m
            args.SNR = SNR
            args.test_size = int(80000 / (args.m**args.K))
            ser = testing_ber_MX(args)
            TP_MX.append((1 - ser) * 1000 * np.log2(args.m))
        axes.plot(SNR_List, TP_MX, label="MX " + str(m))

    TP_NN = []
    args.m = 16
    args.test_size = int(80000 / (args.m**args.K))
    for SNR in tqdm(SNR_List):
        args.SNR = SNR
        ser = testing_ber_NN(args)
        TP_NN.append((1 - ser) * 1000 * np.log2(args.m))
    axes.plot(SNR_List, TP_NN, label="NN")
    axes.grid(True)
    plt.legend()
    plt.savefig(f"Figures/TP_SNR_{args.K}_{args.m}_{args.n}_{args.seed}.png")
    plt.close()


def getComparisonCDF():
    args = get_args()
    args.K, args.n = 5, 2
    m_list = [4]
    args.train_loss = "temp_scale"
    args.file_str = "NN_temp_scale"
    args.test_loss = "hard_decode"
    args.linear_rx = False
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            RATE_NN = []
            RATE_MX = []
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE_MX.append(testing_MX(args))
                RATE_NN.append(testing_NN(args))
            RATE_MX, RATE_MX_centers = np.array(get_hist_plot_data(RATE_MX))
            RATE_NN, RATE_NN_centers = np.array(get_hist_plot_data(RATE_NN))
            # Plot the CDF
            plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
            fig, axes = plt.subplots()
            axes.plot(RATE_MX_centers, RATE_MX, label="MX", color="blue")
            axes.plot(RATE_NN_centers, RATE_NN, label="NN", color="red")

            axes.grid(True)
            plt.legend()
            plt.savefig(
                f"Figures/Final_rate_asymm_{args.K}_{args.m}_{args.n}_{args.SNR}.png"
            )
            plt.xlabel("Rate")
            plt.ylabel("CDF")
            plt.close()


def get_hist_plot_data(X, n_bins=100):
    X, Xedges = np.histogram(X, bins=n_bins)
    X = np.cumsum(X)
    X = X / X[-1]

    Xcenters = (Xedges[1:] + Xedges[:-1]) / 2
    return X, Xcenters


def plot_rate_SNR_dynamic_M():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    SNR_List = [4, 8, 12, 16, 20, 24]

    fig, axes = plt.subplots()
    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    RATE_NN = []
    RATE_TDMA = []
    RATE_MXD = []
    for SNR in tqdm(SNR_List):
        args.SNR = SNR
        snr = 10 ** (SNR / 10)
        RATE_NN.append(testing_NN(args))
        RATE_TDMA.append(0.5 * np.log2(1 + args.K * snr))
        RATE_MXD.append(testing_MX_dyn(args))
    axes.plot(SNR_List, RATE_NN, label="NN")
    axes.plot(SNR_List, RATE_TDMA, label="TDMA")
    axes.plot(SNR_List, RATE_MXD, label="Dynamic MX")

    m_list = [16]
    # for m = 1
    for m in m_list:
        args.m = m
        RATE_MX = []
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            RATE_MX.append(testing_MX(args))
        axes.plot(SNR_List, RATE_MX, label="MX " + str(m))
    axes.grid(True)
    axes.legend()
    plt.savefig(f"Figures/Rate_SNR_dynamic_{args.K}_all_m_{args.n}_{args.seed}.png")


def print_rate_SNR_MX_NN_dynamic():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 16, 2
    args.seed = 0
    SNR_List = [28, 32, 36]

    for SNR in tqdm(SNR_List):
        args.SNR = SNR
        snr = 10 ** (SNR / 10)
        print(testing_NN(args))
        print(testing_MX_dyn(args))
        print(testing_MX(args))


def plot_rate_SNR_theta():
    args = get_args()
    args.K, args.n = 3, 2
    args.seed = 0
    m_list = [4]
    theta_list = [math.pi / 2, math.pi / 3, math.pi / 4, math.pi / 6, math.pi / 12]
    theta_list_str = ["pi_2", "pi_3", "pi_4", "pi_6", "pi_12"]
    theta_tick = ["$\pi/2$", "$\pi/3$", "$\pi/4$", "$\pi/6$", "$\pi/12$"]
    theta_x_tick = np.arange(5)
    args.ch_type = "symmetric"
    args.train_loss = "temp_scale"
    args.test_loss = "hard_decode"
    args.linear_rx = False
    for m in m_list:
        args.m = m
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [16]
        elif m == 16:
            SNR_list = [22]
        for SNR in tqdm(SNR_list):
            args.SNR = SNR
            RATE_MX1 = []
            RATE_MX2 = []
            RATE_NN = []
            plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
            fig, axes = plt.subplots()
            for theta, theta_str in zip(theta_list, theta_list_str):
                args.theta = theta
                args.theta_str = theta_str
                args.file_str = "NN_" + args.theta_str + "_temp_scale"
                args.linear_rx = True
                RATE_MX1.append(testing_MX(args))
                args.linear_rx = False
                RATE_MX2.append(testing_MX(args))
                RATE_NN.append(testing_NN(args))
            axes.plot(RATE_MX1, label="MaxSINR")
            axes.plot(RATE_MX2, label="MaxSINR")
            axes.plot(RATE_NN, label="NN")
            axes.set_xlabel("Angle of Rotation")
            axes.set_ylabel("Rate")
            axes.set_xticks(theta_x_tick)
            axes.set_xticklabels(theta_tick)
            axes.grid(True)
            axes.legend()
            plt.show()
            # plt.savefig(
            #     f"Figures/Final_sym_theta_rate_{args.SNR}_{args.K}_{args.m}_{args.n}_{args.seed}_{args.train_loss}.png"
            # )


def plot_rate_SNR_fixed_theta():
    args = get_args()
    args.K, args.n = 3, 2
    args.seed = 0
    m_list = [2, 4, 8]
    args.ch_type = "symmetric"
    args.theta = math.pi / 12
    args.theta_str = "pi_12"
    args.train_loss = "temp_scale"
    args.test_loss = "hard_decode"
    args.linear_rx = False
    SNR_list = [4, 8, 12, 16, 20, 24]
    for m in m_list:
        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        args.m = m
        RATE_MX = []
        RATE_NN = []
        for SNR in tqdm(SNR_list):
            args.SNR = SNR
            args.file_str = "NN_" + args.theta_str + "_temp_scale"
            RATE_MX.append(testing_MX(args))
            RATE_NN.append(testing_NN(args))
        axes.plot(RATE_MX, label="MaxSINR")
        axes.plot(RATE_NN, label="NN")
        plt.savefig(
            f"Figures/Final_sym_fixed_theta_rate_{args.K}_{args.m}_{args.n}_{args.seed}_{args.train_loss}.png"
        )


def plot_rate_SNR_random_seed():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [2, 4]
    args.ch_type = "random"
    args.train_loss = "temp_scale"
    args.test_loss = "hard_decode"
    args.file_str = "NN_temp_scale"
    args.linear_rx = False
    SNR_list = [4, 8, 12, 16, 20, 24]
    for m in m_list:
        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        args.m = m
        RATE_MX = []
        RATE_NN = []
        for SNR in tqdm(SNR_list):
            args.SNR = SNR
            rate_mx = []
            rate_nn = []
            for seed in range(10):
                args.seed = seed
                rate_mx.append(testing_MX(args))
                rate_nn.append(testing_NN(args))
            RATE_MX.append(np.mean(rate_mx))
            RATE_NN.append(np.mean(rate_nn))
        axes.plot(RATE_MX, label="MaxSINR")
        axes.plot(RATE_NN, label="NN")
        plt.savefig(
            f"Figures/Final_asym_random_seed_rate_{args.K}_{args.m}_{args.n}_{args.seed}_{args.train_loss}.png"
        )


def plot_rate_SNR_MX_theta():
    args = get_args()
    args.K, args.n = 3, 2
    args.seed = 0
    m_list = [2, 4, 8]
    theta_list = [math.pi / 2, math.pi / 3, math.pi / 4, math.pi / 6, math.pi / 12]
    theta_list_str = ["pi_2", "pi_3", "pi_4", "pi_6", "pi_12"]
    args.ch_type = "symmetric"
    args.train_loss = "temp_scale"
    args.test_loss = "hard_decode"
    args.train_power = True
    args.linear_rx_list = [True, False]
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [16]
        elif m == 16:
            SNR_list = [22]
        for SNR in tqdm(SNR_list):
            plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
            fig, axes = plt.subplots()
            for linear_rx in args.linear_rx_list:
                args.linear_rx = linear_rx
                args.m = m
                args.SNR = SNR
                RATE_MX = []
                for theta, theta_str in zip(theta_list, theta_list_str):
                    args.theta = theta
                    args.theta_str = theta_str
                    RATE_MX.append(testing_MX(args))
                axes.plot(RATE_MX, label="MaxSINR")

            RATE_NN = []
            for theta, theta_str in zip(theta_list, theta_list_str):
                args.theta = theta
                args.theta_str = theta_str
                args.file_str = "NN_POW_" + args.theta_str + "_temp_scale"
                RATE_NN.append(testing_NN(args))
            axes.plot(RATE_NN, label="NN")
            axes.set_xlabel("Angle of Rotation")
            axes.set_ylabel("Rate")
            axes.grid(True)
            axes.legend()
            plt.savefig(
                f"Figures/MaxSINR_sym_theta_rate_{args.SNR}_{args.K}_{args.m}_{args.n}_{args.seed}_{args.train_loss}.png"
            )


def plot_temp_double_data():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    SNR_list = [24]
    for SNR in SNR_list:
        # common x data and two y data
        args.SNR = SNR
        args.seed = 0
        args.train_loss = "temp_scale"
        file_name = "TempData/" + get_filename(args, "DATA") + ".pkl"

        with open(file_name, "rb") as f:
            data = pickle.load(f)
            Rate = data["Rate"]
            Temp = data["Temp"]

        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        L = np.arange(len(Rate))
        axes.plot(L, Rate, label="Rate")
        # second y (plot with different scale)
        axes2 = axes.twinx()
        axes2.plot(L, Temp, label="Temp", color="red")
        axes.grid(True)
        plt.savefig(
            f"Figures/Analysis_NN_temp_scale_{args.SNR}_{args.K}_{args.m}_{args.n}.png"
        )


def plot_rate_SNR_loss_comparison():
    args = get_args()

    train_loss_list = ["BCE", "temp_scale", "soft_decode"]
    file_sr_list = ["NN_BCE", "NN_temp_scale", "NN"]
    train_loss_list_str = ["BCE", "temp", "soft"]
    # plot main functions
    args.K, args.m, args.n = 3, 4, 2
    args.seed = 0
    SNR_List = [8, 10, 12, 14, 16]
    m_list = [2, 4, 8, 16]

    for m in m_list:
        args.m = m
        if m == 2:
            SNR_List = [4, 6, 8, 10, 12]
        elif m == 4:
            SNR_List = [8, 10, 12, 14, 16]
        elif m == 8:
            SNR_List = [12, 14, 16, 18, 20]
        elif m == 16:
            SNR_List = [16, 18, 20, 22, 24]
        plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 100})
        fig, axes = plt.subplots()

        RATE_MX = []
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            RATE_MX.append(testing_MX(args))
        axes.plot(SNR_List, RATE_MX, label="MX")

        for train_loss in train_loss_list:
            args.file_str = file_sr_list[train_loss_list.index(train_loss)]
            RATE_NN = []
            for SNR in tqdm(SNR_List):
                args.SNR = SNR
                args.train_loss = train_loss
                RATE_NN.append(testing_NN(args))
            axes.plot(
                SNR_List,
                RATE_NN,
                label=train_loss_list_str[train_loss_list.index(train_loss)],
            )
            axes.grid(True)
            axes.legend()

        plt.savefig(
            f"Figures/Rate_SNR_comp_train_loss_{args.K}_{args.m}_{args.n}_{args.seed}.png"
        )


def plot_rate_SNR_loss_comparison():
    args = get_args()

    train_loss_list = ["BCE", "temp_scale", "soft_decode"]
    file_sr_list = ["NN_BCE", "NN_temp_scale", "NN"]
    train_loss_list_str = ["BCE", "temp", "soft"]
    # plot main functions
    args.K, args.m, args.n = 3, 4, 2
    args.seed = 0
    m_list = [16]

    for m in m_list:
        args.m = m
        if m == 2:
            SNR_List = [4, 6, 8, 10, 12]
        elif m == 4:
            SNR_List = [8, 10, 12, 14, 16]
        elif m == 8:
            SNR_List = [12, 14, 16, 18, 20]
        elif m == 16:
            SNR_List = [20, 24, 28, 32]
        plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 100})
        fig, axes = plt.subplots()

        RATE_MX = []
        for SNR in tqdm(SNR_List):
            args.SNR = SNR
            RATE_MX.append(testing_MX(args))
        axes.plot(SNR_List, RATE_MX, label="MX")

        for train_loss in train_loss_list:
            args.file_str = file_sr_list[train_loss_list.index(train_loss)]
            RATE_NN = []
            for SNR in tqdm(SNR_List):
                args.SNR = SNR
                args.train_loss = train_loss
                RATE_NN.append(testing_NN(args))
            axes.plot(
                SNR_List,
                RATE_NN,
                label=train_loss_list_str[train_loss_list.index(train_loss)],
            )
            axes.grid(True)
            axes.legend()

        plt.savefig(
            f"Figures/Rate_SNR_comp1_train_loss_{args.K}_{args.m}_{args.n}_{args.seed}.png"
        )


def plot_rate_SNR_loss_theta():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 3, 2, 2
    args.seed = 0
    args.ch_type = "symmetric"
    m_list = [2]
    train_loss_list = ["temp_scale"]
    theta_list = [math.pi / 2, math.pi / 3, math.pi / 4, math.pi / 6, math.pi / 12]
    theta_list_str = ["pi_2", "pi_3", "pi_4", "pi_6", "pi_12"]

    for m in m_list:
        args.m = m
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [8, 12, 16]
        elif m == 16:
            SNR_list = [20, 24, 28]

        for SNR in SNR_list:
            args.SNR = SNR
            RATE_MX = []
            plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
            fig, axes = plt.subplots()
            for theta, theta_str in zip(theta_list, theta_list_str):
                args.theta = theta
                args.theta_str = theta_str
                RATE_MX.append(testing_MX_theta(args))
            axes.plot(RATE_MX, label="MX")

            for train_loss in train_loss_list:
                args.train_loss = train_loss

                RATE_NN = []
                for theta, theta_str in zip(theta_list, theta_list_str):
                    args.theta = theta
                    args.theta_str = theta_str

                    RATE_NN.append(testing_NN(args))

                axes.plot(
                    RATE_NN,
                    label=train_loss_list_str[train_loss_list.index(train_loss)],
                )
            axes.grid(True)
            axes.legend()
            plt.savefig(
                f"Figures/Rate_loss_theta_{args.SNR}_{args.K}_{args.m}_{args.n}_{args.seed}.png"
            )


def plot_rate_SNR_random_loss_theta():
    args = get_args()

    # plot main functions
    args.K, args.m, args.n = 5, 4, 2
    args.seed = 0
    args.ch_type = "random"
    m_list = [4]
    args.train_loss = "temp_scale"
    args.linear_rx = False
    for m in m_list:
        args.m = m
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [4, 8, 12, 16, 20, 24]
        elif m == 16:
            SNR_list = [20, 24, 28]

        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        RATE_MX = []
        RATE_NN = []
        for SNR in SNR_list:
            args.SNR = SNR
            RATE_MX.append(testing_MX(args))
            RATE_NN.append(testing_NN(args))
        axes.plot(SNR_list, RATE_MX, label="MX")
        axes.plot(SNR_list, RATE_NN, label="NN")
        axes.grid(True)
        axes.legend()
        plt.savefig(
            f"Figures/Rate_loss_random_{args.K}_{args.m}_{args.n}_{args.seed}.png"
        )


def plot_loss_func_comparison():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    args.SNR = 20
    args.seed = 0
    args.ch_type = "random"

    train_loss_list = [
        "BCE_-1",
        "temp_scale_-1",
        "soft_decode_-1",
        "temp_scale_0.1",
        "temp_scale_0.5",
        "temp_scale_1.0",
        "temp_scale_5.0",
    ]

    plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
    fig, axes = plt.subplots()
    for train_loss in train_loss_list:
        args.train_loss = train_loss
        file_name = "TempData/" + get_filename(args, train_loss) + ".pkl"

        with open(file_name, "rb") as f:
            data = pickle.load(f)
            Rate = data["Rate"]
            if train_loss == "temp_scale_-1":
                print(data["Temp"])
        L = np.arange(len(Rate))
        Rate = np.array(Rate)
        Rate = np.sum(Rate, axis=1)
        axes.plot(L, Rate, label=train_loss)
        # second y (plot with different scale)
    axes.grid(True)
    plt.legend()
    plt.savefig(
        f"Figures/Analysis_NN_train_loss_comp_{args.SNR}_{args.K}_{args.m}_{args.n}.png"
    )


def plot_loss_comp_CDF_plot():
    args = get_args()
    args.K, args.n = 3, 2
    args.ch_type = "random"
    m_list = [4, 8, 16]
    train_loss_list = ["BCE", "soft_decode", "temp_scale"]
    file_sr_list = ["NN_BCE", "NN", "NN_temp_scale"]
    plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
    fig, axes = plt.subplots()
    for train_loss in tqdm(train_loss_list):
        Rate_val = []
        for m in m_list:
            args.m = m
            if m == 2:
                SNR_list = [4]
            elif m == 4:
                SNR_list = [10, 12]
            elif m == 8:
                SNR_list = [16, 18]
            elif m == 16:
                SNR_list = [20, 22]
            for SNR in SNR_list:
                args.SNR = SNR
                for seed in tqdm(range(10)):
                    args.train_loss = train_loss
                    args.file_str = file_sr_list[train_loss_list.index(train_loss)]
                    args.seed = seed
                    args.train_loss = train_loss
                    Rate_val.append(testing_NN(args))
        Rate_val, Rate_val_centers = get_hist_plot_data(np.array(Rate_val), n_bins=50)
        axes.plot(Rate_val_centers, Rate_val, label=train_loss)
    axes.set_xlabel("Rate")
    axes.set_ylabel("CDF")
    axes.grid(True)
    axes.legend()

    plt.savefig(f"Figures/Final_NN_train_loss_CDF_all_settings.png", dpi=500)


def plot_loss_comp_scatter_plot():
    args = get_args()
    args.K, args.n = 3, 2
    args.ch_type = "random"
    m_list = [4, 8, 16]
    train_loss_list_list = [
        [
            "temp_scale",
            "BCE",
        ],
        ["temp_scale", "soft_decode"],
    ]
    file_sr_list_list = [["NN_temp_scale", "NN_BCE"], ["NN_temp_scale", "NN"]]
    for train_loss_list in train_loss_list_list:
        train_loss_list_str = "_".join(train_loss_list)
        file_sr_list = file_sr_list_list[train_loss_list_list.index(train_loss_list)]
        plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
        fig, axes = plt.subplots()
        Rate_val = []
        for m in m_list:
            args.m = m
            if m == 2:
                SNR_list = [4]
            elif m == 4:
                SNR_list = [10, 12]
            elif m == 8:
                SNR_list = [16, 18]
            elif m == 16:
                SNR_list = [20, 22]
            for SNR in SNR_list:
                args.SNR = SNR
                for seed in range(10):
                    for train_loss in train_loss_list:
                        args.file_str = file_sr_list[train_loss_list.index(train_loss)]
                        args.seed = seed
                        args.train_loss = train_loss
                        Rate_val.append(testing_NN(args))
        axes.scatter(Rate_val[0::2], Rate_val[1::2], label=train_loss_list[0])
        axes.add_line(Line2D([0, 10], [0, 10], color="red"))
        axes.set_xlabel(train_loss_list[0])
        axes.set_ylabel(train_loss_list[1])
        axes.grid(True)
        axes.legend()

        plt.savefig(
            f"Figures/NN_train_loss_comp_all_settings_{train_loss_list_str}.png"
        )


def getComparisonCDF_MaxSINR():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [2, 4, 8]
    args.test_loss = "hard_decode"
    linear_rx_list = [True, False]
    args.train_loss = "temp_scale"
    args.train_power = True
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
            fig, axes = plt.subplots()
            for linear_rx in linear_rx_list:
                RATE = []
                for seed in tqdm(range(50)):
                    args.m = m
                    args.linear_rx = linear_rx
                    args.SNR = SNR
                    args.seed = seed
                    RATE.append(testing_MX(args))
                RATE_val, RATE_centers = np.array(get_hist_plot_data(RATE))
                axes.plot(
                    RATE_centers,
                    RATE_val,
                    label="Linear RX" if linear_rx else "Non-Linear RX",
                )
            RATE = []
            args.file_str = "NN_POW_temp_scale"
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_NN(args))
            RATE_val, RATE_centers = np.array(get_hist_plot_data(RATE))
            axes.plot(
                RATE_centers,
                RATE_val,
                label="Linear RX" if linear_rx else "Non-Linear RX",
            )

            axes.grid(True)
            plt.legend()
            plt.savefig(
                f"Figures/MaxSINR_NN_comp_{args.K}_{args.m}_{args.n}_{args.SNR}.png"
            )
            plt.xlabel("Rate")
            plt.ylabel("CDF")
            plt.close()


def plot_progressive_improv_NN():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [8]
    args.train_loss = "temp_scale"
    args.file_str = "NN_temp_scale"
    args.test_loss = "hard_decode"
    args.linear_rx = False
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            RATE_NN1 = []
            RATE_NN2 = []
            RATE_NN3 = []
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE_NN1.append(testing_NN(args))
                RATE_NN2.append(testing_NN_no_change(args))
                RATE_NN3.append(testing_NN_with_constellation(args))
            RATE_NN1_val, RATE_NN1_centers = np.array(get_hist_plot_data(RATE_NN1))
            RATE_NN2_val, RATE_NN2_centers = np.array(get_hist_plot_data(RATE_NN2))
            RATE_NN3_val, RATE_NN3_centers = np.array(get_hist_plot_data(RATE_NN3))
            plt.rcParams.update({"figure.figsize": (4, 4), "figure.dpi": 100})
            fig, axes = plt.subplots()
            axes.plot(
                RATE_NN1_centers, RATE_NN1_val, label="NN pretrained with MaxSINR"
            )
            axes.plot(RATE_NN2_centers, RATE_NN2_val, label="MaxSINR")
            axes.plot(RATE_NN3_centers, RATE_NN3_val, label="NN with Constellation")
            axes.grid(True)
            plt.legend()
            plt.savefig(
                f"Figures/NN_progressive_improv_{args.K}_{args.m}_{args.n}_{args.SNR}.png"
            )


def plot_specific_constellation():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [8]
    args.test_loss = "hard_decode"
    args.train_loss = "temp_scale"
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            args.m = m
            args.linear_rx = True
            args.SNR = SNR
            args.seed = 16
            Data = []

            TX, _ = load_model_NN(args, case="RAW")
            C, _ = get_constellation_NN(args, TX)

            plt.figure()
            for k in range(args.K):
                plt.scatter(C[k][:, 0], C[k][:, 1], label=f"User {k+1}")
            plt.grid(True)
            plt.legend()
            TX, _ = load_model_NN(args, case="CONST")
            C, _ = get_constellation_NN(args, TX)

            plt.figure()
            for k in range(args.K):
                plt.scatter(C[k][:, 0], C[k][:, 1], label=f"User {k+1}")
            plt.grid(True)
            plt.legend()
            args.pretrained_PAM = True
            TX, _ = load_model_NN(args, case="PRE-PAM")
            C, _ = get_constellation_NN(args, TX)
            

            plt.figure()
            for k in range(args.K):
                plt.scatter(C[k][:, 0], C[k][:, 1], label=f"User {k+1}")
            plt.grid(True)
            plt.legend()
            plt.show()

            args.pretrained_PAM = False
            args.train_constellation = True
            TX, _ = load_model_NN(args, case="PRE-NN")
            C, _ = get_constellation_NN(args, TX)

            plt.figure()
            for k in range(args.K):
                plt.scatter(C[k][:, 0], C[k][:, 1], label=f"User {k+1}")
            plt.grid(True)
            plt.legend()
            plt.show()


def plot_training_loss_pretrain():
    args = get_args()
    args.K, args.n = 3, 2

    args.m = 8
    args.SNR = 18
    plt.rcParams.update({"figure.figsize": (8, 8), "figure.dpi": 500})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    fig, axes = plt.subplots()
    Loss1 = []
    Loss2 = []
    for seed in range(50):
        args.seed = seed
        args.pretrained_model = False
        args.train_loss = "temp_scale"
        # Load model
        _, Loss = load_model_NN(args)
        Loss1.append(Loss[800])

        args.pretrained_model = True
        # Load model
        _, Loss = load_model_NN(args)
        Loss2.append(Loss[800])
    # Plot the CDF of loss1 and loss2
    Loss1, Loss1_centers = np.array(get_hist_plot_data(Loss1))
    Loss2, Loss2_centers = np.array(get_hist_plot_data(Loss2))
    axes.plot(Loss1_centers, Loss1, label="NN w/o MX Init", linewidth=2)
    axes.plot(Loss2_centers, Loss2, label="NN with MX Init", linewidth=2)
    # axes.set_xlabel("Sum rate (bps/Hz)")
    # axes.set_ylabel("CDF")
    axes.grid(True)
    axes.legend()
    plt.savefig(f"Figures/NN_pretrain_loss_{args.K}_{args.m}_{args.n}_{args.SNR}.png")
    plt.savefig(
        f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/NN_pretrain_loss_{args.K}_{args.m}_{args.n}_{args.SNR}.png",
        dpi=500,
    )
    plt.savefig(
        f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/NN_pretrain_loss_{args.K}_{args.m}_{args.n}_{args.SNR}.pdf",
    )


# plot_NN_loss_constellation()
# plot_rate_SNR_loss_theta()
# plot_rate_all_SNR()
# plot_rate_SNR_single()
# plot_rate_SNR()
# plot_training_loss()
# plot_training_loss_multiple()
# print_rate_SNR_MX_NN_dynamic()
# plot_ber_SNR()
# plot_ber_TP()

# getComparisonCDF()
# plot_rate_SNR_dynamic_M()
# plot_rate_SNR_dynamic_M()
plot_rate_SNR_theta()
# plot_rate_SNR_fixed_theta()
# plot_temp_double_data()
# plot_loss_func_comparison()
# plot_loss_comp_CDF_plot()
# plot_loss_comp_scatter_plot()
# getComparisonCDF_MaxSINR()
# plot_rate_SNR_MX_theta()
# plot_rate_SNR_random_seed()
# plot_progressive_improv_NN()
# plot_specific_constellation()
# plot_training_loss_pretrain()
# plot_rate_SNR_random_loss_theta()
