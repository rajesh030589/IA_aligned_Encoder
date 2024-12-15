import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import os
from get_default_args import get_args
import pickle as pkl
from math import pi
from tqdm import tqdm
import pandas as pd


def generate_loss_CDF_plot_from_data():
    args = get_args()
    train_loss_list_size = 3
    Data = pkl.load(open("Data/loss_comp_CDF_data.pkl", "rb"))

    plt.figure()
    plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 12
    for i in range(train_loss_list_size):
        data = Data[i]
        label = data["Label"]
        X = data["X"]
        Y = data["Y"]

        if label == "BCE":
            label = "BCE loss"
        elif label == "soft_decode":
            label = "Soft decode loss"
        elif label == "temp_scale":
            label = "Soft decode with temperature scaling"
        plt.plot(X, Y, label=label, linewidth=2)
    plt.xlabel("Rate (Bits/channenl use)")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.xlim([2, 9])
    plt.tight_layout()
    plt.savefig("Final_Figures/loss_comp_CDF_plot.png", dpi=500)

# Generate plot for rate vs SNR for different theta values
# Figure 3 in the paper
def generate_MX_sym_plot_from_data():
    args = get_args()
    train_loss_list_size = 3

    m_list = [2, 4, 8]
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [16]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            Data = pkl.load(
                open(f"Data/data_rate_SNR_MX_theta_m_{m}_SNR_{SNR}.pkl", "rb")
            )

            plt.figure()
            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 12
            for i in range(3):
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == True:
                    label = "MaxSINR"
                elif label == False:
                    label = "MaxSINR with ML decoder"
                elif label == "NN_POW":
                    label = "MX-MLD with power optimization"

                plt.plot(X, Y, label=label, linewidth=2)

            plt.ylabel("Rate (Bits/channenl use)")
            plt.xlabel("Channel rotation angle (radians)")
            plt.xticks(
                [0, 1, 2, 3, 4],
                [
                    r"$\frac{\pi}{2}$",
                    r"$\frac{\pi}{3}$",
                    r"$\frac{\pi}{4}$",
                    r"$\frac{\pi}{6}$",
                    r"$\frac{\pi}{12}$",
                ],
            )
            plt.grid(True)
            plt.legend()
            plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"Final_Figures/Rate_SNR_MX_theta_m_{m}_SNR_{SNR}.png", dpi=500)


def generate_MX_NN_sym_plot_from_data():
    args = get_args()
    train_loss_list_size = 3

    m_list = [4, 8]
    plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 16
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [16]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            Data_MX = pkl.load(
                open(f"Data/data_rate_SNR_MX_theta_m_{m}_SNR_{SNR}.pkl", "rb")
            )
            Data_NN = pkl.load(
                open(f"Data/data_rate_SNR_NN_theta_m_{m}_SNR_{SNR}.pkl", "rb")
            )
            markerlist = ["o", "s", "D"]
            count = 0
            plt.figure()
            for i in range(3):
                data = Data_MX[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == True:
                    label = "MaxSINR"
                elif label == False:
                    label = "MaxSINR with ML decoder"
                    continue
                elif label == "NN_POW":
                    label = "DISC-MaxSINR"

                plt.plot(
                    X,
                    Y,
                    label=label,
                    linewidth=2,
                    marker=markerlist[count],
                    markersize=10,
                )
                count += 1
            for i in range(2):
                data = Data_NN[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == "MX":
                    label = "MaxSINR with ML decoder"
                    pass
                elif label == "NN":
                    label = "DISC-MaxSINR+"
                    plt.plot(
                        X,
                        Y,
                        label=label,
                        linewidth=2,
                        marker=markerlist[count],
                        markersize=10,
                    )

            # plt.ylabel("Rate (Bits/channenl use)")
            # plt.xlabel("Channel rotation angle (radians)")
            plt.xticks(
                [0, 1, 2, 3, 4],
                [
                    r"$\frac{\pi}{2}$",
                    r"$\frac{\pi}{3}$",
                    r"$\frac{\pi}{4}$",
                    r"$\frac{\pi}{6}$",
                    r"$\frac{\pi}{12}$",
                ],
            )
            plt.grid(True)
            plt.legend()
            # plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                f"Final_Figures/Rate_SNR_MX_NN_all_theta_m_{m}_SNR_{SNR}.png", dpi=500
            )
            plt.savefig(
                f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/Rate_SNR_MX_NN_all_theta_m_{m}_SNR_{SNR}.png",
                dpi=500,
            )


def generate_NN_sym_plot_from_data():
    args = get_args()
    train_loss_list_size = 3

    m_list = [2, 4, 8]
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [16]
        elif m == 16:
            SNR_list = [22]
        for SNR in SNR_list:
            Data = pkl.load(
                open(f"Data/data_rate_SNR_NN_theta_m_{m}_SNR_{SNR}.pkl", "rb")
            )

            plt.figure()
            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 16
            for i in range(2):
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == "MX":
                    label = "MaxSINR with ML decoder"
                elif label == "NN":
                    label = "NN pretrained with MaxSINR"

                plt.plot(X, Y, label=label, linewidth=2)

            plt.ylabel("Rate (Bits/channenl use)")
            plt.xlabel("Channel rotation angle (radians)")
            plt.xticks(
                [0, 1, 2, 3, 4],
                [
                    r"$\frac{\pi}{2}$",
                    r"$\frac{\pi}{3}$",
                    r"$\frac{\pi}{4}$",
                    r"$\frac{\pi}{6}$",
                    r"$\frac{\pi}{12}$",
                ],
            )
            plt.grid(True)
            plt.legend()
            # plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"Final_Figures/Rate_SNR_NN_theta_m_{m}_SNR_{SNR}.png", dpi=500)
            plt.savefig(
                f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/Rate_SNR_MX_NN_asym_K_{args.K}_m_{m}_SNR_{SNR}.png",
                dpi=500,
            )


def generate_MX_asym_plot_from_data():
    args = get_args()
    train_loss_list_size = 3

    m_list = [4, 8]
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
            Data = pkl.load(
                open(
                    f"Data/data_asym_getComparisonCDF_MaxSINR_m_{m}_SNR_{SNR}.pkl", "rb"
                )
            )

            plt.figure()
            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 12
            for i in range(3):
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == True:
                    label = "MaxSINR"
                elif label == False:
                    label = "MaxSINR with ML decoder"
                elif label == "NN_POW":
                    label = "MX-MLD with power optimization"

                plt.plot(X, Y, label=label, linewidth=2)

            plt.xlabel("Rate (Bits/channenl use)")
            plt.ylabel("CDF")
            plt.grid(True)
            plt.legend()
            plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                f"Final_Figures/CDF_asym_SNR_MX_theta_m_{m}_SNR_{SNR}.png", dpi=500
            )


def generate_MX_NN_asym_plot_from_data_3user():
    args = get_args()
    train_loss_list_size = 3
    args.K = 3
    m_list = [4]
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
            Data = pkl.load(
                open(
                    f"Data/data_asym_getComparisonCDF_NN__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "rb",
                )
            )
            plt.figure()
            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 16
            for i in range(4):
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == "MX-Linear":
                    label = "MaxSINR"
                elif label == "MX-Non_Linear":
                    label = "MaxSINR with ML decoder"
                    continue
                elif label == "NN_POW":
                    label = "DISC-MaxSINR"
                elif label == "NN":
                    label = "DISC-MaxSINR+"

                plt.plot(X, Y, label=label, linewidth=2)

            # plt.xlabel("Rate (Bits/channenl use)")
            # plt.ylabel("CDF")
            plt.grid(True)
            plt.legend()
            # plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                f"Final_Figures/CDF_asym_SNR_MX_NN_all_k_{args.K}_m_{m}_SNR_{SNR}.png",
                dpi=500,
            )

            plt.savefig(
                f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/Rate_SNR_MX_NN_asym_K_{args.K}_m_{m}_SNR_{SNR}.png",
                dpi=500,
            )
def generate_MX_NN_asym_plot_from_data_5user():
    args = get_args()
    train_loss_list_size = 3
    args.K = 5
    m_list = [4]
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
            Data = pkl.load(
                open(
                    f"Data/data_asym_getComparisonCDF_NN__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "rb",
                )
            )
            plt.figure()
            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 16
            for i in range(4):
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == "MX-Linear":
                    label = "MaxSINR"
                elif label == "MX-Non_Linear":
                    label = "MaxSINR with ML decoder"
                    continue
                elif label == "NN_POW":
                    label = "DISC-MaxSINR"
                elif label == "NN":
                    label = "DISC-MaxSINR+"

                plt.plot(X, Y, label=label, linewidth=2)

            # plt.xlabel("Rate (Bits/channenl use)")
            # plt.ylabel("CDF")
            plt.grid(True)
            plt.legend()
            # plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                f"Final_Figures/CDF_asym_SNR_MX_NN_all_k_{args.K}_m_{m}_SNR_{SNR}.png",
                dpi=500,
            )

            plt.savefig(
                f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/Rate_SNR_MX_NN_asym_K_{args.K}_m_{m}_SNR_{SNR}.png",
                dpi=500,
            )


def generate_NN_asym_plot_from_data():
    args = get_args()
    args.K = 5
    m_list = [4]
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
            Data = pkl.load(
                open(
                    f"Data/data_asym_getComparisonCDF_NN__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "rb",
                )
            )

            plt.figure()
            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 12
            for i in range(2):
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == "MX":
                    label = "MaxSINR with ML decoder"
                elif label == "NN":
                    label = "NN pretrained with MaxSINR"

                plt.plot(X, Y, label=label, linewidth=2)

            plt.xlabel("Rate (Bits/channenl use)")
            plt.ylabel("CDF")
            plt.grid(True)
            plt.legend()
            plt.title(f"{m}-PAM, SNR = {SNR} dB", fontweight="bold")
            plt.tight_layout()
            plt.savefig(
                f"Final_Figures/CDF_asym_SNR_NN_theta_m_{m}_SNR_{SNR}.png", dpi=500
            )


def generate_fixed_theta_rate_SNR():
    Data = pkl.load(open(f"Data/data_rate_snr_pi_12.pkl", "rb"))

    plt.figure()
    plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 12
    for i in range(4):
        data = Data[i]
        label = data["Label"]
        X = data["X"]
        Y = data["Y"]
        if label == "MX2":
            label = "MX-MLD, 2-PAM"
        elif label == "NN2":
            label = "Pretrained NN, 2-PAM"
        elif label == "MX4":
            label = "MX-MLD, 4-PAM"
        elif label == "NN4":
            label = "Pretrained NN, 4-PAM"

        plt.plot(X, Y, label=label, linewidth=2)
    plt.ylabel("Rate (Bits/channenl use)")
    plt.xlabel("SNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Final_Figures/fixed_theta_rate_SNR.png", dpi=500)


def plot_constellation():
    args = get_args()
    args.K = 3
    m_list = [16]
    for m in m_list:
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [16]
        elif m == 16:
            SNR_list = [22]

        for SNR in SNR_list:
            args.K, args.m, args.SNR = 3, m, SNR
            args.seed = 16
            Data = pkl.load(
                open(
                    f"Data/data_NN_asym_constellation_K_{args.K}_m_{m}_SNR_{SNR}_seed_{args.seed}.pkl",
                    "rb",
                )
            )

            plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.size"] = 16
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            for i in range(3):
                data = Data[i]
                label = data["Label"]
                C = data["Y"]
                if label == "MX":
                    title = "MaxSINR"
                elif label == "PRE-NN":
                    title = "DISC-MAXSINR+"
                elif label == "CONST":
                    title = "DISC-MAXSINR+ (Uniform constellation)"
                plt.figure()
                for k in range(args.K):
                    plt.scatter(C[k][:, 0], C[k][:, 1], label=f"User {k+1}")
                plt.grid(True)
                plt.legend()
                plt.xlim([-2.5, 2.5])
                plt.ylim([-2.5, 2.5])
                plt.tight_layout()
                # plt.title(f"{title}, SNR = {SNR} dB", fontweight="bold")
                plt.savefig(
                    f"Final_Figures/constellation_{label}_K_{args.K}_m_{m}_SNR_{SNR}_seed_{args.seed}.png",
                    dpi=500,
                )

                plt.savefig(
                    f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/constellation_{label}_K_{args.K}_m_{m}_SNR_{SNR}_seed_{args.seed}.png",
                    dpi=500,
                )


def plot_rx_signal():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [8]
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
            args.SNR = SNR
            for seed in tqdm(range(50)):
                args.seed = seed
                Data = pkl.load(
                    open(
                        f"Data/data_NN_asym_received_signal_K_{args.K}_m_{m}_SNR_{SNR}_seed_{args.seed}.pkl",
                        "rb",
                    )
                )
                # plot subplots with 3 figures

                plt.rcParams.update({"figure.figsize": (18, 12), "figure.dpi": 500})
                plt.rcParams["font.family"] = "serif"
                plt.rcParams["mathtext.fontset"] = "dejavuserif"
                plt.rcParams["font.size"] = 16
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
                fig, axs = plt.subplots(2, 3)
                # fig.suptitle(f"Received signal, SNR = {SNR} dB", fontweight="bold")
                for i in range(2):
                    data = Data[i]
                    label = data["Label"]
                    R = data["Y"]
                    if label == "MX":
                        title = "MaxSINR"
                    elif label == "PRE-NN":
                        title = "DISC-MAXSINR+"
                    for j in range(args.K):
                        for k in range(args.K):
                            axs[i, j].scatter(
                                R[j][k][:, 0], R[j][k][:, 1], label=f"User {k+1}"
                            )
                            axs[i, j].set_xlim([-2.5, 2.5])  # Set x limits
                            axs[i, j].set_ylim([-2.5, 2.5])  # Set y limits
                        axs[i, j].grid(True)

                # Create some example data for the table
                data = np.random.rand(5, 3)

                # # Create a table
                # table_data = [["", "Column 1", "Column 2", "Column 3"]] + [
                #     ["Row " + str(i)] + list(data[i]) for i in range(5)
                # ]
                # table = plt.table(
                #     cellText=table_data,
                #     loc="center",
                #     cellLoc="center",
                #     bbox=[-1.4, -1.2, 1, 1],
                # )

                # # Adjust layout to make room for the table
                # plt.subplots_adjust(bottom=0.4)
                plt.savefig(
                    f"Final_Figures/Received_signal_K_{args.K}_m_{m}_SNR_{SNR}.png",
                    dpi=500,
                )

                plt.savefig(
                    f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/Received_signal_K_{args.K}_m_{m}_SNR_{SNR}_args.seed_{seed}.png",
                    dpi=500,
                )


def plot_rx_signal_one_case():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [8]
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
            args.SNR = SNR
            args.seed = 16
            Data = pkl.load(
                open(
                    f"Data/data_NN_asym_received_signal_K_{args.K}_m_{m}_SNR_{SNR}_seed_{args.seed}.pkl",
                    "rb",
                )
            )
            # plot subplots with 3 figures

            plt.rcParams.update({"figure.figsize": (18, 12), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            plt.rcParams["font.size"] = 16
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            fig, axs = plt.subplots(2, 3)
            # fig.suptitle(f"Received signal, SNR = {SNR} dB", fontweight="bold")
            markerlist = ["o", "s", "*"]
            markerSize = [100, 100, 200]
            for it, i in enumerate([1, 3]):
                data = Data[i]
                label = data["Label"]
                R = data["Y"]
                if label == "MX":
                    title = "MaxSINR"
                elif label == "PRE-NN":
                    title = "DISC-MAXSINR+"
                elif label == "PRE-PAM":
                    title = "DISC-MAXSINR+ (Uniform constellation)"
                for j in range(args.K):
                    for k in range(args.K):
                        axs[it, j].scatter(
                            R[j][k][:, 0],
                            R[j][k][:, 1],
                            label=f"User {k+1}",
                            marker=markerlist[k],
                            s=markerSize[k],
                        )
                        axs[it, j].set_xlim([-1.75, 1.75])  # Set x limits
                        axs[it, j].set_ylim([-1.75, 1.75])  # Set y limits
                    axs[it, j].grid(True)
            # plt.show()
            plt.savefig(
                f"Final_Figures/Received_signal_K_{args.K}_m_{m}_SNR_{SNR}_1.png",
                dpi=500,
            )

def plot_pretraining_data():
    Data = pkl.load(open("Data/NN_pretraining_loss.pkl", "rb"))
    plt.figure()
    # plt.rcParams.update({"figure.figsize": (8, 8), "figure.dpi": 500})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.rcParams["font.size"] = 12

    for i in range(2):
        data = Data[i]
        label = data["Label"]
        X = data["X"]
        Y = data["Y"]
        if label == "w_o_pretrain":
            label = "DISC-MaxSINR+ w/o pretraining"
        elif label == "w_pretrain":
            label = "DISC-MaxSINR+ with pretraining"
        plt.plot(X, Y, label=label, linewidth=2)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Final_Figures/NN_pretraining_loss.png", dpi=500)
    plt.savefig(
        "/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/NN_pretraining_loss.png",
        dpi=500,
    )


def create_table_from_data():

    Data = pkl.load(open("Data/data_asym1_data__k_3_m_8_SNR_18.pkl", "rb"))
    df = pd.DataFrame()
    df["Serial"] = np.linspace(0, 49, 50)
    for i in range(2):
        data = Data[i]
        label = data["Label"]
        X = data["Rate"]
        df[label] = X
        df[label + "_sum"] = np.round(np.sum(np.array(X), axis=1), 4)
    df["Rate_diff"] = np.round(
        df[Data[1]["Label"] + "_sum"] - df[Data[0]["Label"] + "_sum"], 4
    )
    df.to_csv(
        "/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/628de1cbac93d653b5a3e0b6/Figures/NN_rates.csv",
        index=False,
    )


def plot_improv_data():
    args = get_args()
    args.K = 3
    m_list = [8]
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
            Data = pkl.load(
                open(
                    f"Data/data_improv_NN_asym__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "rb",
                )
            )
            plt.figure()
            # plt.rcParams.update({"figure.figsize": (6, 6), "figure.dpi": 500})
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["mathtext.fontset"] = "dejavuserif"
            # plt.rcParams["font.size"] = 12
            for i in [1, 3, 2, 0]:
                data = Data[i]
                label = data["Label"]
                X = data["X"]
                Y = data["Y"]
                if label == "NN_no_change":
                    label = "Disc-MaxSINR"
                elif label == "NN_with_constellation":
                    label = "Disc-MaxSINR+(U)"
                elif label == "NN":
                    label = "Disc-MaxSINR+"
                elif label == "NN_only_dir":
                    label = "Disc-MaxSINR+(D)"

                plt.plot(X, Y, label=label, linewidth=2)

        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"Final_Figures/improv_NN_asym_K_{args.K}_m_{m}_SNR_{SNR}.png", dpi=500
        )
        plt.savefig(
            f"/Users/rajeshmishra/Library/CloudStorage/OneDrive-Personal/Documents/Research/Interference/Overleaf paper/661a883677f3afa46ceb6c1d/Figures/Improv_NN_asym_K_{args.K}_m_{m}_SNR_{SNR}.png",
            dpi=500,
        )




# generate_MX_NN_sym_plot_from_data()
# generate_MX_NN_asym_plot_from_data_3user()
# generate_MX_NN_asym_plot_from_data_5user()
# plot_rx_signal()
# plot_rx_signal_one_case()
# plot_pretraining_data()
# create_table_from_data()
# plot_improv_data()
