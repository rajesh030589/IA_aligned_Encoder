import numpy as np
import tensorflow as tf
import pickle
from comm1 import get_channel
from NEURAL_MIND_PRE_MODEL import Tx_System, get_RX, get_desired_RX
import math
from MAXSINR_testing import (
    get_values as get_values_MX,
    testing as testing_MX,
    testing_theta as testing_MX_theta,
)
from Archive.MAXSINR_testing1 import (
    testing as testing_ber_MX,
)
from get_default_args import get_args, get_filename
from NEURAL_MIND_PRE_testing import (
    load_model as load_model_NN,
    get_values as get_values_NN,
    get_values_model as get_values_model_NN,
    testing as testing_NN,
    testing_ber as testing_ber_NN1,
    get_constellation as get_NN_constellation,
    testing_no_change as testing_NN_no_change,
    testing_with_constellation as testing_NN_with_constellation,
    testing_only_dir as testing_NN_only_dir,
)

import matplotlib.pyplot as plt
from get_default_args import get_args
from tqdm import tqdm
from scipy.signal import savgol_filter

from Archive.CONSTELLATION_testing import (
    testing_MX as testing_MX_constellation,
    testing_NN as testing_NN_constellation,
)
from Archive.MAXSINR_testing2 import testing as testing_MX_dyn


def plot_loss_comp_CDF_data():
    args = get_args()
    args.K, args.n = 3, 2
    args.ch_type = "random"
    m_list = [4, 8, 16]
    file_sr_list = ["NN_BCE", "NN", "NN_temp_scale"]
    train_loss_list = ["BCE", "soft_decode", "temp_scale"]
    Data = []
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

        # Create a dictionary
        data = {"Label": train_loss, "Y": Rate_val, "X": Rate_val_centers}
        Data.append(data)
    pickle.dump(Data, open("Data/loss_comp_CDF_data.pkl", "wb"))


def get_hist_plot_data(X, n_bins=100):
    X, Xedges = np.histogram(X, bins=n_bins)
    X = np.cumsum(X)
    X = X / X[-1]

    Xcenters = (Xedges[1:] + Xedges[:-1]) / 2
    return X, Xcenters

# Data for Figure 3
def data_rate_SNR_MX_theta():
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
            Data = []
            for linear_rx in args.linear_rx_list:
                args.linear_rx = linear_rx
                args.m = m
                args.SNR = SNR
                RATE_MX = []
                for theta, theta_str in zip(theta_list, theta_list_str):
                    args.theta = theta
                    args.theta_str = theta_str
                    RATE_MX.append(testing_MX(args))
                data = {
                    "Label": linear_rx,
                    "Y": RATE_MX,
                    "X": theta_list_str,
                }
                Data.append(data)

            RATE_NN = []
            for theta, theta_str in zip(theta_list, theta_list_str):
                args.theta = theta
                args.theta_str = theta_str
                args.file_str = "NN_POW_" + args.theta_str + "_temp_scale"
                RATE_NN.append(testing_NN(args))
            data = {
                "Label": "NN_POW",
                "Y": RATE_NN,
                "X": theta_list_str,
            }
            Data.append(data)
            pickle.dump(
                Data,
                open(f"Data/data_rate_SNR_MX_theta_m_{m}_SNR_{SNR}.pkl", "wb"),
            )


def data_rate_SNR_NN_theta():
    args = get_args()
    args.K, args.n = 3, 2
    args.seed = 0
    m_list = [2, 4, 8]
    theta_list = [math.pi / 2, math.pi / 3, math.pi / 4, math.pi / 6, math.pi / 12]
    theta_list_str = ["pi_2", "pi_3", "pi_4", "pi_6", "pi_12"]
    args.ch_type = "symmetric"
    args.train_loss = "temp_scale"
    args.test_loss = "hard_decode"
    args.train_power = False
    args.linear_rx = False
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
            Data = []
            args.m = m
            args.SNR = SNR
            RATE_MX = []
            for theta, theta_str in zip(theta_list, theta_list_str):
                args.theta = theta
                args.theta_str = theta_str
                RATE_MX.append(testing_MX(args))
            data = {
                "Label": "MX",
                "Y": RATE_MX,
                "X": theta_list_str,
            }
            Data.append(data)

            RATE_NN = []
            for theta, theta_str in zip(theta_list, theta_list_str):
                args.theta = theta
                args.theta_str = theta_str
                args.file_str = "NN_" + args.theta_str + "_temp_scale"
                RATE_NN.append(testing_NN(args))
            data = {
                "Label": "NN",
                "Y": RATE_NN,
                "X": theta_list_str,
            }
            Data.append(data)
            pickle.dump(
                Data,
                open(f"Data/data_rate_SNR_NN_theta_m_{m}_SNR_{SNR}.pkl", "wb"),
            )


def data_getComparisonCDF_MaxSINR():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [4, 8]
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
            Data = []
            for linear_rx in linear_rx_list:
                RATE = []
                for seed in tqdm(range(24)):
                    args.m = m
                    args.linear_rx = linear_rx
                    args.SNR = SNR
                    args.seed = seed
                    RATE.append(testing_MX(args))
                RATE_val, RATE_centers = get_hist_plot_data(np.array(RATE))
                data = {"Label": linear_rx, "Y": RATE_val, "X": RATE_centers}
                Data.append(data)

            RATE = []
            args.file_str = "NN_POW_temp_scale"
            for seed in tqdm(range(24)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_NN(args))
            RATE_val, RATE_centers = get_hist_plot_data(np.array(RATE))
            data = {"Label": "NN_POW", "Y": RATE_val, "X": RATE_centers}
            Data.append(data)
            pickle.dump(
                Data,
                open(
                    f"Data/data_asym_getComparisonCDF_MaxSINR_m_{m}_SNR_{SNR}.pkl", "wb"
                ),
            )


def data_getComparisonCDF_NN():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [16]
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
            Data = []
            RATE = []
            for seed in tqdm(range(50)):
                args.m = m
                args.linear_rx = True
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_MX(args))
            RATE_val, RATE_centers = get_hist_plot_data(np.array(RATE))
            data = {"Label": "MX-Linear", "Y": RATE_val, "X": RATE_centers}
            Data.append(data)
            RATE = []
            for seed in tqdm(range(50)):
                args.m = m
                args.linear_rx = False
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_MX(args))
            RATE_val, RATE_centers = get_hist_plot_data(np.array(RATE))
            data = {"Label": "MX-Non_Linear", "Y": RATE_val, "X": RATE_centers}
            Data.append(data)
            RATE = []
            args.file_str = "NN_POW_temp_scale"
            args.train_power = True
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_NN(args))
            RATE_val, RATE_centers = get_hist_plot_data(np.array(RATE))
            data = {"Label": "NN_POW", "Y": RATE_val, "X": RATE_centers}
            Data.append(data)

            RATE = []
            args.file_str = "NN_temp_scale"
            args.train_power = False
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_NN(args))
            RATE_val, RATE_centers = get_hist_plot_data(np.array(RATE))
            data = {"Label": "NN", "Y": RATE_val, "X": RATE_centers}
            Data.append(data)
            pickle.dump(
                Data,
                open(
                    f"Data/data_asym_getComparisonCDF_NN__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "wb",
                ),
            )


def get_NN_rate_asymm_data():
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
            Data = []
            RATE = []
            for seed in tqdm(range(50)):
                args.m = m
                args.linear_rx = True
                args.SNR = SNR
                args.seed = seed
                args.out_type = "user_rate"
                RATE.append(testing_MX(args))
            data = {"Label": "MX-Linear", "Rate": RATE}
            Data.append(data)
            # RATE = []
            # for seed in tqdm(range(50)):
            #     args.m = m
            #     args.linear_rx = False
            #     args.SNR = SNR
            #     args.seed = seed
            #     RATE.append(testing_MX(args))
            # data = {"Label": "MX-Non_Linear", "Rate": RATE}
            # Data.append(data)
            RATE = []
            args.file_str = "NN_temp_scale"
            args.train_power = False
            args.out_type = "user_rate"
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE.append(testing_NN(args))
            data = {"Label": "NN", "Rate": RATE}
            Data.append(data)
            pickle.dump(
                Data,
                open(
                    f"Data/data_asym1_data__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "wb",
                ),
            )


def genrate_data_rate_snr_fixed_theta():
    args = get_args()
    args.K, args.n = 3, 2
    args.seed = 0
    m_list = [2, 4]
    args.ch_type = "symmetric"
    args.theta = math.pi / 12
    args.theta_str = "pi_12"
    args.train_loss = "temp_scale"
    args.test_loss = "hard_decode"
    args.linear_rx = False
    SNR_list = [4, 8, 12, 16, 20, 24]
    Data = []
    for m in m_list:
        args.m = m
        RATE_MX = []
        RATE_NN = []
        for SNR in tqdm(SNR_list):
            args.SNR = SNR
            args.file_str = "NN_" + args.theta_str + "_temp_scale"
            RATE_MX.append(testing_MX(args))
            RATE_NN.append(testing_NN(args))
        data = {"Label": "MX" + str(m), "Y": RATE_MX, "X": SNR_list}
        Data.append(data)
        data = {"Label": "NN" + str(m), "Y": RATE_NN, "X": SNR_list}
        Data.append(data)
    pickle.dump(Data, open(f"Data/data_rate_snr_pi_12.pkl", "wb"))


def get_NN_constellation_data():
    args = get_args()
    args.K, args.n = 5, 2
    m_list = [4]
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
            for seed in tqdm(range(50)):
                args.seed = seed
                Data = []

                TX, _ = load_model_NN(args, case="RAW")
                Const, _ = get_NN_constellation(args, TX)
                data = {"Label": "MX", "Y": Const}
                Data.append(data)

                args.linear_rx = False
                TX, _ = load_model_NN(args, case="PRE-NN")
                Const, _ = get_NN_constellation(args, TX)
                data = {"Label": "PRE-NN", "Y": Const}
                Data.append(data)

                TX, _ = load_model_NN(args, case="CONST")
                Const, _ = get_NN_constellation(args, TX)
                data = {"Label": "NN-Const", "Y": Const}
                Data.append(data)

                pickle.dump(
                    Data,
                    open(
                        f"Data/data_NN_asym_constellation_K_{args.K}_m_{m}_SNR_{SNR}_Seed_{args.seed}.pkl",
                        "wb",
                    ),
                )


def get_NN_received_signal_data():
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
            for seed in tqdm(range(50)):
                args.seed = seed
                plot_rx_signal_specific(args)


def get_NN_received_signal_data_one_case():
    args = get_args()
    args.K, args.n = 3, 2
    args.m = 8
    args.test_loss = "hard_decode"
    args.train_loss = "temp_scale"
    args.SNR = 18
    args.seed = 16
    plot_rx_signal_specific(args)


def plot_rx_signal_specific(args):
    Data = []

    TX, _ = load_model_NN(args, case="RAW")
    args.linear_rx = True
    _, RxH, _, _ = get_values_model_NN(args, TX)
    data = {"Label": "MX", "Y": RxH}
    Data.append(data)
    args.linear_rx = False

    args.linear_rx = False
    TX, _ = load_model_NN(args, case="PRE-NN")
    _, RxH, _, _ = get_values_model_NN(args, TX)
    data = {"Label": "PRE-NN", "Y": RxH}
    Data.append(data)

    TX, _ = load_model_NN(args, case="CONST")
    _, RxH, _, _ = get_values_model_NN(args, TX)
    data = {"Label": "NN-Const", "Y": RxH}
    Data.append(data)

    TX, _ = load_model_NN(args, case="PRE-PAM")
    _, RxH, _, _ = get_values_model_NN(args, TX)
    data = {"Label": "NN-PAM", "Y": RxH}
    Data.append(data)

    pickle.dump(
        Data,
        open(
            f"Data/data_NN_asym_received_signal_K_{args.K}_m_{args.m}_SNR_{args.SNR}_Seed_{args.seed}.pkl",
            "wb",
        ),
    )


def get_NN_pretraining_save_data():
    args = get_args()
    args.K, args.n = 3, 2

    args.m = 8
    args.SNR = 18

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
    Final_loss = []
    Loss_val, Loss_centers = get_hist_plot_data(np.array(Loss1))
    Loss1 = {"Label": "w_o_pretrain", "Y": Loss_val, "X": Loss_centers}
    Loss_val, Loss_centers = get_hist_plot_data(np.array(Loss2))
    Loss2 = {"Label": "w_pretrain", "Y": Loss_val, "X": Loss_centers}
    Final_loss.append(Loss1)
    Final_loss.append(Loss2)
    pickle.dump(Final_loss, open("Data/NN_pretraining_loss.pkl", "wb"))


def get_improv_data():

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
            RATE_NN4 = []
            for seed in tqdm(range(50)):
                args.m = m
                args.SNR = SNR
                args.seed = seed
                RATE_NN1.append(testing_NN(args))
                RATE_NN2.append(testing_NN_no_change(args))
                RATE_NN3.append(testing_NN_with_constellation(args))
                RATE_NN4.append(testing_NN_only_dir(args))
            RATE_NN1_val, RATE_NN1_centers = np.array(get_hist_plot_data(RATE_NN1))
            RATE_NN2_val, RATE_NN2_centers = np.array(get_hist_plot_data(RATE_NN2))
            RATE_NN3_val, RATE_NN3_centers = np.array(get_hist_plot_data(RATE_NN3))
            RATE_NN4_val, RATE_NN4_centers = np.array(get_hist_plot_data(RATE_NN4))
            Data = []
            data = {"Label": "NN", "Y": RATE_NN1_val, "X": RATE_NN1_centers}
            Data.append(data)
            data = {"Label": "NN_no_change", "Y": RATE_NN2_val, "X": RATE_NN2_centers}
            Data.append(data)
            data = {
                "Label": "NN_with_constellation",
                "Y": RATE_NN3_val,
                "X": RATE_NN3_centers,
            }
            Data.append(data)
            data = {
                "Label": "NN_only_dir",
                "Y": RATE_NN4_val,
                "X": RATE_NN4_centers,
            }
            Data.append(data)
            pickle.dump(
                Data,
                open(
                    f"Data/data_improv_NN_asym__k_{args.K}_m_{m}_SNR_{SNR}.pkl",
                    "wb",
                ),
            )


# plot_loss_comp_CDF_data()
# data_rate_SNR_MX_theta()
# data_rate_SNR_NN_theta()
# data_getComparisonCDF_MaxSINR()
# data_getComparisonCDF_NN()
# genrate_data_rate_snr_fixed_theta()
# get_NN_constellation_data()
# get_NN_received_signal_data()
# get_NN_pretraining_save_data()
# get_NN_rate_asymm_data()
get_improv_data()
# get_NN_received_signal_data_one_case()
