import pickle
import numpy as np
import itertools
import tensorflow as tf
from comm1 import (
    get_channel,
    get_theta_channel,
    transmit_sequence,
    PAMMod,
    PAMMod_new,
    distance2D,
)
from NEURAL_MIND_PRE_MODEL import Tx_System, get_RX, get_desired_RX
from MAXSINR_testing import compute_nl_rate

from get_default_args import get_args, get_filename
from itertools import product
import matplotlib.pyplot as plt


def load_model(args, case=None):
    # Load the model that is learned
    if case is None or case == "PRE-NN":
        file_str = get_model_name(args)
        args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"
        TX = Tx_System(args)
        Network = pickle.load(open(args.nn_file, "rb"))
        TX.set_weights(Network["TX"])
        Loss = Network["Loss"]

    # Load the raw model pretrained from maxsinr
    elif case == "RAW":
        if args.ch_type == "random":
            mx_filename = "Models/" + get_filename(args, "MX") + ".pkl"
        elif args.ch_type == "symmetric":
            mx_filename = (
                "Models/" + get_filename(args, "MX") + "_" + args.theta_str + ".pkl"
            )
        Beam = pickle.load(open(mx_filename, "rb"))
        V = Beam["V"].T

        A = PAMMod(np.arange(args.m), np.log2(args.m), 1)
        A = np.squeeze(
            np.repeat(A[np.newaxis, : int(args.m / 2), :], repeats=args.K, axis=0)
        )
        A = np.reshape(A, (args.K, -1))

        # Load the model
        TX = Tx_System(args, V, A)
        Loss = None

    # Load the model with pre determined constellation
    elif case == "CONST":
        TX, Loss = load_model(args, case="PRE-NN")
        _, PAM = get_constellation(args, TX)
        A = []
        for k in range(args.K):
            m1 = PAM[k]
            M = PAMMod_new(m1)
            A.append(M[: int(m1 / 2) + 1])

        # Load the model
        for k in range(args.K):
            M = A[k]
            cnt = 0
            for j in range(TX.M.shape[1]):
                TX.M[k, j].assign(-M[cnt])
                cnt += 1
                if cnt == M.shape[0]:
                    cnt = 0

    elif case == "DIR":
        TX, Loss = load_model(args, case="PRE-NN")
        PAM = [args.m, args.m, args.m]
        A = []
        for k in range(args.K):
            m1 = PAM[k]
            M = PAMMod_new(m1)
            A.append(M[: int(m1 / 2) + 1])

        # Load the model
        for k in range(args.K):
            M = A[k]
            cnt = 0
            for j in range(TX.M.shape[1]):
                TX.M[k, j].assign(-M[cnt])
                cnt += 1
                if cnt == M.shape[0]:
                    cnt = 0

    elif case == "PRE-PAM":
        args.pretrained_PAM = True
        args.train_constellation = False
        file_str = get_model_name(args)
        args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"
        TX = Tx_System(args)
        Network = pickle.load(open(args.nn_file, "rb"))
        TX.set_weights(Network["TX"])
        Loss = Network["Loss"]
        args.pretrained_PAM = False
        args.train_constellation = True
    return TX, Loss


def prepare_data(args):
    K, m, n, b_s = args.K, args.m, args.n, args.test_size
    sigma = 10 ** (-args.SNR / 20)

    I = np.asarray(list(product(np.arange(m), repeat=K)))
    M = I
    for _ in range(b_s - 1):
        M = np.concatenate((M, I), axis=0)
    L = M.shape[0]
    X = np.zeros((m, L, K))
    for i, j in itertools.product(range(L), range(K)):
        X[M[i, j], i, j] = 1
    X = tf.convert_to_tensor(X, dtype=tf.float64)

    T = []
    for j in range(K):
        V = np.zeros((m, m, L))
        V[M[np.arange(L), j], :, np.arange(L)] = 1
        T.append(V)
    T = tf.convert_to_tensor(T, dtype=tf.float64)

    Noise = sigma * np.random.randn(K, L, n)
    XD = transmit_sequence(K, int(np.log2(m)))

    return X, XD, Noise, T


def get_constellation(args, TX):
    # get the constellation from NN
    m, K = args.m, args.K

    C, _, _, _ = get_values_model(args, TX)
    C = np.array(C)
    Const = []
    PAM = []
    for k in range(args.K):
        # Find the unique values in the set
        const = C[k, : m ** (K - k) : m ** (K - 1 - k)]
        count = 1
        new_const = [const[0, :]]
        for i in range(1, const.shape[0]):
            new_const_arr = np.array(new_const)
            act_const = const[i, :]
            diff = min(distance2D(new_const_arr, act_const))
            if diff > 0.05:
                count += 1
                new_const.append(const[i, :])
        Const.append(np.array(new_const))
        # count = m
        PAM.append(count)
    return Const, PAM


def test_model(args, model, channel):
    TX = model
    H = channel

    # Load the dataset
    X, XD, Noise, Target = prepare_data(args)
    V = TX.normalize_weights()
    TxSignal = TX.transmit(X, V)
    DesRX = get_desired_RX(args, TX, V, H, XD)
    _, _, RxSignal = get_RX(args, TxSignal, H, Noise)
    R, _ = compute_nl_rate(args, RxSignal, DesRX, Target)
    return R


def testing(args):
    # Load the channel
    if args.ch_type == "symmetric":
        H = get_theta_channel(args)
    else:
        H = get_channel(args)

    args.test_size = int(90000 / args.m**args.K)

    # Load the model
    TX, Loss = load_model(args)

    R = test_model(args, TX, H)

    if args.out_type == "user_rate":
        return np.round(R, 3)
    else:
        return np.sum(R)


def testing_no_change(args):
    K, m = args.K, args.m
    # Load the channel
    if args.ch_type == "symmetric":
        H = get_theta_channel(args)
    else:
        H = get_channel(args)

    args.test_size = int(90000 / args.m**args.K)

    TX, _ = load_model(args, case="RAW")

    R = test_model(args, TX, H)

    return np.sum(R)


def testing_only_dir(args):
    K, m = args.K, args.m
    # Load the channel
    if args.ch_type == "symmetric":
        H = get_theta_channel(args)
    else:
        H = get_channel(args)

    args.test_size = int(90000 / args.m**args.K)

    TX, _ = load_model(args, case="DIR")

    R = test_model(args, TX, H)

    return np.sum(R)


def testing_with_constellation(args):
    K, m = args.K, args.m
    # Load the channel
    if args.ch_type == "symmetric":
        H = get_theta_channel(args)
    else:
        H = get_channel(args)

    args.test_size = int(90000 / args.m**args.K)
    TX, _ = load_model(args, case="CONST")
    R = test_model(args, TX, H)

    return np.sum(R)


def testing_ber(args):
    H = get_channel(args)

    args.nn_file = "Models/" + get_filename(args, "NN") + ".pkl"
    TX, Loss = load_model(args)

    X, XD, Noise, Target = prepare_data(args)
    V = TX.normalize_weights()

    TxSignal = TX.transmit(X, V)

    DesRX = get_desired_RX(args, TX, V, H, XD)
    _, _, RxSignal = get_RX(args, TxSignal, H, Noise)

    BER, _ = compute_nl_rate_ber(args, RxSignal, DesRX, Target)

    return np.mean(BER)


def get_model_name(args):
    if args.pretrained_model:
        file_str = "NN"
        if args.train_power:
            file_str += "_POW"
        if args.pretrained_PAM:
            file_str += "_PAM_1"
    else:
        file_str = "NN_NOPRE"

    if args.ch_type == "symmetric":
        file_str += "_" + args.theta_str
    if args.train_loss == "BCE":
        file_str += "_" + "BCE"
    if args.train_loss == "temp_scale":
        file_str += "_" + "temp_scale"

    return file_str


def get_values(args):
    file_str = get_model_name(args)
    args.nn_file = "Models/" + get_filename(args, file_str) + ".pkl"

    # Load model
    TX, Loss = load_model(args, case="PRE-NN")

    TxSignal, RxH, RxSigH, RxSignal = get_values_model(args, TX)
    return TxSignal, RxH, RxSigH, RxSignal


def get_values_model(args, TX):
    args.test_size = 1

    # Get channel
    if args.ch_type == "symmetric":
        H = get_theta_channel(args)
    else:
        H = get_channel(args)

    X, _, Noise, _ = prepare_data(args)
    V = TX.normalize_weights()
    TxSignal = TX.transmit(X, V)

    RxH, RxSigH, RxSignal = get_RX(args, TxSignal, H, Noise)

    return TxSignal, RxH, RxSigH, RxSignal


def main_function():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    SNR_list = [4]
    RATE = []
    for SNR in SNR_list:
        for seed in range(1):
            args.SNR = SNR
            args.seed = seed
            print(SNR)
            Rate = testing(args)
            RATE.append(np.sum(Rate))

    print(RATE)


def main_function_single_case():
    args = get_args()
    args.K, args.m, args.n = 3, 8, 2
    args.SNR = 18
    args.seed = 16
    args.ch_type = "random"
    args.train_loss = "temp_scale"
    args.file_str = "NN_temp_scale"
    args.test_loss = "hard_decode"
    args.train_power = False
    args.out_type = "user_rate"
    args.pretrained_PAM = True
    args.train_constellation = False
    Rate = testing(args)
    print(Rate)
    # Rate = testing_with_constellation(args)
    # print(Rate)
    # Rate = testing_no_change(args)
    # print(Rate)


if __name__ == "__main__":
    # main_function()
    main_function_single_case()

    # if args.pretrained_model:
    #     file_str = "NN"
    # else:
    #     file_str = "NN_NOPRE"

    # if args.ch_type == "symmetric":
    #     file_str += "_" + args.theta_str
