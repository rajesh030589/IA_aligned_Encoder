import numpy as np
import pickle
from comm1 import get_channel, get_theta_channel
from tqdm import tqdm
from MAXSINR_MODEL import Distributed_maxSINR as distn
from math import pi
from get_default_args import get_args, get_filename
import math


def save_beamvectors(args, V, U):
    Beam = {"V": V, "U": U}
    pickle.dump(Beam, open(args.model_file, "wb"))


def generate_beamvectors(args, H):
    Vp, Up, _ = distn(args, H)
    return Vp, Up


def load_model(args, H):
    Vp, Up = generate_beamvectors(args, H)
    V = np.transpose(np.array(Vp))[0, :, :]
    U = np.transpose(np.array(Up))[0, :, :]

    save_beamvectors(args, V, U)

    return V, U


def training(args):
    if args.ch_type == "random":
        H = get_channel(args)
        args.model_file = "Models/" + get_filename(args, "MX") + ".pkl"
    elif args.ch_type == "symmetric":
        args.model_file = (
            "Models/" + get_filename(args, "MX") + "_" + args.theta_str + ".pkl"
        )
        H = get_theta_channel(args)
    load_model(args, H)


def main_function_symmetric():
    args = get_args()
    args.K, args.m, args.n = 3, 2, 2
    SNR_list = [8, 12, 16]
    args.MX_convergence = False
    args.ch_type = "symmetric"
    for m in [8, 16]:
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
            print(SNR)
            for theta, theta_str in [
                [math.pi / 2, "pi_2"],
                [math.pi / 3, "pi_3"],
                [math.pi / 4, "pi_4"],
                [math.pi / 6, "pi_6"],
                [math.pi / 12, "pi_12"],
            ]:
                args.seed = 0
                args.SNR = SNR
                args.m = m
                args.theta = theta
                args.theta_str = theta_str
                training(args)


def main_function_random():
    args = get_args()
    args.K, args.m, args.n = 5, 4, 2
    args.MX_convergence = False
    args.ch_type = "random"
    args.MX_iterations = 100
    m_list = [4]
    for m in m_list:
        if m == 2:
            SNR_list = [4, 8, 12, 16, 20, 24]
        elif m == 4:
            SNR_list = [4, 8, 12, 16, 20, 24]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]
        for SNR in tqdm(SNR_list):
            for seed in range(10):
                args.seed = seed
                args.SNR = SNR
                args.m = m
                training(args)


def main_function_single_case():
    args = get_args()
    args.K, args.m, args.n = 3, 7, 2
    args.MX_convergence = False
    args.ch_type = "random"
    args.MX_iterations = 100
    args.SNR = 14
    args.seed = 0
    training(args)


if __name__ == "__main__":
    main_function_random()
    # main_function_symmetric()
    # main_function_single_case()
