import tensorflow as tf
import numpy as np
import argparse
import math


def get_args():
    parser = argparse.ArgumentParser()

    # System parameters
    parser.add_argument("-K", type=int, default=3)
    parser.add_argument("-n", type=int, default=2)
    parser.add_argument("-m", type=int, default=1)

    # Model
    parser.add_argument("-new_model", type=bool, default=False)
    parser.add_argument("-save_model", type=bool, default=False)
    parser.add_argument("-verbose", type=bool, default=False)
    parser.add_argument("-pretrained_model", type=bool, default=True)

    parser.add_argument("-train_size", type=int, default=2)
    parser.add_argument("-test_size", type=int, default=10)
    parser.add_argument("-n_epochs", type=int, default=200)
    parser.add_argument("-test_loss", type=str, default="hard_decode")
    parser.add_argument("-train_loss", type=str, default="soft_decode")
    parser.add_argument("-out_type", type=str, default="sum_rate")
    parser.add_argument("-temp_value", type=float, default=-1)
    parser.add_argument("-train_test_freq", type=int, default=100)
    parser.add_argument("-MX_iterations", type=int, default=100)
    parser.add_argument("-train_power", type=bool, default=False)
    parser.add_argument("-fixed_mx_snr", type=bool, default=False)
    parser.add_argument("-pretrained_PAM", type=bool, default=False)
    parser.add_argument("-fixed_mx_snr_val", type=int, default=4)
    parser.add_argument("-train_direction", type=bool, default=True)
    parser.add_argument("-train_constellation", type=bool, default=True)

    # Channel
    parser.add_argument("-theta", type=float, default=0)
    parser.add_argument("-theta_str", type=str, default="pi_2")
    parser.add_argument("-ch_type", type=str, default="random")

    args, unknown = parser.parse_known_args()
    return args


def get_filename(args, init_string):
    filename = f"{init_string}__K_{str(args.K)}_SNR_{str(args.SNR)}_m_{str(args.m)}_n_{str(args.n)}_rnd_{args.seed}"
    return filename
