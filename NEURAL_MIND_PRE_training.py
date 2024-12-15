import pickle
import numpy as np
import itertools
import tensorflow as tf
from comm1 import get_channel, get_theta_channel, transmit_sequence, PAMMod, PAMMod_new
from NEURAL_MIND_PRE_MODEL import get_RX, get_desired_RX, Tx_System
from get_default_args import get_args, get_filename
from NEURAL_MIND_PRE_testing import testing as testing_NN
from MAXSINR_testing import testing as testing_MX
from itertools import product
from tqdm import tqdm
import math
from MAXSINR_training import training as training_MX
from NEURAL_MIND_PRE_testing import test_model as test_model_NN, get_constellation

np.set_printoptions(threshold=np.inf)
tf.keras.backend.set_floatx("float64")

rng = np.random.RandomState(0)


# save the weights of the NN model
def save_weights(args, TX, Loss):
    Tx = TX.get_weights()
    NETWORK = {"TX": Tx, "Loss": Loss}
    pickle.dump(NETWORK, open(args.model_file, "wb"))
    if args.verbose:
        print("new model saved")


# compute the soft coded rate from transition probabilities
def compute_rate(tp, tar, m):
    tp = tf.expand_dims(tp, axis=0)
    tp = tf.repeat(tp, m, axis=0)
    tp = tf.math.multiply(tp, tar)

    # Computes the probability matrix P(What|W)
    PWhat_g_W = tf.math.divide(tf.reduce_sum(tp, axis=2), tf.reduce_sum(tar, axis=2))
    PWWhat = (1 / (m)) * PWhat_g_W
    PWhat = tf.reduce_sum(PWWhat, axis=0)
    PWhat_temp = tf.matmul(tf.ones((m, 1), dtype=tf.float64), tf.reshape(PWhat, (1, m)))
    PW_g_What = tf.math.divide(PWWhat, PWhat_temp)

    PW_g_What += 1e-40

    R = tf.convert_to_tensor(np.log2(m)) + tf.reduce_sum(
        tf.math.multiply(
            PWWhat,
            (tf.math.log(PW_g_What)) / tf.cast(tf.math.log(2.0), dtype=tf.float64),
        )
    )
    return R


# update the weights of the NN model
def update_weights(args, TX, X, XD, T, H, Noise):
    K, m, SNR = args.K, args.m, args.SNR
    with tf.GradientTape(persistent=True) as tape:
        V = TX.normalize_weights()
        TxSignal = TX.transmit(X, V)
        DesRX = get_desired_RX(args, TX, V, H, XD)
        _, _, RxSignal = get_RX(args, TxSignal, H, Noise)

        loss = 0
        trans_probs = TX.get_posterior_prob(RxSignal, DesRX, SNR)
        for k in range(K):
            tp = trans_probs[k]
            if args.train_loss == "BCE":
                tar = T[:, :, :, k]
                tar = (1 / m) * tf.reduce_sum(tar, axis=1)
                loss += tf.reduce_mean(tf.keras.losses.binary_crossentropy(tar, tp))
            elif args.train_loss == "soft_decode":
                tar = T[:, :, :, k]
                R = compute_rate(tp, tar, m)
                loss -= R
            elif args.train_loss == "temp_scale":
                tar = T[:, :, :, k]
                tp = tf.divide(tp, TX.temp_scale)
                tp = tf.math.exp(tp)
                tp = tf.math.divide(tp, tf.reduce_sum(tp, axis=0, keepdims=True))
                R = compute_rate(tp, tar, m)
                loss -= R

    Gradients = tape.gradient(loss, TX.trainable_variables)
    optimizer = tf.keras.optimizers.legacy.Adam(0.001)
    optimizer.apply_gradients((zip(Gradients, TX.trainable_variables)))
    return TX, loss


def prepare_data(args):
    K, m, n, b_s = args.K, args.m, args.n, args.train_size
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

    T = np.zeros((m, m, L, K))
    for j in range(K):
        T[M[np.arange(L), j], :, np.arange(L), j] = 1
    T = tf.convert_to_tensor(T, dtype=tf.float64)

    Noise = sigma * np.random.randn(K, L, n)
    XD = transmit_sequence(K, int(np.log2(m)))

    return X, XD, Noise, T


def run_model(args, TX, H):
    X, XD, Noise, T = prepare_data(args)
    TX, loss = update_weights(args, TX, X, XD, T, H, Noise)
    return TX, loss


def load_model(args, V, A):
    if args.pretrained_PAM:
        args.train_constellation = False
    TX = Tx_System(args, V, A)
    Network = pickle.load(open(args.model_file, "rb"))
    TX.set_weights(Network["TX"])
    Loss = Network["Loss"]
    return TX, Loss


def get_pretrained_PAM(TX, args):
    # _, PAM = get_constellation(args, TX)
    PAM = [args.m, 5, 4]
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
    return TX


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

    args.model_file = "Models/" + get_filename(args, file_str) + ".pkl"


def training(args):
    # Settings
    K, m = args.K, args.m

    # Load the model file name

    # Load the channel
    if args.ch_type == "random":
        H = get_channel(args)
    elif args.ch_type == "symmetric":
        H = get_theta_channel(args)

    # Load the pretrained beamformer
    if args.pretrained_model:
        if args.fixed_mx_snr:
            temp = args.SNR
            args.SNR = args.fixed_mx_snr_val
        if args.ch_type == "random":
            mx_filename = "Models/" + get_filename(args, "MX") + ".pkl"
        elif args.ch_type == "symmetric":
            mx_filename = (
                "Models/" + get_filename(args, "MX") + "_" + args.theta_str + ".pkl"
            )
        try:
            Beam = pickle.load(open(mx_filename, "rb"))
            V = Beam["V"].T
        except:
            args.MX_convergence = False
            training_MX(args)

            Beam = pickle.load(open(mx_filename, "rb"))
            V = Beam["V"].T
        A = PAMMod(np.arange(m), np.log2(m), 1)
        A = np.squeeze(np.repeat(A[np.newaxis, : int(m / 2), :], repeats=K, axis=0))
        A = np.reshape(A, (K, -1))
        if args.fixed_mx_snr:
            args.SNR = temp
    else:
        V, A = None, None

    # get the model name
    get_model_name(args)

    args.train_size = int(8000 / (args.m**args.K))

    # Load the older model
    if args.new_model:
        if args.train_power:
            args.train_constellation = False
            args.train_direction = False
        if args.pretrained_PAM:
            args.train_constellation = False
        TX = Tx_System(args, V, A)
        model_exists = False
        if args.pretrained_PAM:
            TX = get_pretrained_PAM(TX, args)
        Loss = []
    else:
        try:
            TX, Loss = load_model(args, V, A)
            model_exists = True
        except:
            TX = Tx_System(args, V, A)
            model_exists = False
            Loss = []

    Loss = []

    if args.train_test:
        Rate = []
        if args.train_loss == "temp_scale":
            Temp = []
    if args.train_new_model or not model_exists:
        for e in tqdm(range(args.n_epochs)):
            TX, loss = run_model(args, TX, H)
            Loss.append(-loss)
            if args.train_test:
                if (e + 1) % args.train_test_freq == 0:
                    args.test_size = int(40000 / args.m**args.K)
                    Rate.append(test_model_NN(args, TX, H))
                    print("Rate: ", sum(Rate[-1]))
                    if args.train_loss == "temp_scale":
                        Temp.append(TX.temp_scale.numpy())

            if loss == 0:
                break
        if args.train_test:
            if args.train_loss == "temp_scale":
                DATA = {"Rate": Rate, "Temp": Temp}
            else:
                DATA = {"Rate": Rate}

            # pickle.dump(
            #     DATA,
            #     open(
            #         "TempData/"
            #         + get_filename(args, args.train_loss + "_" + str(args.temp_value))
            #         + ".pkl",
            #         "wb",
            #     ),
            # )

        if args.save_model:
            save_weights(args, TX, Loss)
        return Loss
    else:
        print("Model already exists")
        return Loss


def main_training_symmetric():
    args = get_args()
    args.K, args.m, args.n = 3, 2, 2
    m_list = [2, 4, 8]
    train_loss_list = ["temp_scale"]
    theta_list = [
        # [math.pi / 2, "pi_2"],
        # [math.pi / 3, "pi_3"],
        # [math.pi / 4, "pi_4"],
        # [math.pi / 6, "pi_6"],
        [math.pi / 12, "pi_12"],
    ]
    args.n_epochs = 1800
    args.ch_type = "symmetric"
    args.fixed_mx_snr = False
    for m in m_list:
        args.m = m
        if m == 2:
            SNR_list = [4, 8, 12, 16, 20, 24]
            args.fixed_mx_snr_val = 4
        elif m == 4:
            SNR_list = [4, 8, 12, 16, 20, 24]
            args.fixed_mx_snr_val = 4
        elif m == 8:
            SNR_list = [4, 8, 12, 16, 20, 24]
            args.fixed_mx_snr_val = 12
        elif m == 16:
            SNR_list = [22]
            args.fixed_mx_snr_val = 16
        for train_loss in train_loss_list:
            for SNR in tqdm(SNR_list):
                for theta, theta_str in theta_list:
                    print(
                        "Training for SNR = ",
                        SNR,
                        " and m = ",
                        m,
                    )
                    args.new_model = False
                    args.save_model = True
                    args.seed = 0
                    args.SNR = SNR
                    args.theta = theta
                    args.theta_str = theta_str
                    args.train_loss = train_loss
                    args.train_test = False
                    args.train_new_model = False
                    args.train_power = False
                    training(args)


def main_training_random():
    args = get_args()
    args.K, args.n = 5, 2
    m_list = [4]
    train_loss_list = ["temp_scale"]
    args.n_epochs = 1200
    args.ch_type = "random"
    args.fixed_mx_snr = False
    args.fixed_mx_snr_val = 4

    for m in m_list:
        args.m = m
        if m == 2:
            SNR_list = [4, 8, 12, 16, 20, 24]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]

        for train_loss in train_loss_list:
            args.train_loss = train_loss
            for SNR in tqdm(SNR_list):
                args.SNR = SNR
                for seed in tqdm(range(10)):
                    print(
                        "Training for SNR = ",
                        SNR,
                        "seed = ",
                        seed,
                        " and train_loss = ",
                        train_loss,
                    )
                    args.seed = seed
                    args.train_power = True
                    args.new_model = False
                    args.train_new_model = False
                    args.save_model = True
                    args.train_test = False
                    training(args)


def main_training_random_no_pre():
    args = get_args()
    args.K, args.n = 3, 2
    m_list = [8]
    train_loss_list = ["temp_scale"]
    args.n_epochs = 1200
    args.ch_type = "random"
    args.pretrained_model = False

    for m in m_list:
        args.m = m
        if m == 2:
            SNR_list = [4]
        elif m == 4:
            SNR_list = [12]
        elif m == 8:
            SNR_list = [18]
        elif m == 16:
            SNR_list = [22]

        for train_loss in train_loss_list:
            args.train_loss = train_loss
            for SNR in tqdm(SNR_list):
                args.SNR = SNR
                for seed in tqdm(range(50)):
                    print(
                        "Training for SNR = ",
                        SNR,
                        "seed = ",
                        seed,
                        " and train_loss = ",
                        train_loss,
                    )
                    args.seed = seed
                    args.train_power = False
                    args.new_model = True
                    args.train_new_model = False
                    args.save_model = True
                    args.train_test = False
                    args.pretrained_PAM = False
                    training(args)


def main_training_single_case():
    args = get_args()
    args.K, args.m, args.n = 5, 4, 2
    args.n_epochs = 1200
    args.ch_type = "random"
    args.fixed_mx_snr = False
    args.fixed_mx_snr_val = 4
    args.seed = 0
    args.train_test_freq = 100
    args.SNR = 24
    args.train_loss = "temp_scale"
    args.new_model = True
    args.save_model = False
    args.train_test = True
    args.pretrained_model = False
    args.train_new_model = True
    args.pretrained_PAM = False
    training(args)


def main_training_loss_comp():
    args = get_args()
    args.K, args.m, args.n = 3, 16, 2
    train_loss_list = [
        "BCE",
        "temp_scale",
        "soft_decode",
        "temp_scale_0.1",
        "temp_scale_0.5",
        "temp_scale_1",
        "temp_scale_5",
    ]
    args.SNR = 20
    args.n_epochs = 1200
    args.ch_type = "random"

    args.fixed_mx_snr = True
    args.fixed_mx_snr_val = 4
    args.seed = 0
    args.train_test_freq = 100
    for train_loss in train_loss_list:
        if train_loss in [
            "temp_scale_0.1",
            "temp_scale_0.5",
            "temp_scale_1",
            "temp_scale_5",
        ]:
            args.temp_value = float(train_loss.split("_")[-1])
            args.train_loss = "temp_scale"
        else:
            args.train_loss = train_loss
            args.temp_value = -1
        args.new_model = True
        args.save_model = False
        args.train_test = True
        training(args)


def main_training_given_constellation():
    args = get_args()
    args.K, args.m, args.n = 3, 8, 2
    args.n_epochs = 1200
    args.ch_type = "random"
    args.fixed_mx_snr = False
    args.fixed_mx_snr_val = 4
    args.seed = 16
    args.train_test_freq = 100
    args.SNR = 18
    args.train_loss = "temp_scale"
    args.new_model = True
    args.save_model = True
    args.train_test = True
    args.pretrained_model = True
    args.train_new_model = True
    args.pretrained_PAM = True
    training(args)


def main_function():
    # print("Training random")
    # main_training_random()
    # main_training_random_no_pre()
    # print("Training symmetric")
    # main_training_symmetric()
    # main_training_loss_comp()
    # main_training_single_case()
    main_training_given_constellation()


if __name__ == "__main__":
    main_function()
