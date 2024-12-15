import pickle
import numpy as np
import itertools
import tensorflow as tf
from comm1 import get_channel, transmit_sequence


np.set_printoptions(threshold=np.inf)
tf.keras.backend.set_floatx("float64")

rng = np.random.RandomState(0)


class Tx_System(tf.keras.Model):
    def __init__(self, args, V=None, X=None):
        super(Tx_System, self).__init__()

        self.K, self.m, self.n = args.K, args.m, args.n
        self.sigma = 10 ** (-args.SNR / 20)
        self.msq = np.sqrt(int(self.m / 2))
        if V is None:
            V = np.random.randn(self.K, self.n)

        if X is None:
            if args.m % 2 == 0:
                X = np.random.randn(self.K, int(self.m / 2))
            else:
                X = np.random.randn(self.K, int(self.m / 2) + 1)
                X[:, -1] = 0
        Pow = np.ones((self.K, 1))

        self.W = tf.Variable(
            initial_value=V,
            dtype=tf.float64,
            trainable=args.train_direction,
        )
        self.P = tf.Variable(
            initial_value=Pow,
            constraint=lambda x: tf.clip_by_value(x, 0.005, 1),
            trainable=True,
        )
        self.M = tf.Variable(
            initial_value=X,
            dtype=tf.float64,
            trainable=args.train_constellation,
        )
        if args.train_loss == "temp_scale":
            if args.temp_value == -1:
                self.temp_scale = tf.Variable(
                    initial_value=1.0,
                    constraint=lambda x: tf.clip_by_value(x, 0.005, 1),
                    dtype=tf.float64,
                    trainable=True,
                )
            else:
                self.temp_scale = tf.Variable(
                    initial_value=args.temp_value,
                    constraint=lambda x: tf.clip_by_value(x, 0.005, 1),
                    dtype=tf.float64,
                    trainable=False,
                )

    def normalize_weights(self):
        V_2 = []
        for k in range(self.K):
            V_1 = []
            for m_i in range(int(self.m / 2)):
                A = (
                    self.msq
                    * self.M[k, m_i]
                    / tf.sqrt(tf.reduce_sum(tf.square(self.M[k, :])))
                )

                V = A * self.W[k, :] / tf.sqrt(tf.reduce_sum(tf.square(self.W[k, :])))
                V_1.append(V)
            for m_i in range(int(self.m / 2)):
                A = (
                    self.msq
                    * self.M[k, m_i]
                    / tf.sqrt(tf.reduce_sum(tf.square(self.M[k, :])))
                )

                V = (
                    -1.0
                    * A
                    * self.W[k, :]
                    / tf.sqrt(tf.reduce_sum(tf.square(self.W[k, :])))
                )
                V_1.append(V)
            if self.m % 2 != 0:
                V = tf.zeros_like(self.W[k, :])
                V_1.append(V)
            V_2.append(tf.stack(V_1))

        return tf.stack(V_2)

    def transmit(self, X, V):
        Out = []
        for k in range(self.K):
            out_val = tf.matmul(tf.transpose(X[:, :, k]), V[k, :, :])
            out_val1 = out_val * tf.sqrt(self.P[k])
            Out.append(out_val1)
        return Out

    def get_indiv_prob(self, Inp, Des, sigma):
        Des = tf.transpose(Des)
        Des = tf.expand_dims(Des, axis=2)

        Inp = tf.transpose(Inp)
        Inp = tf.expand_dims(Inp, axis=1)

        distance = tf.math.subtract(Inp, Des)
        dist_squared = tf.reduce_sum(tf.square(distance), axis=0)
        prob_exp = tf.exp(-dist_squared / (2 * sigma**2))
        prob = 1 / (self.m**self.K) * tf.reduce_sum(prob_exp, axis=0)
        return prob

    def get_posterior_prob(self, input, Des, SNR):
        snr = 10 ** (SNR / 10)
        sigma = 1 / np.sqrt(snr)
        P = []
        for k in range(self.K):
            P1 = []
            for m_i in range(self.m):
                prob = self.get_indiv_prob(input[k], Des[k][m_i], sigma)
                P1.append(prob)

            P1 = tf.stack(P1)
            P1 = P1 / tf.reduce_sum(P1, axis=0, keepdims=True)
            P.append(P1)
        return P


def get_RX(args, TxSignal, H, Noise):
    K = args.K

    RxSignal = []
    RxSignalH = []
    HOut = []
    for k in range(K):
        RxSig = 0
        hOut = []
        for k1 in range(K):
            rxSig = tf.matmul(TxSignal[k1], H[k][k1].T)
            hOut.append(rxSig)
            RxSig += rxSig
        HOut.append(hOut)
        RxSignalH.append(RxSig)
        RxSig = RxSig + Noise[k]
        RxSignal.append(RxSig)

    return HOut, RxSignalH, RxSignal


def get_desired_RX(args, TX, V, H, XD):
    K, m = args.K, args.m
    DesTX = []
    for k in range(K):
        DesTX1 = []
        for m_i in range(m):
            DesTX1.append(TX.transmit(XD[k][m_i], V))
        DesTX.append(DesTX1)
    DesRX = []
    for k in range(K):
        DesRX1 = []
        for m_i in range(m):
            des = 0
            for k1 in range(K):
                des += tf.matmul(DesTX[k][m_i][k1], H[k][k1].T)
            DesRX1.append(des)
        DesRX.append(DesRX1)
    return DesRX
