from collections import defaultdict
import copy
from functools import reduce
import json
import os.path

import numpy as np
from scipy.stats import entropy


def calc_kp_divergence(X, Y):
    bins = np.linspace(min(np.min(X), np.min(Y)), max(np.max(X), np.max(Y)), 20)

    px, _ = np.histogram(X, bins=bins, density=True)
    qx, _ = np.histogram(Y, bins=bins, density=True)

    px += np.finfo(float).eps
    qx += np.finfo(float).eps

    kl_divergence = entropy(px, qx)
    return kl_divergence


def calc_symmmetric_kl(X, Y):
    return 0.5 * calc_kp_divergence(X, Y) + 0.5 * calc_kp_divergence(Y, X)


def generate_sample(theta, zeta, activation):
    z0 = activation((zeta + theta[0]))
    z1 = activation((z0 + theta[1]))
    z2 = activation((z1 + theta[2]))
    return (float(round(z0, 3)), float(round(z1, 3)), float(round(z2, 3)))


def generate_samples(theta, num_samples, activation, seed=0):
    x_rng = np.random.RandomState(seed)
    samples = []
    noise_values = x_rng.uniform(-0.5, 0.5, num_samples)
    for zeta in noise_values:
        sample = generate_sample(theta, zeta, activation)
        samples.append(sample)
    return samples


def myjoin(S, t):
    return f"{t}".join([str(si) for si in S])


def generate_samples_set(Theta, num_samples, activation, seed, str_key=False):
    if str_key:
        sample_set = dict(
            [(str(theta), generate_samples(theta, num_samples=num_samples, activation=activation, seed=seed)) for theta
             in Theta])
    else:
        sample_set = dict(
            [(theta, generate_samples(theta, num_samples=num_samples, activation=activation, seed=seed)) for theta in
             Theta])
    return sample_set


def flatten_set(Theta_train, Theta_test):
    set_train = set(reduce(lambda a, b: a + b, [[(i, z) for i, z in enumerate(s)] for s in Theta_train]))
    set_test = set(reduce(lambda a, b: a + b, [[(i, z) for i, z in enumerate(s)] for s in Theta_test]))
    return set_train == set_test


def build_dist(samples):
    theta_2_in = defaultdict(list)
    theta_2_out = defaultdict(list)
    theta_3_in = defaultdict(list)
    theta_3_out = defaultdict(list)
    for theta, theta_samples in samples.items():
        for z in theta_samples:
            theta_2_in[theta[1]].append(z[0])
            theta_2_out[theta[1]].append(z[1])
            theta_3_in[theta[2]].append(z[1])
            theta_3_out[theta[2]].append(z[2])
    return {
        "theta_2_in": theta_2_in,
        "theta_2_out": theta_2_out,
        "theta_3_in": theta_3_in,
        "theta_3_out": theta_3_out,
    }


def get_sample_kl(train_samples, test_samples):
    train_sample_d = build_dist(train_samples)
    test_sample_d = build_dist(test_samples)
    kl_divergence = []
    for train_sample_di, test_sample_di in zip(train_sample_d.values(), test_sample_d.values()):
        kl_divergence.extend(
            [calc_symmmetric_kl(train_sample_di[ikey], test_sample_di[ikey]) for ikey in train_sample_di.keys()])
    return np.average(kl_divergence)


def run(h=3, L=2, n=500, activation="lrelu", ratio=0.5, seed_num=20):
    train_seed = 0
    test_seed = train_seed + 1

    activation_dict = {
        "tanh": np.tanh,
        "lrelu": lambda x: x if x > 0 else 0.5 * x
    }
    activation_func = activation_dict[activation]

    theta_step = [[x] for x in list(range(-L, L + 1)) if x != 0]

    Theta_all = copy.deepcopy(theta_step)
    for _ in range(h - 1):
        theta_bar_new = []
        for tb in Theta_all:
            for ts in theta_step:
                theta_bar_new.append(tuple(tb) + tuple(ts))
        Theta_all = theta_bar_new

    split_ok = {}
    t = 0
    s = 0
    while True:
        print(s)
        Theta_train = []
        Theta_test = []
        rng2 = np.random.RandomState(s)
        s += 1
        for i, theta in enumerate(Theta_all):
            if rng2.random() < ratio:
                Theta_test.append(theta)
            else:
                Theta_train.append(theta)
        if len(Theta_train) == 0 or len(Theta_test) == 0:
            continue
        equal = flatten_set(Theta_train, Theta_test)
        if equal:
            split_ok[t] = Theta_train, Theta_test
            t += 1
            if t == seed_num:
                break

    for t in split_ok.keys():
        print(t)
        datasets_out = {
            "train": generate_samples_set(split_ok[t][0], num_samples=n, activation=activation_func, seed=train_seed,
                                          str_key=True),
            "testZ": generate_samples_set(split_ok[t][0], num_samples=n, activation=activation_func, seed=test_seed,
                                          str_key=True),
            "testB": generate_samples_set(split_ok[t][1], num_samples=n, activation=activation_func, seed=test_seed,
                                          str_key=True),
            "testC050": generate_samples_set([tuple([0.45 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC060": generate_samples_set([tuple([0.56 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC070": generate_samples_set([tuple([0.67 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC080": generate_samples_set([tuple([0.78 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC090": generate_samples_set([tuple([0.89 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC095": generate_samples_set([tuple([0.95 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC105": generate_samples_set([tuple([1.05 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC110": generate_samples_set([tuple([1.11 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC120": generate_samples_set([tuple([1.22 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC130": generate_samples_set([tuple([1.33 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC140": generate_samples_set([tuple([1.44 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
            "testC150": generate_samples_set([tuple([1.55 * xi for xi in x]) for x in split_ok[t][0]], num_samples=n,
                                             activation=activation_func, seed=test_seed, str_key=True),
        }
        dataset_dir = os.path.join("data",
                                   f"discrete14_{h}_{str(int(ratio * 100)).zfill(2)}_{activation}_l{L}_s{str(t).zfill(2)}")
        os.makedirs(dataset_dir, exist_ok=True)
        for fname, fdata in datasets_out.items():
            json.dump(fdata, open(os.path.join(dataset_dir, f"{fname}.json"), "w"), indent=2)


if __name__ == '__main__':
    run(ratio=0.05, seed_num=5)
    run(ratio=0.1, seed_num=5)
    run(ratio=0.2, seed_num=5)
    run(ratio=0.3, seed_num=5)
    run(ratio=0.4, seed_num=5)
    run(ratio=0.5, seed_num=5)
