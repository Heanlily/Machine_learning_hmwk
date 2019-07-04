#!/usr/bin/env python3
from scipy.stats import multivariate_normal
import numpy as np
if not __file__.endswith('_em_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as LastName_em_gaussian.py (replacing LastName with your last name)!')
    exit(1)

DATA_PATH = "/u/cs246/data/em/" #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    clusters = []
    if args.cluster_num:
        lambdas = np.zeros(args.cluster_num)
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.array([np.identity(2) for i in range(len(lambdas))])
        else:
            sigmas = np.identity(2)
        np.random.seed(8)

        x = np.random.rand(len(lambdas))
        lambdas = np.round(x / np.sum(x), decimals=2)

        mus = np.random.rand(args.cluster_num, 2)
    else:
        lambdas = []
        mus = []
        sigmas = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #lambda mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1
                lambda_k, mu_k_1, mu_k_2, sigma_k_0_0, sigma_k_0_1, sigma_k_1_0, sigma_k_1_1 = map(float,line.split())
                lambdas.append(lambda_k)
                mus.append([mu_k_1, mu_k_2])
                sigmas.append([[sigma_k_0_0, sigma_k_0_1], [sigma_k_1_0, sigma_k_1_1]])
        lambdas = np.asarray(lambdas)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(lambdas)

    model=(lambdas,mus,sigmas)
    #NOTE: if args.tied was provided, sigmas will have a different shape
    return model

def train_model(model, train_xs, dev_xs, args):
    #NOTE: you can use multivariate_normal like this:
    #probability_of_xn_given_mu_and_sigma = multivariate_normal(mean=mu, cov=sigma).pdf(xn)
    #remove when model training is implemented
    lambdas, mus, sigmas = model

    gamma = []
    for i in range(args.iterations):
        for j in range(len(lambdas)):
            gamma.append(lambdas[j] * multivariate_normal(mus[j], sigmas[j]).pdf(train_xs))
    gamma = np.array(gamma) / np.sum(gamma, axis=0)
    for j in range(len(lambdas)):
        mus[j] = 1 / np.sum(gamma[j]) * np.dot(gamma[j], train_xs)

        if not args.tied:
            sigmas[j] = 1 / np.sum(gamma[j]) * np.dot(gamma[j] * (train_xs - mus[j]).T, train_xs - mus[j])
        else:
            sigmas = 1 / np.sum(gamma[j]) * np.dot(gamma[j] * (train_xs - mus[j]).T, train_xs - mus[j])
        lambdas[j] = np.sum(gamma[j]) / train_xs.shape[0]
    model=((lambdas, mus, sigmas))
    return model

def average_log_likelihood(model, data):
    ll = 0.0
    lambdas, mus, sigmas = model
    for i in range(data.shape[0]):
        ll_class = 0.0
        for j in range(len(lambdas)):
            ll_class += lambdas[j] * multivariate_normal(mus[j], sigmas[j]).pdf(data[i])
        ll += np.log(ll_class)

    return ll / data.shape[0]
    #remove when average log likelihood calculation is implemented

def extract_parameters(model):
    lambdas,mus,sigmas=model
    return lambdas, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points.')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    ll_train = average_log_likelihood(model, train_xs)
    print('Train LL: {}'.format(ll_train))
    if not args.nodev:
        ll_dev = average_log_likelihood(model, dev_xs)
        print('Dev LL: {}'.format(ll_dev))
    lambdas, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Lambdas: {}'.format(intersperse(' | ')(np.nditer(lambdas))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
