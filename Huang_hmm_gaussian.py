#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
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
    if args.cluster_num:
        mus=np.random.rand(args.cluster_num,2)
        if not args.tied:
            sigmas = np.array([np.eye(2) for i in range(args.cluster_num)])
        else:
            sigmas = np.eye(2)

        transitions = np.random.rand((args.cluster_num,args.cluster_num))#transitions[i][j] = probability of moving from cluster i to cluster j
        transitions = transitions/transitions.sum(axis=1,keepdims=1)

        initials = np.array([1/args.cluster_num for i in range(args.cluster_num)]) #probability for starting in each state


        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        raise NotImplementedError #remove when random initialization is implemented
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = (mus,sigmas,initials,transitions)
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    mus, sigmas, initials, transitions=extract_parameters(model)
    alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0


    for i in range(args.cluster_num):
        if not args.tied:
            alphas[0, i] = initials[i] * multivariate_normal(data[0], mus[i], sigmas[i])
        else:
            alphas[0, i] = initials[i] * multivariate_normal(data[0], mus[i], sigmas)


    s=sum(alphas[0])
    log_likelihood+=log(s)

    for i in range(args.cluster_num):
        alphas[0][i]=alphas[0][i]/s


    for t in range(1,len(data)):
        for k in range(args.cluster_num):
            if not args.tied:
                alphas[t,k] = np.sum(alphas[t - 1, :] * transitions[:, k]) * multivariate_normal(data[t], mus[k],
                                                                                       sigmas[k])
            else:
                alphas[t,k] = np.sum(alphas[t - 1, :] * transitions[:, k]) * multivariate_normal(data[t], mus[k],
                                                                                       sigmas)
        s=sum(alphas[t])
        alphas[t]=alphas[t]/s
        log_likelihood+=log(s)

    return alphas,log_likelihood

    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.

def backward(model, data, args):
    mus, sigmas, initials, transitions=extract_parameters(model)
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))

    betas[len(data)-1,:]=1


    for t in range(len(data-1))[::-1]:
        for k in range(args.cluster_num):
            for j in range(len(betas[t+1])):
                if not args.tied:
                    betas[t][k] += betas[t+1,j] * transitions[k,j] * multivariate_normal(data[t+1], mus[j], sigmas[j])
                else:
                    betas[t][k] += betas[t+1,j] * transitions[k,j] * multivariate_normal(data[t+1],mus[j], sigmas)

            betas[t][k]=betas[t][k]/sum(betas[t])

    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    return betas

def train_model(model, train_xs, dev_xs, args):
    mus, sigmas, initials, transitions=extract_parameters(model)

    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)

    length,k=len(train_xs),args.cluster_num
    train_ll,dev_ll=[],[]
     #remove when model training is implemented

    for iter in range(args.iterations):
        alphas,ll = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)
        gamas = np.zeros((length, k))
        eikeshen = np.zeros((length,k,k))

        for t in range(length):
            for i in range(k):
                gamas[t,i] = alphas[t,i] * betas[t,i]

            for i in range(k):
                for j in range(k):
                    if t != 0:
                        eikeshen[t,i,j] = alphas[t-1,i] * betas[t,j] * transitions[i,j] * multivariate_normal(train_xs[t], mus[j], sigmas[j])
            gamas[t] = gamas[t]/np.sum(gamas[t])

            if t:
                eikeshen[t,] = eikeshen[t,]/np.sum(eikeshen[t,])


        initials = gamas[0]
        for i in range(args.cluster_num):
            # Update mu
            mus[i] = np.dot(gamas[:,i], train_xs) / np.sum(gamas[:,i])
            # Update sigma
            if not args.tied:
                sigmas[i] = np.dot(gamas[:,i] * (train_xs - mus[i]).T, (train_xs - mus[i])) / np.sum(gamas[:,i])
            else:
                sigmas += np.dot(gamas[:,i] * (train_xs - mus[i]).T, (train_xs - mus[i]))
            # Update transition matrix
            for j in range(args.cluster_num):
                transitions[i,j] = np.sum(eikeshen[:,i,j]) / np.sum(gamas[:,i])

        if args.tied:
            sigmas = sigmas / train_xs.shape[0]

        model = (mus,sigmas,initials,transitions)

        train_ll.append(average_log_likelihood(model,train_xs,args))

        if not args.nodev:
            dev_ll.append(average_log_likelihood(model,dev_xs,args))

        if args.iterations > 10:
            if iter % round(args.iterations/10) == 0:
                print('Iteration #{}'.format(iter))
            elif iter == (args.iterations - 1):
                print('Iteration #{}'.format(iter+1))
        else:
            print('Iteration #{}'.format(iter))


    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    ll = 0.0
    alphas, llll = forward(model, data, args)

    ll = llll/ data.shape[0]
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model[2]
    transitions = model[3]
    mus = model[0]
    sigmas = model[1]
    return mus, sigmas, initials, transitions

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
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
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
