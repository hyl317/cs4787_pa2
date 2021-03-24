import math
import time
import scipy
from scipy.special import softmax
from matplotlib import pyplot
import os
import numpy as np
from numpy import random
import matplotlib
import mnist
import pickle
matplotlib.use('agg')

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables
#wgt0 = np.load('./initial_W.npy')
wgt0 = np.zeros((10, 784))


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory,
                                 return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training()
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        Xs_tr = np.ascontiguousarray(Xs_tr)
        Ys_tr = np.ascontiguousarray(Ys_tr)
        Xs_te, Lbls_te = mnist_data.load_testing()
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should implement this
    Xs = Xs[:, ii]
    Ys = Ys[:, ii]
    loss_grad = np.dot(softmax(np.dot(W, Xs), axis=0) - Ys, Xs.T)
    reg_grad = gamma * W
    grad = loss_grad / Xs.shape[1] + reg_grad
    return grad


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a fraction of incorrect labels (a number between 0 and 1)
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    n = Xs.shape[1]
    pred = np.dot(W, Xs).T
    pred_max_idx = np.argmax(pred, axis=1)
    gt = np.argmax(Ys.T, axis=1)
    correct = np.sum(pred_max_idx == gt)
    return (n-correct) / n


# ALGORITHM 1: run stochastic gradient descent on a multinomial logistic regression objective, with regularization
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def stochastic_gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    T = num_epochs * n
    W = W0
    res = []
    for t in range(T):
        if t % monitor_period == 0:
            res.append(np.copy(W))
        ii = np.random.choice(n, 1, replace=True)
        W -= alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    res.append(np.copy(W))
    return res


# ALGORITHM 2: run stochastic gradient descent with sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of iterations (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" iterations
def sgd_sequential_scan(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    W = W0
    res = []
    for t in range(num_epochs):
        for i in range(n):
            if (t*n+i) % monitor_period == 0:
                res.append(np.copy(W))
            ii = np.array([i])
            W -= alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    res.append(np.copy(W))
    return res


# ALGORITHM 3: run stochastic gradient descent with minibatching
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    ratio = math.floor(n/B)
    T = num_epochs * ratio
    W = W0
    res = []
    for t in range(T):
        if t % monitor_period == 0:
            res.append(np.copy(W))
        ii = np.random.choice(n, B, replace=True)
        W -= alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    res.append(np.copy(W))
    return res

# ALGORITHM 4: run stochastic gradient descent with minibatching and sequential sampling order
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches


def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    W = W0
    res = []
    floor = math.floor(n/B)
    for t in range(num_epochs):
        for i in range(floor):
            if (t*floor+i) % monitor_period == 0:
                res.append(np.copy(W))
            ii = np.arange(i*B, i*B+B)
            W -= alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    res.append(np.copy(W))
    return res


def compute_errors(wgts, Xs_tr, Ys_tr, Xs_te, Ys_te):
    tr_errs = []
    te_errs = []

    for i in range(len(wgts)):
        # if i % 10 != 0:
        #     continue
        tr_err = multinomial_logreg_error(Xs_tr, Ys_tr, wgts[i])
        tr_errs.append(tr_err)

        te_err = multinomial_logreg_error(Xs_te, Ys_te, wgts[i])
        te_errs.append(te_err)

    return tr_errs, te_errs


def generate_plots(tr_errs, te_errs):
    num_epochs = len(tr_errs[0])

    fig, ax = pyplot.subplots(1, 2)
    fig.suptitle("Part 1 errors")

    x_axis = [i for i in range(num_epochs)]

    for i in range(4):
        ax[0].plot(x_axis, tr_errs[i], label="Train error alg "+str(i+1))
        ax[1].plot(x_axis, te_errs[i], label="Test error alg "+str(i+1))

    ax[0].legend()
    ax[1].legend()

    pyplot.savefig('./p1errors.png')


def parts13(Xs_tr, Ys_tr, Xs_te, Ys_te):
    start_time = time.time()
    p1a1 = stochastic_gradient_descent(
        Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), 0.001, 10, 6000)
    elapsed_time = time.time() - start_time
    print('p1a1 elapsed_time: ', elapsed_time)
    print(len(p1a1))
    np.save('./p1a1.npy', np.array(p1a1))

    start_time = time.time()
    p1a2 = sgd_sequential_scan(
        Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), 0.001, 10, 6000)
    elapsed_time = time.time() - start_time
    print('p1a2 elapsed_time: ', elapsed_time)
    print(len(p1a2))
    np.save('./p1a2.npy', np.array(p1a2))

    start_time = time.time()
    p1a3 = sgd_minibatch(Xs_tr, Ys_tr, 0.0001,
                         np.copy(wgt0), 0.05, 60, 10, 100)
    elapsed_time = time.time() - start_time
    print('p1a3 elapsed_time: ', elapsed_time)
    print(len(p1a3))
    np.save('./p1a3.npy', np.array(p1a3))

    start_time = time.time()
    p1a4 = sgd_minibatch_sequential_scan(
        Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), 0.05, 60, 10, 100)
    elapsed_time = time.time() - start_time
    print('p1a4 elapsed_time: ', elapsed_time)
    print(len(p1a4))
    np.save('./p1a4.npy', np.array(p1a4))

    p1a1errs = compute_errors(p1a1, Xs_tr, Ys_tr, Xs_te, Ys_te)
    p1a2errs = compute_errors(p1a2, Xs_tr, Ys_tr, Xs_te, Ys_te)
    p1a3errs = compute_errors(p1a3, Xs_tr, Ys_tr, Xs_te, Ys_te)
    p1a4errs = compute_errors(p1a4, Xs_tr, Ys_tr, Xs_te, Ys_te)

    all_errs = [p1a1errs, p1a2errs, p1a3errs, p1a4errs]
    tr_errs = [err[0] for err in all_errs]
    # print('tr_errs')
    # print(tr_errs)
    te_errs = [err[1] for err in all_errs]
    # print('te_errs')
    # print(te_errs)
    generate_plots(tr_errs, te_errs)


def part2sgd(Xs_tr, Ys_tr, Xs_te, Ys_te):
    step_sizes = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004,
                  0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.1]
    p2p1wgts = []
    for alpha in step_sizes:
        p2p1wgts.append(stochastic_gradient_descent(
            Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), alpha, 10, 6000))
    np.save('./p2p1wgts.npy', np.array(p2p1wgts))

    tr_errs = []
    te_errs = []
    for wgt in p2p1wgts:
        tr_err, te_err = compute_errors(wgt, Xs_tr, Ys_tr, Xs_te, Ys_te)
        tr_errs.append(tr_err)
        te_errs.append(te_err)
    np.save('./p2p1tr_err.npy', np.array(tr_errs))
    np.save('./p2p1te_err.npy', np.array(te_errs))


def part2alg2(Xs_tr, Ys_tr, Xs_te, Ys_te):
    step_sizes = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.004,
                  0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.05, 0.1]
    p2p1wgts = []
    for alpha in step_sizes:
        p2p1wgts.append(stochastic_gradient_descent(
            Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), alpha, 10, 6000))
    np.save('./p2p1wgts.npy', np.array(p2p1wgts))

    tr_errs = []
    te_errs = []
    for wgt in p2p1wgts:
        tr_err, te_err = compute_errors(wgt, Xs_tr, Ys_tr, Xs_te, Ys_te)
        tr_errs.append(tr_err)
        te_errs.append(te_err)
    np.save('./p2p1tr_err.npy', np.array(tr_errs))
    np.save('./p2p1te_err.npy', np.array(te_errs))

def part2alg4(Xs_tr, Ys_tr, Xs_te, Ys_te):
    p2p4wgts = []
    for i in range(51):
        for B in [10, 20, 30, 60, 120, 240, 500, 1000]:
            alpha = 0.002 * i
            p2p4wgts.append(sgd_minibatch_sequential_scan(
                Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), alpha, B, 10, 100))
    np.save('./p2p4wgts.npy', np.array(p2p4wgts))

    tr_errs = []
    te_errs = []
    for wgt in p2p4wgts:
        tr_err, te_err = compute_errors(wgt, Xs_tr, Ys_tr, Xs_te, Ys_te)
        tr_errs.append(tr_err)
        te_errs.append(te_err)
    np.save('./p2p4tr_err.npy', np.array(tr_errs))
    np.save('./p2p4te_err.npy', np.array(te_errs))


def p2p5plot_algo1():
    # p2p5wgts = stochastic_gradient_descent(
    #   Xs_tr, Ys_tr, 0.0001, np.copy(wgt0), 0.006, 10, 6000)
    #np.save('./p2p5wgts.npy', np.array(p2p5wgts))
    p2p5wgts = np.load('./p2p5wgts.npy')

    tr_err, te_err = compute_errors(p2p5wgts, Xs_tr, Ys_tr, Xs_te, Ys_te)

    fig, ax = pyplot.subplots(1, 2)
    fig.suptitle("Part 2.5 SGD alpha=6e-3")

    x_axis = [i/10 for i in range(101)]

    ax[0].plot(x_axis, tr_err, label="Train error")
    ax[1].plot(x_axis, te_err, label="Test error")

    ax[0].legend()
    ax[1].legend()

    pyplot.savefig('./p2p5_sgd_6e-3.png')

def p2p5plot_algo4(Xs_tr, Ys_tr, Xs_te, Ys_te):
    alpha = 0.054
    B = 10
    W0 = np.random.rand(10, 784)
    logs = sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, 1e-4, W0, alpha, B, 10, 600)
    tr_err = []
    te_err = []
    for W in logs:
        tr_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, W))
        te_err.append(multinomial_logreg_error(Xs_te, Ys_te, W))
    
    fig, (ax1, ax2) = pyplot.subplots(1,2)
    fig.suptitle('PA2 part5 SGD_minibatch_seqScan')
    xs = [i/10 for i in range(101)]
    ax1.plot(xs, tr_err)
    ax1.set_title("training error")

    ax2.plot(xs, te_err)
    ax2.set_title("testing error")
    pyplot.savefig("part2_5_algo4.png", dpi=300)



if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    # parts13(Xs_tr, Ys_tr, Xs_te, Ys_te)
    # part2sgd(Xs_tr, Ys_tr, Xs_te, Ys_te)
    # part2alg4(Xs_tr, Ys_tr, Xs_te, Ys_te)
    # p2p5plot_algo1()
    p2p5plot_algo4(Xs_tr, Ys_tr, Xs_te, Ys_te)
    #part2alg2(Xs_tr, Ys_tr, Xs_te, Ys_te)
