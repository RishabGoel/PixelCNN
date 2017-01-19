import os
import sys
import time
import json
import argparse

import numpy as np
# import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--data_dir', type=str, default='/tmp/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/tmp/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')

# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
# parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
# parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')

# optimization parameters
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
# parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
# parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')

# evaluation
# parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()

args = parser.parse_args()

