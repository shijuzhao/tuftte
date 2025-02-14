#!/usr/bin/env python

"""
This file contains all the functions to do the experiments. 
For convenience, choose the experiment by inputting the arguments, 
and change the parameters in the file 'benchmark_const.py'.
"""

from benchmark_consts import *
from availability import availability_plot
from cutoff_error import topology_cutoff_error
from dl_experiment import demand_loss_expr, noise_expr, noise_effect
from path_selection import path_selection
from probability_noise import probability_noise
from motivation import ffc_under_uncertain_demand, teavar_under_uncertain_demand
from prediction_details import compute_MSE, check_pos_neg, watch_pos_neg_variation
from algorithms.DoteSolver import NeuralNetworkMaxUtil

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=str, default="dl_experiment")
parser.add_argument("--plot", action="store_true")

args = parser.parse_args(sys.argv[1:])

if args.experiment == "availability":
    availability_plot(TOPOLOGY, NUM_DMS_FOR_TRAIN, 30, CUTOFF, plot=args.plot, start=START, step=STEP, stop=STOP)

elif args.experiment == "cutoff_error":
    topology_cutoff_error("B4", NUM_DMS_FOR_TRAIN, NUM_DMS_FOR_TEST, CUTOFFS, demand_scale=1, weibull_scale=0.0001, plot=args.plot)

elif args.experiment == "dl_experiment":
    demand_loss_expr(TOPOLOGY, NUM_DMS_FOR_TRAIN, NUM_DMS_FOR_TEST, demand_scale=DEMAND_SCALE, plot=args.plot)

elif args.experiment == "ffc_experiment":
    ffc_under_uncertain_demand("GEANT", NUM_DMS_FOR_TRAIN, 50, plot=args.plot)

elif args.experiment == "MSE_test":
    compute_MSE(TOPOLOGY, NUM_DMS_FOR_TRAIN, 100, hist_len=12, plot=args.plot)

elif args.experiment == "pos_neg":
    check_pos_neg("GEANT", NUM_DMS_FOR_TRAIN, NUM_DMS_FOR_TEST, hist_len=12, plot=args.plot)

elif args.experiment == "noise_effect":
    noise_effect("GEANT", NUM_DMS_FOR_TRAIN, NUM_DMS_FOR_TEST, demand_scale=DEMAND_SCALE, plot=args.plot)

elif args.experiment == "noise_experiment":
    noise_expr(TOPOLOGY, NUM_DMS_FOR_TRAIN, NUM_DMS_FOR_TEST, demand_scale=DEMAND_SCALE, noise=NOISE, plot=args.plot)

elif args.experiment == "path_selection":
    path_selection(TOPOLOGIES2, NUM_DMS_FOR_TRAIN, BETAS, 4, 0.0001, paths=PATHS, ksp=[4, 6], plot=args.plot)

elif args.experiment == "probability_noise":
    probability_noise(TOPOLOGIES2, NUM_DMS_FOR_TRAIN, NUM_DMS_FOR_TEST, 0.00001, 1, NOISES, weibull_scale=0.001, plot=args.plot)

elif args.experiment == "teavar_experiment":
    teavar_under_uncertain_demand("GEANT", NUM_DMS_FOR_TRAIN, 30, cutoff=CUTOFF, hist_len=12, plot=args.plot, start=START, step=STEP, stop=STOP)

elif args.experiment == "watch_variation":
    watch_pos_neg_variation("GEANT", NUM_DMS_FOR_TRAIN, hist_len=12, demand_scale=1, plot=args.plot)