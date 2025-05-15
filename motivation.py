"""
Motivational experiments. (Figure 1)
"""

from utils.NetworkParser import *
from utils.prediction import ALLMETHOD, predict_traffic_matrix
from utils.scenario import scenarios_with_k_failed_links, subScenarios
from utils.GurobiSolver import GurobiSolver
from utils.riskMetric import calculate_risk
from algorithms.FFCSolver import FFCSolver
from algorithms.TEAVARSolver import TEAVARSolver
from algorithms.DoteSolver import DoteSolver

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def ffc_under_uncertain_demand(topology, num_dms_for_train=None, num_dms_for_test=None, K=1, hist_len=12, demand_scale=1, plot=False):
    """
    Experiment with respect to FFC. (Figure 1a)

    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        K(int): consider the scenarios with K failed links;
        hist_len(int): the number of historical demand matrices in a training instance;
        demand_scale(float): the demand scale factor used to multiply the demand matrix;
        plot(bool): whether to plot the figure.
    """
    network = parse_topology(topology)
    parse_histories(network, demand_scale)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    scenarios_all = scenarios_with_k_failed_links(int(len(network.edges)/2), K)
    network.set_scenario(scenarios_all)
    method_loss = {}
    for method in ALLMETHOD:
        predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method)
        for tm in predicted_tms:
            network.set_demand_amount(tm)
            lp = GurobiSolver()
            solver = FFCSolver(lp, network, K)
            solver.solve()
            sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
            network.add_sol(sol)

        demand_loss, _ = calculate_risk(network, hist_len)
        network.clear_sol()
        method_loss[method] = demand_loss

    real_tms = network.test_hists._tms[hist_len:]
    for tm in real_tms:
        network.set_demand_amount(tm)
        lp = GurobiSolver()
        solver = FFCSolver(lp, network, K)
        solver.solve()
        sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
        network.add_sol(sol)

    demand_loss, _ = calculate_risk(network, hist_len)
    network.clear_sol()
    method_loss["oracle"] = demand_loss
    
    solver = DoteSolver(network)
    solver.solve()
    demand_loss, _ = calculate_risk(network, hist_len)
    network.clear_sol()
    method_loss["DOTE"] = demand_loss

    loss_reduction = 0.0
    for i, loss in enumerate(method_loss["oracle"]):
        if loss > 0.0:
            loss_reduction += (method_loss["LR"][i] - loss) / loss
        elif method_loss["LR"][i] > 0.0:
            loss_reduction += 1
    print("Method Linear Regression has larger demand loss by", loss_reduction / len(method_loss["oracle"]))
    
    if plot:
        _, ax = plt.subplots()
        labels = []
        x_ticks = []
        for i, method in enumerate(method_loss.keys()):
            x_ticks.append(i)
            if method == "DOTE":
                labels.append("DOTE")
            else:
                labels.append(method + "\n + FFC")
            plt.boxplot(method_loss[method], positions=[i], widths=0.5, showfliers=False)

        ax.set_ylabel("Demand loss (Mbps)", fontsize=20)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels)
        plt.savefig("plot.pdf", bbox_inches='tight')

def teavar_under_uncertain_demand(topology, num_dms_for_train=None, num_dms_for_test=None, cutoff=1e-6, hist_len=12, plot=False, start=0.1, step=0.01, stop=0.4):
    """
    Experiment with respect to TEAVAR. (Figure 1b)

    parameters:
        topology(str): the name of topology;
        num_dms_for_train(int): the number of demand matrices for training/predicting;
        num_dms_for_test(int): the number of demand matrices for testing;
        cutoff(float): consider the scenarios whose probability exceeds cutoff;
        hist_len(int): the number of historical demand matrices in a training instance;
        plot(bool): whether to plot the figure;
        start(float): the smallest demand scale factor;
        step(float): the distance between two adjacent demand scale factors;
        stop(float): the biggest demand scale factor.
    """

    network = parse_topology(topology, use_weibull=True)
    parse_histories(network)
    network.reduce_data(num_dms_for_train, num_dms_for_test)
    parse_tunnels(network, k=8)
    network.prepare_solution_format()
    prob_failure = []
    edge_included = []
    # edge and its inverse edge fail together
    for edge in network.edges:
        if set(edge) in edge_included:
            continue

        prob_failure.append(network.edges[edge].prob_failure)
        edge_included.append(set(edge))
        
    scenarios_all = subScenarios(prob_failure, cutoff)
    network.set_scenario(scenarios_all)
    method_loss = defaultdict(list)
    scales = [start]
    while start < stop:
        start += step
        scales.append(start)

    for method in ALLMETHOD:
        predicted_tms = predict_traffic_matrix(network.train_hists._tms, network.test_hists._tms, hist_len, method)
        for demand_scale in scales:
            network.set_scale(demand_scale)
            for tm in predicted_tms:
                network.set_demand_amount(tm)
                lp = GurobiSolver()
                solver = TEAVARSolver(lp, network)
                solver.solve()
                sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
                network.add_sol(sol)

            _, availability = calculate_risk(network, hist_len)
            network.clear_sol()
            method_loss[method].append(np.mean(availability))

    real_tms = network.test_hists._tms[hist_len:]
    for demand_scale in scales:
        network.set_scale(demand_scale)
        for tm in real_tms:
            network.set_demand_amount(tm)
            lp = GurobiSolver()
            solver = TEAVARSolver(lp, network)
            solver.solve()
            sol = [tunnel.v_flow_value for tunnel in network.solutions.tunnels]
            network.add_sol(sol)

        _, availability = calculate_risk(network, hist_len)
        network.clear_sol()
        method_loss["oracle"].append(np.mean(availability))
    
    solver = DoteSolver(network)
    solver.solve()
    for demand_scale in scales:
        network.set_scale(demand_scale)
        _, availability = calculate_risk(network, hist_len)
        method_loss["DOTE"].append(np.mean(availability))
    
    network.clear_sol()    
    print(method_loss)
    if plot:
        marker_styles = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D', '*', '+', 'x']
        for i, method in enumerate(method_loss):
            if method == "DOTE":
                label = "DOTE"
            else:
                label = method + " + TEAVAR"
            plt.plot(scales, method_loss[method], marker=marker_styles[i], label = label)
            
        plt.xlabel("Demand Scale", fontsize=20)
        plt.ylabel("Availability", fontsize=20)
        plt.ylim((0.999, 1.0001))
        plt.legend()
        plt.savefig("plot.pdf", bbox_inches='tight')