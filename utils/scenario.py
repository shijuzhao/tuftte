"""
Generate scenario set.
"""

import numpy as np
from functools import reduce
from copy import copy
from itertools import combinations

from .NetworkTopology import Scenario

def scenarios_with_k_failed_links(num_edges, k):
    """
    compute scenarios whose number of failed links is equal to k.
    Note: Because scenarios with k failures are more serious, we can neglect those with less failures
    """
    positions = combinations(range(num_edges), k)
    bitmap_arrays = [[pos not in comb for pos in range(num_edges)] for comb in positions]
    scnarios = [Scenario(bitmap) for bitmap in bitmap_arrays]
    return scnarios

def subScenarios(original, cutoff, first=True, last=False):
    print(f"Computing scenarios cutoff={cutoff}...")
    scenarios = []
    def subScenariosRecursion(original, cutoff, remaining=[], offset=0, partial=[]):
        if partial == []:  # first iteration
            scenarios.append(Scenario(np.ones(len(original), dtype=bool), reduce(lambda x, y: x * y, [1 - i for i in original])))
            remaining = copy(original)
        else:
            probs = [1 - i for i in original]
            bitmap = np.ones(len(original), dtype=bool)
            for index in partial:
                probs[index] = original[index]
                bitmap[index] = False
            product = reduce(lambda x, y: x * y, probs)
            if product >= cutoff:
                scenarios.append(Scenario(bitmap, product))
            else:
                return

        offset = len(original) - len(remaining)
        for i in range(len(remaining)):
            subScenariosRecursion(original, cutoff, remaining[i+1:], offset, partial+[offset+i])

        return
    
    subScenariosRecursion(original, cutoff)
    if not first:
        scenarios = scenarios[1:]

    s = sum(i.prob for i in scenarios)
    if last:
        scenarios.append(Scenario(np.zeros_like(original), 1 - s))
    elif s < 1:
        for i in scenarios:
            i.prob /= s

    return scenarios

def calculateLossReallocation(network):
    scenarios = network.scenarios
    loss = np.zeros(len(scenarios))
    total_demand = sum(demand.amount for demand in network.demands.values()) * network.scale
    tunnel_idx = {}
    for i, t in enumerate(network.tunnels.keys()):
        tunnel_idx[t] = i

    for idx, s in enumerate(scenarios):
        # compute amount of traffic without failure
        routed = np.zeros(len(network.tunnels))
        for demand in network.demands.values():
            ratio = sum(t.v_flow_value for t in demand.tunnels if t.pathstr not in s.failed_tunnels)
            if ratio == 0:
                continue
            ratio = demand.amount * network.scale / ratio
            for t in demand.tunnels:
                if t.pathstr not in s.failed_tunnels:
                    routed[tunnel_idx[t.pathstr]] = t.v_flow_value * ratio

        loss[idx] = sum(routed)
        # compute congestion loss
        for edge in network.edges.values():
            edge_utilization = sum(routed[tunnel_idx[t.pathstr]] for t in edge.tunnels)
            loss[idx] -= max(0, edge_utilization - edge.capacity)

        loss[idx] = 1 - loss[idx] / total_demand

    return loss