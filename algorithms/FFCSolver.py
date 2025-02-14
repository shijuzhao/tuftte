import itertools

from .TESolver import TESolver

class FFCSolver(TESolver):
    def __init__(self, lp, network, k=1):
        TESolver.__init__(self, lp, network)
        self.k = k
            
        for demand in self.network.demands.values():
            demand.init_b_d(self.lp)
        
    def failure_scenario_edge_constraint(self, failed_tunnels):
        for demand in self.network.demands.values():
            flow_on_tunnels = sum([t.v_flow for t in demand.tunnels if t not in failed_tunnels])
            self.lp.Assert(demand.b_d <= flow_on_tunnels)
                            
    def add_demand_constraints(self):
        for demand in self.network.demands.values():
            self.lp.Assert(demand.b_d <= demand.amount * self.network.scale)
            
    def pairwise_failures(self, k):
        return itertools.combinations(self.network.edges.values(), r = k)

    def solve(self):
        self.add_demand_constraints()
        self.add_edge_capacity_constraints()
        scenarios = self.pairwise_failures(self.k)
        for s in scenarios:
            # include the inverse edge.
            edges = [edge.e for edge in s]
            edges.extend([edge[::-1] for edge in edges])
            failed_tunnels = set.union(*(set(self.network.edges[e].tunnels) for e in edges))
            self.failure_scenario_edge_constraint(failed_tunnels)

        objective = sum([demand.b_d for demand in self.network.demands.values()])
        self.Maximize(objective)

        obj = self.lp.Solve()
        self.network.set_tunnel_flow(self.lp.Value)
        return obj