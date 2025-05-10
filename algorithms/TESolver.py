"""
TESolver serves as a basic framework, including three common optimization objectives:
    MaxFlow: Maximize the throughput;
    MaxMin: Minimize the max-min fairness;
    MLU: Minimize the maximum link utilization (MLU).

FFCSolver and TEAVARSolver are implemented based on TESolver.
"""

class TESolver:
    def __init__(self, lp, network):
        self.lp = lp
        self.network = network
        self.initialize_optimization_variables()
        
    def initialize_optimization_variables(self):
        for tunnel in self.network.tunnels.values():
            tunnel.init_flow_var(self.lp)
        
    def add_demand_constraints(self):
        for demand in self.network.demands.values():
            flow_on_tunnels = self.lp.Sum([tunnel.v_flow for tunnel in demand.tunnels])
            assert len(demand.tunnels) > 0
            self.lp.Assert(flow_on_tunnels == demand.amount * self.network.scale)

    def add_edge_capacity_constraints(self):
        for edge in self.network.edges.values():
            self.lp.Assert(edge.capacity >= sum(t.v_flow for t in edge.tunnels))
                    
    def Maximize(self, objective):
        self.lp.Maximize(objective)

    def Minimize(self, objective):
        self.lp.Minimize(objective)
        
    def solve(self, obj="MaxFlow"):
        if obj == "MaxFlow":
            self.add_demand_constraints()
            self.add_edge_capacity_constraints()
            self.Maximize(sum([t.v_flow for t in self.network.tunnels.values()]))
        elif obj == "MaxMin":
            amin = self.lp.Variable()
            self.lp.Assert(amin <= 1)
            self.add_edge_capacity_constraints()
            for demand in self.network.demands.values():
                flow_on_tunnels = sum([tunnel.v_flow for tunnel in demand.tunnels])
                assert len(demand.tunnels) > 0
                self.lp.Assert(amin * demand.amount * self.network.scale == flow_on_tunnels)

            self.Maximize(amin)
        elif obj == "MLU":
            Z = self.lp.Variable()
            self.add_demand_constraints()
            for edge in self.network.edges.values():
                self.lp.Assert(Z * edge.capacity >= sum(t.v_flow for t in edge.tunnels))

            self.Minimize(Z)
        else:
            print(f"Objective {obj} is not defined!")
            raise NotImplementedError

        obj = self.lp.Solve()
        self.network.set_tunnel_flow(self.lp.Value)
        return obj