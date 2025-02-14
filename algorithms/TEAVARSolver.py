from .TESolver import TESolver

class TEAVARSolver(TESolver):
    def __init__(self, lp, network, beta=0.9):
        TESolver.__init__(self, lp, network)
        self.beta = beta

        for s in network.scenarios:
            s.init_u_s(self.lp)
                            
    def add_demand_constraints(self, alpha):
        for s in self.network.scenarios:
            for demand in self.network.demands.values():
                if demand.amount > 0:
                    expr = sum(t.v_flow for t in demand.tunnels if t.pathstr not in s.failed_tunnels)
                    self.lp.Assert(expr / demand.amount / self.network.scale + alpha + s.u_s >= 1)

    def solve(self):
        alpha = self.lp.Variable(lb=0)
        self.add_demand_constraints(alpha)
        self.add_edge_capacity_constraints()
        objective = sum(s.u_s * s.prob for s in self.network.scenarios) / (1-self.beta) + alpha
        self.Minimize(objective)
        obj = self.lp.Solve()
        self.network.set_tunnel_flow(self.lp.Value)
        return obj