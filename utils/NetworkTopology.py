"""
This file contains data structures of Node, Edge, Demand, Histories, Tunnel, Scenario, Solution and Network.
"""

from itertools import islice
import numpy as np
from tqdm import tqdm

BPS_TO_MBPS = 1e-6

class Node:
    def __init__(self, mkt):
        self.mkt = mkt
        self.latitude = None
        self.longitude = None
        self.devices = []
        self.regions = []
        
    def update(self, device=None, region=None, latitude=None, longitude=None):
        if device and device not in self.devices:
            self.devices.append(device)
        if region and region not in self.regions:
            self.regions.append(region)
        if latitude:
            self.latitude = latitude
        if longitude:
            self.longitude = longitude
            
class Edge:
    #
    # An Edge contains a Graph edge object.
    # and additional attributes.
    # tunnels   - List of tunnels that he edge is part of
    # x_e_t     - Traffic allocation on e for tunnel t
    #
    def __init__(self, e, unity, capacity, prob_failure):
        self.e = e
        self.is_shortcut = False
        self.unity = unity
        self.capacity = capacity
        self.prob_failure = prob_failure
        self.distance = None
        self.tunnels = []
        self.x_e_t = {}

    def __repr__(self):
        return f"{self.e}"

    def add_tunnel(self, t):
        assert self.e in [edge.e for edge in t.path]
        if all(t.pathstr != x.pathstr for x in self.tunnels):
            self.tunnels.append(t)

    def add_shortcut(self, s):
        assert self.e in [edge.e for edge in s.path]
        if all(s.pathstr != x.pathstr for x in self.shortcuts):
            self.shortcuts.append(s)
    
    def increment_capacity(self, capacity_increment):
        self.capacity += capacity_increment

    def add_distance(self, distance):
        self.distance = distance
        
    def init_x_e_vars(self, model):
        for idx in range(len(self.tunnels)):
            tunnel = self.tunnels[idx]
            var = model.Variable()
            model.Assert(var >= 0)
            self.x_e_t[tunnel] = var
        return model            
    
class Demand:
    def __init__(self, src, dst, amount, idx=None):
        self.src = src
        self.dst = dst
        self.amount = amount
        self.id = idx
        self.tunnels = []
        self.b_d = None

    def __repr__(self):
        return f"({self.src}:{self.dst})"

    def add_tunnel(self, t):
        assert t.pathstr.split('-')[0] == self.src
        assert t.pathstr.split('-')[-1] == self.dst
        if t.pathstr not in [x.pathstr for x in self.tunnels]:
            self.tunnels.append(t)
            
    def init_b_d(self, model):
        self.b_d = model.Variable(lb=0)

class Histories:
    def __init__(self, files_tms, htype, num_nodes, tm_length_func=lambda: 5, max_steps=60):
        self._num_nodes = num_nodes
        self._tms = []
        self._tm_times = []
        self._tm_ind = 0
        self._type = htype
        self._max_steps = max_steps
        self._tm_mask = np.ones((num_nodes, num_nodes), dtype=bool).flatten()
        self._tm_mask[np.eye(num_nodes).flatten() == 1] = False

        for fname in files_tms:
            print('[+] Populating TMS.')
            self._populate_tms(fname, tm_length_func)
        
    def _populate_tms(self, fname, tm_length_func):
        with open(fname) as f:
            for line in tqdm(f.readlines()):
                try:
                    tm = self._parse_tm_line(line)
                except:
                    import pdb;
                    pdb.set_trace()

                tm_time = tm_length_func()
                tm = tm * BPS_TO_MBPS
                self._tms.append(tm)
                self._tm_times.append(tm_time)

    def __len__(self):
        return len(self._tms)
    
    def _parse_tm_line(self, line):
        tm = np.array([np.float64(_) for _ in line.split(" ") if _], dtype=np.float64)
        tm = tm.reshape((self._num_nodes, self._num_nodes))
        tm = (tm - tm * np.eye(self._num_nodes))
        return tm.flatten()[self._tm_mask]
    
    def get_next(self):
        tm = self._tms[self._tm_ind]
        tm_time = self._tm_times[self._tm_ind]
        return tm, tm_time
    
    def num_tms(self):
        return len(self._tms)
    
    def num_histories(self):
        return 1
    
    def reset(self):
        self._tm_ind = 0

class Scenario:
    def __init__(self, bitmap, probability=0):
        self.bitmap = bitmap
        self.prob = probability
        self.u_s = None
        self.failed_tunnels = []

    def init_u_s(self, model):
        self.u_s = model.Variable(lb=0)
        
class Tunnel:
    def __init__(self, path, pathstr):
        # path here is a list of edges
        self.path = path
        self.pathstr = pathstr
        self.weight = 0       
        self.id = None    # identity number
        self.v_flow = None    # Solver variable for flow
        self.v_flow_value = 0    # optimal solution from solver
        # add this tunnel to all relevant edges
        for e in path:
            e.add_tunnel(self)
        
    def name(self):
        return self.pathstr

    def __repr__(self):
        return self.name()

    def __len__(self):
        return len(self.path)
        
    def init_flow_var(self, model):
        self.v_flow = model.Variable(lb=0)
    
    def add_weight(self, weight):
        self.weight = weight

class Solution:
    def __init__(self, tunnels):
        self.tunnels = tunnels
        self.val = []

    def add_sol(self, sol):
        self.val.append(sol)

    def clear(self):
        self.val = []

class Network:
    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.tunnels = {}
        self.demands = {}
        self.train_hists = None
        self.test_hists = None
        self.scenarios = []
        self.solutions = None
        self.tunnel_type = ""
        self.graph = None
        self.scale = 1
        
    def add_node(self, mkt, region=None, device=None):
        assert isinstance(mkt, str)
        if mkt in self.nodes:
            node = self.nodes[mkt]
        else:
            node = Node(mkt)
            self.nodes[mkt] = node
        node.update(device=device, region=region)
        return node

    def add_edge(self, mktA, mktB, unity=None, capacity=0, prob_failure=0):
        assert isinstance(mktA, str)
        assert isinstance(mktB, str)
        self.add_node(mktA)
        self.add_node(mktB)
        if mktA == mktB: return None
        
        if (mktA, mktB) in self.edges:
            edge = self.edges[(mktA, mktB)]
            edge.increment_capacity(capacity)
        else:
            edge = Edge((mktA, mktB), unity, capacity, prob_failure)
            self.edges[(mktA, mktB)] = edge
            
        return edge

    def remove_zero_capacity_edges(self):
        edges_to_rm = []
        for edge in self.edges:
            if self.edges[edge].capacity == 0:
                edges_to_rm.append(edge)
        for edge in edges_to_rm:
            self.edges.pop(edge)
                
    def add_demand(self, src, dst, amount, scale=1, idx=None):
        assert isinstance(src, str)
        assert isinstance(dst, str)
        self.add_node(src)
        self.add_node(dst)
        
        if (src, dst) not in self.demands:
            self.demands[(src, dst)] = Demand(src, dst, amount*scale, idx)

        return self.demands[(src, dst)]

    def add_sol(self, sol):
        self.solutions.add_sol(sol)

    def add_tunnel(self, tunnel):
        assert isinstance(tunnel, list)
        assert isinstance(tunnel[0], str)
        tunnel_str = "-".join(tunnel)
        if tunnel_str in self.tunnels: return
        
        tunnel_start = tunnel[0]
        tunnel_end = tunnel[-1]
        tunnel_edge_list = []
        for src, dst in zip(tunnel, tunnel[1:]):
            nodeA = self.add_node(src)
            nodeB = self.add_node(dst)
            assert (src, dst) in self.edges
            edge = self.edges[(src, dst)]
            tunnel_edge_list.append(edge)

        tunnel_obj = Tunnel(tunnel_edge_list, tunnel_str)
        self.tunnels[tunnel_str] = tunnel_obj        
        if (tunnel_start, tunnel_end) in self.demands:
            demand = self.demands[(tunnel_start, tunnel_end)]
            demand.add_tunnel(tunnel_obj)
    
    def clear_sol(self):
        self.solutions.clear()
    
    def init_tunnel(self):
        self.tunnels = {}
        for d in self.demands.values():
            d.tunnels = []
        for e in self.edges.values():
            e.tunnels = []
       
    def prepare_solution_format(self):
        # Solution format depends on the number of tunnels
        assert(self.tunnels)
        num_nodes = len(self.nodes)
        if len(self.demands) == 0:
            for t in self.tunnels.values():
                tunnel_start = t.path[0].e[0]
                tunnel_end = t.path[-1].e[-1]
                if (tunnel_start, tunnel_end) in self.demands:
                    demand = self.demands[(tunnel_start, tunnel_end)]
                else:
                    idx = int(tunnel_start)*(num_nodes-1) + int(tunnel_end) - int(int(tunnel_start)<int(tunnel_end))
                    demand = self.add_demand(tunnel_start, tunnel_end, 0, idx=idx)

                demand.add_tunnel(t)
                
        tunnels = []
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src == dst:
                    continue
                tunnels.extend(self.demands[(str(src), str(dst))].tunnels)

        assert(len(self.tunnels) == len(tunnels))
        for i, t in enumerate(tunnels):
            t.id = i
        self.solutions = Solution(tunnels)
    
    def reduce_data(self, num_train=None, num_test=None):
        if num_train is not None and num_train < len(self.train_hists):
            self.train_hists._tms = self.train_hists._tms[-num_train:]
        if num_test is not None and num_test < len(self.test_hists):
            self.test_hists._tms = self.test_hists._tms[:num_test]

    def set_demand_amount(self, traffic_matrix):
        num_nodes = len(self.nodes)
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if (str(src), str(dst)) in self.demands:
                    idx = self.demands[(str(src), str(dst))].id
                    self.demands[(str(src), str(dst))].amount = max(traffic_matrix[idx], 0)
    
    def set_scale(self, scale=1):
        self.scale = scale
        
    def set_scenario(self, scenarios):
        self.scenarios = scenarios
        for s in scenarios:
            failed_links = []
            probability = 1.0
            i = 0
            edge_included = []
            for edge in self.edges:
                if set(edge) in edge_included:
                    continue

                if not s.bitmap[i]:
                    failed_links.append(set(edge))
                    probability *= self.edges[edge].prob_failure
                else:
                    probability *= 1 - self.edges[edge].prob_failure
                    
                edge_included.append(set(edge))
                i += 1

            # supplement probability of scenario
            if s.prob == 0:
                s.prob = probability
            
            for t in self.tunnels.values():
                for edge in t.path:
                    if set(edge.e) in failed_links:
                        s.failed_tunnels.append(t.pathstr)
                        break

    def set_tunnel_flow(self, val_func):
        for t in self.tunnels.values():
            t.v_flow_value = max(val_func(t.v_flow), 0)

    def to_nx(self):
        import networkx
        graph = networkx.DiGraph()
        for n in self.nodes.keys():
            graph.add_node(n)
        # Putting 100 km distance for all edges as of now, fix later.
        for (s,t) in self.edges:
            graph.add_edge(s, t, distance=400)
        return graph

    def draw(self, labels={}):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = self.to_nx()
        pos = nx.spring_layout(G, weight=1, k=0.5, )
                            #    pos={'1':(0,0), '2':(0,2), '3':(4,2), '4':(4,-2), 
                            #         '5': (8,-1), '6': (8,2), '7': (12,2), '8':(12,-2),
                            #         '9':(16,0), '10': (16,2), '11': (20, 4), '12': (20, 0)}, 
                            #    fixed=['1', '2', '3', '4', '5', '6', '7', '8', '9',
                            #           '10', '11', '12'])
        plt.figure(figsize=(10,8))
        options = {
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12
        }
        nx.draw(G, pos, edge_color = 'black', linewidths = 1,
                # connectionstyle='arc3, rad = 0.1',
                node_size = 500, node_color = 'pink',
                alpha = 0.9, with_labels = True, **options)
        nx.draw_networkx_edge_labels(G, pos, font_size=8,
                                     label_pos=0.3,
                                     edge_labels=labels)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        plt.axis('off')
        plt.show()
        plt.savefig('nx.pdf')

    def k_shortest_paths(self, source, target, k):
        import networkx as nx
        G = self.to_nx()
        return list(islice(nx.shortest_simple_paths(G, source, target), k))