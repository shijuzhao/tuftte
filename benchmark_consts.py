"""
Constants for benchmarking
"""

# # Abilene parameters
# TOPOLOGY = "Abilene"
# DEMAND_SCALE = 15
# START = 16
# STOP = 20.2
# STEP = 0.2

# GEANT parameters
TOPOLOGY = "GEANT"
DEMAND_SCALE = 1
START = 0.2
STOP = 0.32
STEP = 0.01

# Web parameters
# TOPOLOGY = 'Facebook_DB'
# DEMAND_SCALE = 1

TOPOLOGIES = ['Abilene', 'ATT', 'B4', 'IBM', 'Sprint']
# TOPOLOGIES = ['XNet']
ALGORITHMS = ['FFC-1', 'TEAVAR', 'SMORE', 'MaxMin', 'FFC-2']
# ALGORITHMS = ['FFC-1', 'TEAVAR', 'SMORE', 'MaxMin']
CUTOFF = 1e-6
NUM_DMS_FOR_TRAIN = 1000
NUM_DMS_FOR_TEST = 300
NOISE = 0.3

CUTOFFS = [0.0001, 0.00001, 0.000001, 0.0000001]

TOPOLOGIES2 = ['B4']
BETAS = [0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
PATHS = ["SMORE", "FFC"]

NOISES = [0.01, 0.05, 0.1, 0.15, 0.2]