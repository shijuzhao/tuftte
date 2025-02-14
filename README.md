# TUFTTE: handling Traffic Uncertainty in Failure-Tolerant Traffic Engineering

## 1. Overview
This is a naive version of TUFTTE. I will supplement the description here later this year. I am also exploring another efficient tools.

## 2. Code Structure
```
Tuftte
├── algorithms
│   ├── DOTE-main
│   ├── __init__.py
│   ├── DoteSolver.py
│   ├── TEAVARSolver.py
│   ├── TESolver.py
│   └── TUFTTESolver.py
├── data
│   ├── gml_gen_topo.py
│   └── snd_gen_topo.py
├── utils
│   ├── __init__.py
│   ├── CvxpySolver.py
│   ├── GurobiSolver.py
│   ├── helper.py
│   ├── history.py
│   ├── NetworkParser.py
│   ├── NetworkTopology.py
│   ├── prediction.py
│   ├── riskMetric.py
│   └── scenario.py
├── availability.py
├── benchmark_consts.py
├── dl_experiment.py
├── environment.yml
├── main.py
├── motivation.py
└── prediction_details.py
```

## 3. Getting Started
Install the conda environment using `environment.yml`.
```bash
conda create -n tuftte --file environment.yml
```

Prepare datasets. Download `abilene.txt`, `directed-abilene-zhang-5min-over-6months-ALL-native.tar` from SNDlib and move them to `data/`. Generate the topology by the following commands.
```bash
cd data/
python3 snd_gen_topo.py
```

## 4. Running Experiments
```bash
python3 main.py
```