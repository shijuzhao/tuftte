# TUFTTE: handling Traffic Uncertainty in Failure-Tolerant Traffic Engineering

## 1. Overview
TUFTTE is a novel TE framework with decision-focused learning. Its objective function and constraints can be flexibly specified by the network operators. It is currently built upon the cvxpylayer, but I am also exploring other efficient tools.

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

We import `DOTE` from their public repository without modification.

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
It is simple to conduct the experiments. The configuration of parameters is completed in `benchmark_consts.py`. You just need to run `main.py` with different values of the argument `--experiment`.
```bash
python3 main.py
```

- `./main.py --experiment ffc_experiment` for Fig. 1a.
- `./main.py --experiment teavar_experiment` for Fig. 1b.
- `./main.py --experiment dl_experiment` for Fig. 4.
- `./main.py --experiment noise_effect` for Table II.
- `./main.py --experiment availability` for Fig. 5.
- `./main.py --experiment MSE_test` for Fig. 6a.
- `./main.py --experiment pos_neg` for Fig. 6b.
- `./main.py --experiment watch_variation` for Fig. 7.

## 5. Citation
Please cite our paper if our contributions benefit your research.