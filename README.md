# Automation, Investment Incentives, and the Coordination Problem of Capital

This repository contains the full model implementation, simulation scripts, and experiment configurations used in the study:

**“Automation and the Coordination Problem of Capital: Divergent Investment Incentives in Highly Automated Economies”**

---

## Overview

The model explores how automation investment, income distribution, and governance regimes interact to shape long-run welfare outcomes.

It is implemented using:
- A **Vensim system dynamics model** (.mdl)
- A **Python simulation framework** for running experiments and parameter sweeps

The study identifies:
- A **social reinvestment corridor**
- A **divergence between private and social investment incentives**
- A **bidirectional coordination problem** depending on governance regime

---

## Repository Structure

```text
/venSim_model/
    model.mdl              # Vensim system dynamics model

/python_model/
    main_simulation.py     # main simulation script
    experiment_runner.py   # parameter sweeps / experiments

/experiments/
    config_*.json         # experiment configurations

/results/
    figures/              # generated figures
    data/                 # raw outputs
