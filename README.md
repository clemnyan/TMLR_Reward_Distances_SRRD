# TMLR REWARD DISTANCES - SRRR

This repository contains the code for the paper "Reward Distance Comparisons Under Transition Sparsity" published in TMLR (https://openreview.net/forum?id=haP586YomL).

It implements reward distance calculations and related experiments as described in the paper.

## Repository Contents
- **reward_distance_utils.py**: Utility functions. Contains code for SRRD, EPIC, and DARD as well as sample-based approximations and variations (unbiased estimates, regression etc.) 

- **calculate_reward_distances.py**: Main script for calculating reward distances using the Gridworld and Bouncing Balls domains (Mostly Experiment 1, and different case studies)

- **calculate_distances_regression.py**: Incorporates regression in reward distance calculations.

- **bouncing_balls.py**: Adapted domain for for the bouncing balls simulation.

- **KNN_classification**: Module for KNN-based classification(Experiment 2), suing reward pseudometrics as distance measures. Contains other domains: Starcraft 2, Montezuma's Revenge, Robomimic, MIMC_IV, Drone Combat.

- **ADDITIONAL_EXPERIMENTS**: Other additional experiments and scripts.


## Usage
1. Clone the repo: `git clone https://github.com/clemnyan/TMLR_REWARD_DISTANCES_SRRRD.git`
2. Install dependencies: [list dependencies, e.g., `pip install numpy pandas scikit-learn`]
3. Run the main script: `python calculate_reward_distances.py`

## Status
This repository is actively being updated. Check back for additional documentation.
