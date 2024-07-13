# Code for the Paper: Preserving the Privacy of Reward Functions in MDPs through Deception

Published at the 27th European Conference on Artificial Intelligence (ECAI 2024)

## Installation

1. Install the requirements provided in `requirements.txt`.
2. Replace `util.py` and `mce_irl.py` with the given files in the source code of the installed version of imitation, as there are some errors in the original package.
3. There might be compatibility issues due to changes in `stable_baselines3` and `imitation` to incorporate `gymnasium` instead of `gym`. Make sure to download the exact versions as provided in `requirements.txt`.

## Running the experiments

### 1. Running the evaluation scripts

Run the `discrete_eval.py` file in `/test_discrete/` to compare with the MEIR algorithm, and `dp_eval.py` to compare against the DQFN algorithm.

#### Configuration options:

- `beta`: -1 if MM, MEIR; else `beta > 0` for MMBE.
- `use_model_free`: Set to `True` to run IQ-Learn.
- `ntrajs` (number of demo trajectories): Set as -1 to pass true occupancy measures. Must be > 0 if using IQ-learn.
- `policies`: The private algorithm to be run. Valid options:
  - `['MaxEnt']` for MEIR
  - `['KL', 'WD (Linear)', 'WDNN (with NN)', 'f_div_kl', 'f_div_rkl', 'f_div_hellinger', 'f_div_pearson', 'f_div_tv', 'f_div_js']` for different variants of MM.
- `randomization`: Reward constraint(s) `E_{min} = E_hat + (E_star - E_hat) * r`.
- `sigma` (comparison with DQFN): Noise parameter, higher sigma -> more noise.
- `n_modes`: Number of policies to mix in MM^{mix} set = 1 for regular MM.
- `strat`: Set `random` for `IRL^{random}` and `max` for `IRL^{max}`, else set to `None`.

All configurations are passed as lists, and results are generated for all possible combinations.

### 2. Set the log directory

Set the `log_dir` where you intend to store the results.

### 3. Environments

- The environment name must be from the list: `['random', 'FrozenLake_{grid_size}', 'FourRooms_{room_size}', 'CyberBattle']`.
- To run the CyberBattle environment:
  1. Generate the data of the network configs by running the `get_network_configs.ipynb` notebook.
  2. Follow up by running the `read_network_configs.ipynb` notebook.

### 4. Plotting the results

Use the `test_discrete/plot.py` file to plot the results.

#### Plotting configurations:

- Specify the `read_dir` from which the results have to be loaded.
- Set `average = False` to plot the Pearson correlation across different values of r. Otherwise, the results will be averaged over r, returning a scalar.
- Set other configurations (`beta`, `ntrajs`, etc.) based on the exact result required.