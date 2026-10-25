# Social Satisficing

This repository contains Python implementations, experiment scripts, notebooks, and simulation outputs for the study:

> **Social satisficing: Multi-agent reinforcement learning with satisficing agents**  
> Daisuke Uragami, Noriaki Sonota, and Tatsuji Takahashi  
> *BioSystems*, Volume 243, Article 105276 (2024)

Paper DOI: [10.1016/j.biosystems.2024.105276](https://doi.org/10.1016/j.biosystems.2024.105276)

## Overview

The project investigates **social satisficing**, a multi-agent reinforcement learning framework in which agents share a small amount of information to coordinate exploration and exploitation. Instead of sharing complete value functions, the proposed **episodic satisficing (ES)** algorithm shares the maximum episodic return achieved by the group and uses it to set adaptive aspiration levels.

The experiments use **SuboptimaWorld**, a 9 × 9 GridWorld containing one optimal goal with reward 8 and seven suboptimal goals with rewards 1–7. The repository compares:

- **ES — Episodic satisficing:** agents share the maximum episodic return.
- **SS — State-wise satisficing:** agents share aspiration levels for every state.
- **Q-shared:** agents share all Q-values and use an ε-greedy policy.
- **RS — Risk-sensitive satisficing:** the underlying action-selection framework used by ES and SS.
- **GRC — Global reference conversion:** converts an episodic aspiration into state-wise aspiration levels.

## Main idea

The risk-sensitive satisficing value is computed as:

\[
RS(s,a) = \tau(s,a) [Q(s,a) - \rho(s)]
\]

where:

- `Q(s, a)` is the learned action value;
- `τ(s, a)` is the reliability of the state-action pair; and
- `ρ(s)` is the local aspiration level.

For ES, the shared episodic aspiration is the best moving-average return among the agents:

\[
\rho_E = \max_m R_E^m
\]

Each agent then converts this group-level reference into local state-wise aspiration levels using global reference conversion. This allows the agents to preserve behavioral diversity while collectively improving their exploration target.

## Repository contents

### Python modules and experiment entry points

- `env.py` — SuboptimaWorld environment.
- `model.py` — social satisficing model.
- `model_RS.py` — risk-sensitive satisficing model.
- `model_Qshare.py` — Q-shared model.
- `main_social_satisficing.py` — social satisficing experiments.
- `main_RS.py` — RS experiments.
- `main_Qleaning.py` — Q-learning/Q-shared experiments.
- `main_GRC.py` — global reference conversion experiments.

### Analysis and results

- `graph.ipynb` — notebook for generating figures and analyzing results.
- `graph/` — generated plots corresponding to the simulation experiments.
- `output/` — CSV files containing experiment results for different algorithms, numbers of agents, and exploration settings.

The repository also includes Jupyter checkpoint files under `.ipynb_checkpoints/`.

## Reproducing the experiments

The scripts require Python and commonly used numerical/data-analysis packages such as NumPy, pandas, and Matplotlib. A typical workflow is:

```bash
python main_social_satisficing.py
python main_RS.py
python main_Qleaning.py
python main_GRC.py
```

Generated data can be inspected in `output/`, and plots can be regenerated with `graph.ipynb`.

> The exact runtime and output depend on the experiment parameters configured in each script. Review the scripts before running large batches of simulations.

## Key findings from the paper

The simulations reported in the paper show that:

1. ES can adapt the exploration–exploitation balance without an externally tuned ε-greedy exploration schedule.
2. Sharing only a scalar group-level return can outperform methods that share substantially more information.
3. Increasing the number of ES agents improves both the average reward and the probability of discovering the optimal goal.
4. ES maintains useful diversity among agents while their learned behavior eventually converges toward the optimal reward.
5. ES remains effective when agents learn in similar but non-identical environments, whereas methods that copy complete Q-tables are more sensitive to environmental differences.

## Paper metadata

The accompanying paper metadata and extracted equations, tables, figures, and references are provided in `social_satisficing_full.json` when that file is included in the repository.

## Citation

```text
Uragami, D., Sonota, N., & Takahashi, T. (2024).
Social satisficing: Multi-agent reinforcement learning with satisficing agents.
BioSystems, 243, 105276.
https://doi.org/10.1016/j.biosystems.2024.105276
```

## License and attribution

Please consult the source paper and repository history for licensing and attribution information. The paper is published under the **CC BY-NC-ND 4.0** license.
