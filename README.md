# Social Satisficing

This repository contains Python implementations and experiment scripts for:

> **Social satisficing: Multi-agent reinforcement learning with satisficing agents**  
> Daisuke Uragami, Noriaki Sonota, and Tatsuji Takahashi  
> *BioSystems*, 243, 105276 (2024)

DOI: [10.1016/j.biosystems.2024.105276](https://doi.org/10.1016/j.biosystems.2024.105276)

## What is implemented

- **SuboptimaWorld** (`env.py`): 9x9 grid, center start, 4 actions, 8 terminal circumferential goals with rewards 1–8.
- **Q-learning**: α=0.1, γ=0.9 defaults (configurable from CLI).
- **RS (risk-sensitive satisficing)**: includes reliability (`tau`) updates.
- **GRC (global reference conversion)**: episodic-to-local aspiration conversion.
- **ES-style social satisficing**: each agent keeps separate Q/tau and shares only the group max episodic reference.
- **Q-shared baseline**: element-wise max Q-sharing across agents.

## Important implementation notes

- Terminal Q updates are **done-aware** (no bootstrap term from terminal next states).
- Episodes now have a configurable safety cap (`--max-steps`) to avoid unbounded loops.
- Multi-agent experiments use **one environment instance per agent**.
- Non-identical environment mode (`--non-identical-envs`) shuffles reward assignments independently per agent while keeping the same 8 goal coordinates.
- Deterministic seeding is supported via `--seed` (Python `random` + NumPy).

## Repository structure

- `env.py` — GridWorld/SuboptimaWorld environment.
- `model.py` — RS/GRC/social satisficing model code.
- `model_RS.py` — RS-sharing model code.
- `model_Qshare.py` — Q-shared model code.
- `main_social_satisficing.py` — ES-style social satisficing experiments.
- `main_RS.py` — RS-sharing experiments.
- `main_Qleaning.py` — Q-shared experiments.
- `main_GRC.py` — GRC experiments.
- `tests/` — lightweight unit tests.

## Running experiments

All scripts expose CLI options so source edits are not required for common settings.

### Social satisficing (ES-style)

```bash
python main_social_satisficing.py \
  --agents 4 \
  --simulations 1000 \
  --episodes 1000 \
  --aspiration 0.0 \
  --seed 0 \
  --max-steps 1000
```

Non-identical/shuffled environments (paper Figure 6 style):

```bash
python main_social_satisficing.py --agents 4 --non-identical-envs --seed 0
```

### RS-sharing

```bash
python main_RS.py --agents 8 --aspiration 0.0 --seed 0 --max-steps 1000
```

Output is normalized under `output/RSShare/`.

### Q-shared

```bash
python main_Qleaning.py --agents 4 --tg 600 --seed 0 --max-steps 1000
```

### GRC fixed aspiration experiments

```bash
python main_GRC.py --aspiration 6.5 --seed 0
python main_GRC.py --aspiration 7.5 --seed 0
python main_GRC.py --aspiration 8.5 --seed 0
```

## Tests

Run:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Current tests cover:

- SuboptimaWorld reward placement and shuffled rewards
- boundary no-op behavior
- Q-learning terminal/non-terminal updates
- RS tie handling
- GRC reference conversion consistency
- Q-sharing element-wise max behavior

## Reproducibility and limitations

This repository is an implementation based on the paper and supports the key experimental mechanisms, including shuffled non-identical environments.

It is **not claimed as an exact numeric reproduction** of every published figure out-of-the-box, since published curves depend on long-run simulation settings, seeds, plotting/aggregation details, and run-time environment.

## Citation

```text
Uragami, D., Sonota, N., & Takahashi, T. (2024).
Social satisficing: Multi-agent reinforcement learning with satisficing agents.
BioSystems, 243, 105276.
https://doi.org/10.1016/j.biosystems.2024.105276
```
