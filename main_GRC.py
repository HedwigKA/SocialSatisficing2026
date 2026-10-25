import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

import env
import model


# default hyper parameter
DEFAULT_LR = 0.1
DEFAULT_DR = 0.9
DEFAULT_ALEPH = 6.5
DEFAULT_TAU_ALPHA = 0.1
DEFAULT_TAU_GAMMA = 0.9
DEFAULT_EG_GAMMA = 0.9
DEFAULT_SIMULATION_TIMES = 1000
DEFAULT_EPISODE_TIMES = 1000
DEFAULT_MAX_STEPS = 1000


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRC experiments.")
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATION_TIMES)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODE_TIMES)
    parser.add_argument("--aspiration", type=float, default=DEFAULT_ALEPH)
    parser.add_argument("--tau-alpha", type=float, default=DEFAULT_TAU_ALPHA)
    parser.add_argument("--tau-gamma", type=float, default=DEFAULT_TAU_GAMMA)
    parser.add_argument("--eg-gamma", type=float, default=DEFAULT_EG_GAMMA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_output_path():
    path = "output/GRC/"
    os.makedirs(path, exist_ok=True)
    return path


def train(agent_env, simulations, episodes, output_path, params):
    player = model.GRC(
        params["lr"],
        params["dr"],
        params["aleph"],
        params["tau_alpha"],
        params["tau_gamma"],
        params["eg_gamma"],
        agent_env.num_state,
        num_action=agent_env.num_action,
    )
    reward_graph = np.zeros((simulations, episodes))
    step_graph = np.zeros((simulations, episodes))
    for n_simu in range(simulations):
        player.init_params()

        sys.stdout.write("\r%s/%s" % (str(n_simu), str(simulations - 1)))
        for n_epi in range(episodes):
            current_state = agent_env.start_state
            reward = 0
            step = 0
            for step in range(params["max_steps"]):
                current_action = player.get_select_action(current_state)
                next_state, reward, done = agent_env.step(current_state, current_action)
                player.update(current_state, current_action, reward, next_state, done=done)

                current_state = next_state
                if done:
                    break
            player.update_GRC_params(reward)
            reward_graph[n_simu, n_epi] = reward
            step_graph[n_simu, n_epi] = step + 1

    reward_graph = pd.DataFrame(reward_graph.T)
    reward_graph.to_csv(output_path + f"/aleph_{params['aleph']}.csv")
    return reward_graph, step_graph


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    agent_env = env.SuboptimaWorld(row=9, col=9, start=int(9 * 9 / 2))
    output_path = make_output_path()
    train(
        agent_env,
        args.simulations,
        args.episodes,
        output_path,
        {
            "lr": DEFAULT_LR,
            "dr": DEFAULT_DR,
            "aleph": args.aspiration,
            "tau_alpha": args.tau_alpha,
            "tau_gamma": args.tau_gamma,
            "eg_gamma": args.eg_gamma,
            "max_steps": args.max_steps,
        },
    )
