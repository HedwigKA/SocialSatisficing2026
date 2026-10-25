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
DEFAULT_ALEPH = 0.0
DEFAULT_TAU_ALPHA = 0.1
DEFAULT_TAU_GAMMA = 0.9
DEFAULT_EG_GAMMA = 0.9
DEFAULT_N_AGENT = 1
DEFAULT_SIMULATION_TIMES = 1000
DEFAULT_EPISODE_TIMES = 1000
DEFAULT_MAX_STEPS = 1000


def parse_args():
    parser = argparse.ArgumentParser(description="Run social satisficing (ES-style) experiments.")
    parser.add_argument("--agents", type=int, default=DEFAULT_N_AGENT)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATION_TIMES)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODE_TIMES)
    parser.add_argument("--aspiration", type=float, default=DEFAULT_ALEPH)
    parser.add_argument("--tau-alpha", type=float, default=DEFAULT_TAU_ALPHA)
    parser.add_argument("--tau-gamma", type=float, default=DEFAULT_TAU_GAMMA)
    parser.add_argument("--eg-gamma", type=float, default=DEFAULT_EG_GAMMA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--non-identical-envs", action="store_true")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_output_path(n_agent):
    path = f"output/SocialSatisficing/{n_agent}_agents"
    os.makedirs(path, exist_ok=True)
    return path


def create_envs(n_agent, non_identical_envs=False, seed=0):
    envs = []
    for n_agent_idx in range(n_agent):
        reward_values = None
        if non_identical_envs:
            reward_values = env.SuboptimaWorld.DEFAULT_REWARD_LAYOUT.copy()
            rng = random.Random(seed + 10000 + n_agent_idx)
            rng.shuffle(reward_values)

        envs.append(
            env.SuboptimaWorld(
                row=9,
                col=9,
                start=int(9 * 9 / 2),
                reward_values=reward_values,
            )
        )
    return envs


def train(envs, simulations, episodes, output_path, params):
    players = [
        model.GRC(
            params["lr"],
            params["dr"],
            params["aleph"],
            params["tau_alpha"],
            params["tau_gamma"],
            params["eg_gamma"],
            envs[0].num_state,
            num_action=envs[0].num_action,
        )
        for _ in range(len(envs))
    ]

    df_reward = [pd.DataFrame() for _ in range(len(players))]
    df_eg = [pd.DataFrame() for _ in range(len(players))]

    for n_simu in range(simulations):
        reward_graph = np.zeros((len(players), episodes))
        eg_graph = np.zeros((len(players), episodes))
        for n_agent in range(len(players)):
            players[n_agent].init_params()

        sys.stdout.write("\r%s/%s" % (str(n_simu), str(simulations - 1)))
        for n_epi in range(episodes):
            eg_list = []
            for n_agent, agent_env in enumerate(envs):
                current_state = agent_env.start_state
                reward = 0
                done = False
                for _ in range(params["max_steps"]):
                    current_action = players[n_agent].get_select_action(current_state)
                    next_state, reward, done = agent_env.step(current_state, current_action)
                    reward_graph[n_agent, n_epi] += reward

                    players[n_agent].update(
                        current_state,
                        current_action,
                        reward,
                        next_state,
                        done=done,
                    )

                    current_state = next_state
                    if done:
                        break

                players[n_agent].update_GRC_params(reward)
                eg_list.append(players[n_agent].EG)
                eg_graph[n_agent, n_epi] = players[n_agent].EG

            max_eg = max(eg_list)
            for n in range(len(players)):
                players[n].update_GRC_reference(max_eg)

        for n in range(len(players)):
            df_reward[n][f"{n_simu}"] = reward_graph[n]
            df_eg[n][f"{n_simu}"] = eg_graph[n]

    for n in range(len(players)):
        df_reward[n].to_csv(output_path + f"/agent_{n}.csv")
        eg_path = f"output/SocialSatisficing/EG/{len(players)}_agents"
        os.makedirs(eg_path, exist_ok=True)
        df_eg[n].to_csv(eg_path + f"/agent_{n}.csv")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    envs = create_envs(
        n_agent=args.agents,
        non_identical_envs=args.non_identical_envs,
        seed=args.seed,
    )
    output_path = make_output_path(args.agents)
    train(
        envs,
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
