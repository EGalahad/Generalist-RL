from generalist_rl.api.datatypes import SampleBatch, RolloutRequest

from generalist_rl.impl.environment.legged_gym import LeggedGymEnv, get_args
from generalist_rl.impl.algorithm.ppo import (
    BufferNamedArray,
    ActorCriticPolicy,
    PPOTrainer,
)
from generalist_rl.utils.logging import get_logging_str_from_dict
from generalist_rl.utils.namedarray import from_dict

import torch
import tqdm
import os
import wandb
from collections import deque


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_timesteps = int(1e4)
    num_steps_per_env = 25
    update_interval = 1
    log_interval = 1

    lr_actor = 3e-4
    lr_critic = 1e-3
    critic_loss_weight = 1.0
    entropy_loss_weight = 0.001
    ppo_epochs = 5
    gamma = 0.99
    lamda = 0.95
    num_minibatch = 4
    shuffle = True

    args = get_args()
    env = LeggedGymEnv(args)
    has_continuous_action_space = True
    action_std_init = 1.0

    policy = ActorCriticPolicy(
        env.num_obs,
        env.num_actions,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=action_std_init,
        device=device,
    )

    if os.path.exists("policy.pth"):
        policy_ckpt = torch.load("policy.pth")
        policy.load_checkpoint(policy_ckpt)
        print(f"loaded policy ckpt")

    buffer = BufferNamedArray(
        num_envs=env.num_envs,
        num_transitions_per_env=num_steps_per_env * update_interval,
        num_obs=env.num_obs,
        num_privileged_obs=env.num_privileged_obs,
        num_action=env.num_actions,
        has_continuous_action_space=has_continuous_action_space,
    )

    trainer = PPOTrainer(
        policy,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        ppo_epochs=ppo_epochs,
        critic_loss_weight=critic_loss_weight,
        entropy_loss_weight=entropy_loss_weight,
        gamma=gamma,
        lamda=lamda,
        num_minibatch=num_minibatch,
        shuffle=shuffle,
    )

    wandb.init(
        project="generalist-rl",
        job_type="legged_gym",
        config={
            "total_timesteps": total_timesteps,
            "num_steps_per_env": num_steps_per_env,
            "update_interval": update_interval,
            "log_interval": log_interval,
            "num_envs": env.num_envs,
            "env_args": args,
            "policy_args": {
                "has_continuous_action_space": has_continuous_action_space,
                "action_std_init": action_std_init,
            },
            "trainer_args": {
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                "ppo_epochs": ppo_epochs,
                "critic_loss_weight": critic_loss_weight,
                "entropy_loss_weight": entropy_loss_weight,
                "gamma": gamma,
                "lamda": lamda,
                "num_minibatch": num_minibatch,
                "shuffle": shuffle,
                
            },
        },
    )
    episode_return_buffer = deque(maxlen=100)
    episode_length_buffer = deque(maxlen=100)
    episode_return = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=device)
    episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=device)

    stats = {}
    update_timesteps = num_steps_per_env * update_interval
    log_timesteps = num_steps_per_env * log_interval
    timesteps = 0
    # pbar = tqdm.tqdm(range(total_timesteps))
    env_step_result = env.reset()
    while timesteps < total_timesteps:
        # rollout
        policy.eval_mode()
        for step in range(num_steps_per_env):
            transition = SampleBatch()
            timesteps += 1

            transition.obs = from_dict(env_step_result.obs)
            transition.policy_state = None

            request = RolloutRequest(obs=from_dict(env_step_result.obs))
            rollout_result = policy.rollout(requests=request)
            env_step_result = env.step(rollout_result.action)

            transition.action = rollout_result.action
            transition.analyzed_result = rollout_result.analyzed_result
            transition.reward = env_step_result.reward
            transition.done = env_step_result.done
            transition.truncated = env_step_result.truncated

            buffer.put(transition)

            # update episode return and length
            episode_return += env_step_result.reward
            episode_length += 1
            reset_ids = torch.where(
                torch.logical_or(transition.done, transition.truncated)
            )[0]
            episode_return_buffer.extend(
                episode_return[reset_ids].cpu().numpy().tolist()
            )
            episode_length_buffer.extend(
                episode_length[reset_ids].cpu().numpy().tolist()
            )
            episode_return[reset_ids] = 0
            episode_length[reset_ids] = 0

        # train
        if timesteps % update_timesteps == 0:
            policy.train_mode()
            samples = buffer.get()
            buffer.clear()
            # if timesteps // update_timesteps >= 30:
            #     breakpoint()
            trainer_step_result = trainer.step(samples)
            trainer_step_result.stats.update(
                {
                    "step": trainer_step_result.step,
                    "timesteps": timesteps,
                    "mean_return": torch.mean(
                        torch.tensor(episode_return_buffer)
                    ).item(),
                    "mean_length": torch.mean(
                        torch.tensor(episode_length_buffer)
                    ).item(),
                }
            )
            wandb.log(trainer_step_result.stats)

        if timesteps % log_timesteps == 0:
            logging_str = get_logging_str_from_dict(trainer_step_result.stats)
            print(logging_str)

            # pbar.update(log_timesteps)
            # pbar.set_description(
            #     f"step: {timesteps}, mean_return: {torch.mean(torch.tensor(episode_return_buffer)):.4f}, mean_length: {torch.mean(torch.tensor(episode_length_buffer)):.4f}"
            # )

            # record stats
            for key in trainer_step_result.stats.keys():
                if key not in stats:
                    stats[key] = []
                stats[key].append(trainer_step_result.stats[key])


if __name__ == "__main__":
    main()
