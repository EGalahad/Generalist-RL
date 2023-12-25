from generalist_rl.api.datatypes import SampleBatch, RolloutRequest

from generalist_rl.impl.environment.gymnasium import GymEnv
from generalist_rl.impl.algorithm.ppo import (
    PPOBufferTensorGPU,
    ActorCriticPolicy,
    PPOTrainer,
)
from generalist_rl.utils.logging import get_logging_str_from_dict

import torch
import tqdm
import os
import wandb
from collections import deque
import gymnasium as gym


def main():
    wandb.init(project="generalist-rl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps_per_env = 500
    update_interval = 4
    log_interval = 10

    env = GymEnv(
        "CartPole-v1",
        num_envs=1,
        autoreset=True,
        max_episode_steps=num_steps_per_env,
        device=device,
    )
    has_continuous_action_space = isinstance(env._env.action_space, gym.spaces.Box)

    policy = ActorCriticPolicy(
        env.num_obs,
        env.num_actions,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=0.6,
        device=device,
    )

    if os.path.exists("policy.pth"):
        policy_ckpt = torch.load("policy.pth")
        policy.load_checkpoint(policy_ckpt)
        print(f"loaded policy ckpt")

    buffer = PPOBufferTensorGPU(
        env.num_envs,
        num_steps_per_env * update_interval,
        env.num_obs,
        env.num_privileged_obs,
        env.num_actions,
        device=device,
        has_continuous_action_space=has_continuous_action_space,
    )

    trainer = PPOTrainer(policy, lr_actor=3e-4, lr_critic=1e-3)

    episode_return_buffer = deque(maxlen=100)
    episode_length_buffer = deque(maxlen=100)
    episode_return = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=device)
    episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=device)

    stats = {}
    total_timesteps = int(2e5)
    update_timesteps = num_steps_per_env * update_interval
    log_timesteps = num_steps_per_env * log_interval
    timesteps = 0
    pbar = tqdm.tqdm(range(total_timesteps))
    while timesteps < total_timesteps:
        episode_return[:] = 0
        episode_length[:] = 0
        env_step_result = env.reset()

        # rollout
        policy.eval_mode()
        for step in range(num_steps_per_env):
            transition = SampleBatch()
            timesteps += 1

            transition.obs = env_step_result.obs
            transition.policy_state = None

            request = RolloutRequest(obs=env_step_result.obs)
            rollout_result = policy.rollout(requests=request)

            env_step_result = env.step(rollout_result.action)

            transition.action = rollout_result.action
            transition.analyzed_result = rollout_result.analyzed_result
            transition.reward = env_step_result.reward
            transition.done = env_step_result.done

            buffer.put(transition)

            episode_return += env_step_result.reward
            episode_length += 1

            reset_ids = torch.where(
                torch.logical_or(env_step_result.done, env_step_result.truncated)
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
            trainer_step_result = trainer.step(samples)

        if timesteps % log_timesteps == 0:
            trainer_step_result.stats.update(
                {
                    "step": trainer_step_result.step,
                    "timesteps": timesteps,
                    "mean_return": torch.mean(torch.tensor(episode_return_buffer)),
                    "mean_length": torch.mean(torch.tensor(episode_length_buffer)),
                }
            )
            wandb.log(trainer_step_result.stats)
            logging_str = get_logging_str_from_dict(trainer_step_result.stats)
            print(logging_str)

            pbar.update(log_timesteps)
            pbar.set_description(
                f"step: {timesteps}, mean_return: {torch.mean(torch.tensor(episode_return_buffer)):.4f}, mean_length: {torch.mean(torch.tensor(episode_length_buffer)):.4f}"
            )

            # record stats
            for key in trainer_step_result.stats.keys():
                if key not in stats:
                    stats[key] = []
                stats[key].append(trainer_step_result.stats[key])

    # initial_iteration = policy.version + 1
    # pbar = tqdm.tqdm(range(initial_iteration, initial_iteration + total_iterations))
    # for epoch in range(initial_iteration, initial_iteration + total_iterations):
    #     # rollout
    #     policy.eval_mode()
    #     for step in range(num_steps_per_env):
    #         obs, critic_obs = env_step_result.obs['obs'], env_step_result.obs['critic_obs']
    #         transition.obs = {
    #             "obs": obs,
    #             "critic_obs": critic_obs
    #         }
    #         transition.policy_state = None

    #         request = RolloutRequest(obs={'obs': obs})
    #         rollout_result = policy.rollout(requests=request)

    #         transition.action = rollout_result.action
    #         transition.analyzed_result = rollout_result.analyzed_result

    #         env_step_result = env.step(rollout_result.action)

    #         transition.reward = env_step_result.reward
    #         transition.done = env_step_result.done

    #         # if step == num_steps_per_env - 1:
    #         #     transition.truncated = torch.ones(1, dtype=torch.bool)

    #         buffer.put(transition)

    #         episode_return += env_step_result.reward
    #         episode_length += 1

    #         reset_ids = torch.where(torch.logical_or(env_step_result.done, env_step_result.truncated))[0]
    #         episode_return_buffer.extend(episode_return[reset_ids].cpu().numpy().tolist())
    #         episode_length_buffer.extend(episode_length[reset_ids].cpu().numpy().tolist())
    #         episode_return[reset_ids] = 0
    #         episode_length[reset_ids] = 0

    #     # train
    #     if epoch > 0 and epoch % update_interval == 0:
    #         policy.train_mode()
    #         samples = buffer.get()
    #         buffer.clear()
    #         trainer_step_result = trainer.step(samples)
    #         policy.inc_version()

    #         trainer_step_result.stats.update({ "step": trainer_step_result.step })
    #         # wandb.log(trainer_step_result.stats)
    #         if epoch % log_interval == 0:
    #             logging_str = get_logging_str_from_dict(trainer_step_result.stats)
    #             print(logging_str)
    #     pbar.update(1)
    #     pbar.set_description(f"step: {epoch}, mean_return: {torch.mean(torch.tensor(episode_return_buffer)):.4f}, mean_length: {torch.mean(torch.tensor(episode_length_buffer)):.4f}")

    # policy_ckpt = policy.get_checkpoint()
    # torch.save(policy_ckpt, "policy.pth")


if __name__ == "__main__":
    main()
