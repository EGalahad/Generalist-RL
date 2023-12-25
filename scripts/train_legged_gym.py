from generalist_rl.api.datatypes import SampleBatch, RolloutRequest

from generalist_rl.impl.environment.legged_gym import LeggedGymEnv, get_args
from generalist_rl.impl.algorithm.ppo import PPOBufferTensorGPU, ActorCriticPolicy, PPOTrainer
from generalist_rl.utils.logging import get_logging_str_from_dict

import torch
import tqdm
import os
import wandb
from collections import deque


def main():
    # wandb.init(project="generalist-rl")
    device = torch.device('cuda')
    args = get_args()
    env = LeggedGymEnv(args)

    num_steps_per_env = 25

    policy = ActorCriticPolicy(env.num_obs, env.num_actions, has_continuous_action_space=True, action_std_init=0.6, device=device)
    # load ckpt if exists
    if os.path.exists("policy.pth"):
        policy_ckpt = torch.load("policy.pth")
        policy.load_checkpoint(policy_ckpt)
        print(f"loaded policy ckpt")
    buffer = PPOBufferTensorGPU(env.num_envs, num_steps_per_env, env.num_obs, env.num_privileged_obs, env.num_actions)
    trainer = PPOTrainer(policy, lr=0.0001, weight_decay=0.0001, entropy_loss_weight=0.0)

    initial_iteration = policy.version + 1
    total_iterations = 1500
    log_interval = 5
    
    env_step_result = env.reset()
    
    transition = SampleBatch()

    episode_return_buffer = deque(maxlen=100)
    episode_length_buffer = deque(maxlen=100)
    cur_return = torch.zeros((env.num_envs, 1), dtype=torch.float32, device=device)
    cur_length = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    
    
    pbar = tqdm.tqdm(range(initial_iteration, initial_iteration + total_iterations))
    for epoch in range(initial_iteration, initial_iteration + total_iterations):
        # rollout
        policy.eval_mode()
        for step in range(num_steps_per_env):
            obs, critic_obs = env_step_result.obs['obs'], env_step_result.obs['critic_obs']
            transition.obs = {
                "obs": obs,
                "critic_obs": critic_obs
            }
            transition.policy_state = None

            rollout_result = policy.rollout(RolloutRequest(obs={'obs': obs}))

            transition.action = rollout_result.action
            transition.analyzed_result = rollout_result.analyzed_result
            
            env_step_result = env.step(rollout_result.action)
            
            transition.reward = env_step_result.reward
            transition.done = env_step_result.done
            
            if step == num_steps_per_env - 1:
                transition.truncated = torch.ones(1, dtype=torch.bool)
            
            buffer.put(transition)

            cur_return += env_step_result.reward
            cur_length += 1
            reset_ids = torch.where(env_step_result.done)[0]
            episode_return_buffer.extend(cur_return[reset_ids].cpu().numpy().tolist())
            episode_length_buffer.extend(cur_length[reset_ids].cpu().numpy().tolist())
            cur_return[reset_ids] = 0
            cur_length[reset_ids] = 0
            
        # train
        policy.train_mode()
        samples = buffer.get()
        buffer.clear()
        trainer_step_result = trainer.step(samples)
        policy.inc_version()

        trainer_step_result.stats.update({ "step": trainer_step_result.step })
        # wandb.log(trainer_step_result.stats)
        if epoch % log_interval == 0:
            logging_str = get_logging_str_from_dict(trainer_step_result.stats)
            print(logging_str)
        pbar.update(1)
        pbar.set_description(f"step: {epoch}, mean_return: {torch.mean(torch.tensor(episode_return_buffer)):.4f}, mean_length: {torch.mean(torch.tensor(episode_length_buffer)):.4f}")

    
    policy_ckpt = policy.get_checkpoint()
    torch.save(policy_ckpt, "policy.pth")
            

if __name__ == '__main__':
    main()