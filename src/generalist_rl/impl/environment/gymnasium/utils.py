import os
import cv2
import torch
import gymnasium as gym

from generalist_rl.api.policy import Policy, RolloutRequest


def save_video(frames, filename, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)
        
    video.release()
        
def gym_eval_policy(env_name: str, policy: Policy, episode_index: int):
    device = next(iter(policy.net.parameters())).device
    env = gym.make(env_name, autoreset=True, render_mode='rgb_array_list')
    state, info = env.reset()
    while True:
        state = torch.FloatTensor(state).to(device)
        with torch.inference_mode():
            rollout_result = policy.rollout(RolloutRequest(obs={'obs': state}))
        action = rollout_result.action.x.detach().cpu().numpy().flatten()
        if action.dtype == int:
            action = action[0]
        state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            dir = os.path.join(os.path.dirname(__file__), f"videos/{env_name}")
            os.makedirs(dir, exist_ok=True)
            save_video(env.render(), f"{dir}/{env_name}-episode-{episode_index}.mp4", fps=env.metadata["render_fps"])
            break
