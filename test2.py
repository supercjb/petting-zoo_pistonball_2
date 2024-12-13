import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import cv2
import pygame
from pettingzoo.butterfly import pistonball_v6
from test1 import Agent, batchify_obs, unbatchify
from matplotlib import rcParams



# 新增 EmergenceBehaviorVisu alizer 類
class EmergenceBehaviorVisualizer:
    def __init__(self, n_pistons):
        self.n_pistons = n_pistons
        self.piston_actions = []
        self.ball_positions = []

    def record_step(self, piston_actions, ball1_pos, ball2_pos):
        self.piston_actions.append(piston_actions)
        self.ball_positions.append((ball1_pos, ball2_pos))

    def visualize(self):
        piston_actions = np.array(self.piston_actions)
        ball_positions = np.array(self.ball_positions)

        plt.figure(figsize=(12, 8))

        # 繪製活塞動作
        plt.subplot(2, 1, 1)
        plt.imshow(piston_actions.T, aspect='auto', cmap='coolwarm')
        plt.title('piston actions over time')
        plt.xlabel('time step')
        plt.ylabel('piston index')
        plt.colorbar(label='action')

        # 繪製球的位置
        plt.subplot(2, 1, 2)
        plt.plot(ball_positions[:, 0, 0], label='ball 1 X-position')
        plt.plot(ball_positions[:, 1, 0], label='ball 2 X-position')
        plt.title('piston actions over time')
        plt.xlabel('time step')
        plt.ylabel('X-position')
        plt.legend()

        plt.tight_layout()
        plt.show()


class VideoRecorder:
    def __init__(self, filename, fps=30.0):
        self.filename = filename
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        """從 Pygame surface 獲取畫面並儲存"""
        if isinstance(frame, np.ndarray):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.frames.append(frame)
            cv2.imshow('Pistonball', frame)
            cv2.waitKey(1)

    def save(self):
        """將儲存的幀保存為影片檔案"""
        if not self.frames:
            print("沒有幀可以保存")
            return

        # 獲取第一幀的尺寸
        height, width = self.frames[0].shape[:2]

        # 創建影片寫入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))

        # 寫入每一幀
        for frame in self.frames:
            out.write(frame)

        # 釋放資源
        out.release()
        print(f"影片已保存至 {self.filename}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

""" RENDER THE POLICY """
#env = pistonball_v6.parallel_env(render_mode="human", continuous=False, max_cycles=400)
env = pistonball_v6.parallel_env(render_mode="rgb_array", continuous=False, max_cycles=400)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=4)
num_actions = env.action_space(env.possible_agents[0]).n

agent = Agent(num_actions=num_actions)

# 載入儲存的模型參數
agent.load_state_dict(torch.load("agent_2000.pth"))

agent.to(device)
# 將模型設置為評估模式
agent.eval()

# 初始化可視化器
visualizer = EmergenceBehaviorVisualizer(n_pistons=len(env.possible_agents))
video_recorder = VideoRecorder("pistonball_episode.mp4", fps=10.0)

with torch.no_grad():
    # render 10 episodes out
    for episode in range(1):
        obs, infos = env.reset(seed=None)
        obs = batchify_obs(obs, device)
        terms = [False]
        truncs = [False]
        while not any(terms) and not any(truncs):
            actions, logprobs, _, values = agent.get_action_and_value(obs)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))

            # 記錄數據用於可視化
            piston_actions = actions.cpu().numpy()
            ball1_pos = env.unwrapped.ball1.position
            ball2_pos = env.unwrapped.ball2.position
            visualizer.record_step(piston_actions, ball1_pos, ball2_pos)

            # 記錄影片幀
            # 獲取 pygame 視窗的 surface
            frame = env.render()
            if frame is not None:
                video_recorder.add_frame(frame)

            obs = batchify_obs(obs, device)
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]

# 保存影片
video_recorder.save()

# 生成可視化
visualizer.visualize()