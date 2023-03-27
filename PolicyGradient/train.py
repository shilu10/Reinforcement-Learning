import numpy as np 
import random 
import tensorflow as tf 
from policy_gradient_agent import *

class Trainer: 
  def __init__(self, env, action_space, input_dims, out_dims, 
                          video_prefix, is_tg, noe, max_steps, record, 
                          lr, gamma, chkpt, algo_name
                        ): 
    self.env = env
    self.noe = noe 
    self.max_steps = max_steps 

    self.recorder = RecordVideo(video_prefix)
    self.is_tg = is_tg 
    self.record = record
    self.agent = PolicyGradientAgent(input_dims, out_dims, lr, action_space,
                                                         gamma, chkpt, algo_name)
    
  def train(self): 

    ep_rewards = []
    avg_rewards = []
    best_reward = float("-inf")

    for episode in range(self.noe): 
      state = self.env.reset()
      rewards = []
      actions = []
      states = []

      if self.record and episode % 50 == 0: 
        img = self.env.render()
        self.recorder.add_image(img)

      for step in range(self.max_steps):
        
        if type(state) == tuple: 
          state = state[0]
        states.append(state) 
        action = self.agent.get_action(state)

        next_info = self.env.step(action)

        next_state, reward_prob, terminated, truncated, _ = next_info 
        done = terminated or truncated 

        rewards.append(reward_prob)
        actions.append(action)

        state = next_state

        if record and episode % 50 == 0:
          img = self.env.render()
          self.recorder.add_image(img)

        if done: 
          l = self.agent.learn(rewards, actions, states)
          break 
        
      if self.record and episode % 50 == 0:
        self.recorder.save(episode)

      if sum(rewards) > best_reward: 
        self.agent.save_model()
        best_reward = sum(rewards)

      ep_rewards.append(sum(rewards))
      avg_reward = np.mean(ep_rewards[-100:])
      avg_rewards.append(avg_reward)
      print(f"Episode: {episode} Reward: {sum(rewards)} Best Score: {best_reward}, Average Reward: {avg_reward}")

    return ep_rewards, avg_rewards

