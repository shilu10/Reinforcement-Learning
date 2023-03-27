import gymnasium as gym 
from utils import * 
from actor_critic_agent import *

class Trainer: 
  def __init__(self, env, action_space, input_dims, out_dims, video_prefix, is_tg, noe, max_steps, record, lr1, lr2, gamma, chkpt, algo_name): 
    self.env = env
    self.noe = noe 
    self.max_steps = max_steps 

    self.recorder = RecordVideo(video_prefix)
    self.is_tg = is_tg 
    self.record = record
    self.agent = ActorCriticAgent(input_dims, out_dims, gamma, lr1, lr2, action_space, 32, chkpt, algo_name)
    
  def train(self): 

    ep_rewards = []
    avg_rewards = []
    best_reward = float("-inf")

    for episode in range(self.noe): 
      state = self.env.reset()
      rewards = 0 

      if self.record and episode % 50 == 0: 
        img = self.env.render()
        self.recorder.add_image(img)

      for step in range(self.max_steps):
        
        if type(state) == tuple: 
          state = state[0]
       
        action = self.agent.get_action(state)

        next_info = self.env.step(action)

        next_state, reward_prob, terminated, truncated, _ = next_info 
        done = terminated or truncated 
        rewards += reward_prob

        self.agent.learn(state, action, reward_prob, next_state, done)

        state = next_state

        if self.record and episode % 50 == 0:
          img = self.env.render()
          self.recorder.add_image(img)

        if done: 
          break 
        
      if self.record and episode % 50 == 0:
        self.recorder.save(episode)

      if rewards > best_reward: 
        self.agent.save_models()
        best_reward = rewards

      ep_rewards.append(rewards)
      avg_reward = np.mean(ep_rewards[-100:])
      avg_rewards.append(avg_reward)
      print(f"Episode: {episode} Reward: {rewards} Best Score: {best_reward}, Average Reward: {avg_reward}")

    return ep_rewards, avg_rewards

