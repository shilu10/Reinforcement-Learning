from npy_append_array import NpyAppendArray
import numpy as np

class Trainer:   
    def __init__(self, env, gamma, alpha, beta, batch_size, 
                                     tau, sot_update, noe, max_steps, 
                                     is_tg, tg_bot_freq_epi, record, 
                                     mem_size, noise): 
       
        self.env = env 
        self.target_score = 80
        self.noe = noe
        self.max_steps = max_steps
        self.is_tg = is_tg
        self.tg_bot_freq_epi = tg_bot_freq_epi
        self.record = record 
        self.writer = Writer("model_training_results.txt")
        self.recorder = RecordVideo("ddpg", "videos/", 20)
        self.agent = DDPGAgent(env.observation_space.shape, 
                               env.action_space.shape[0], gamma, alpha,
                               beta, batch_size, mem_size,
                               soft_update, tau, env.action_space.low[0],
                               env.action_space.high[0], noise
                            )

    def train_rl_model(self): 
        avg_rewards = []
        best_reward = float("-inf")
        episode_rewards = []
        for episode in range(self.noe): 
            n_steps = 0 
            state = self.env.reset()
            reward = 0 
            
            if record and episode % 50 == 0:
                img = self.env.render()
                self.recorder.add_image(img)

            for step in range(self.max_steps): 

                if type(state) == tuple: 
                    state = state[0]
                state = state

                action = self.agent.get_action(state, evaluate=False)

                next_info = self.env.step(action)
                next_state, reward_prob, terminated, truncated, _ = next_info
                done = truncated or terminated
                reward += reward_prob

                self.agent.store_experience(state, action, reward_prob, next_state, done)
                self.agent.learn()

                state = next_state
                n_steps += 1   
                
                # record
                if record and episode % 50 == 0:
                    img = self.env.render()
                    self.recorder.add_image(img)
                
                if done: 
                    break
            
            episode_rewards.append(reward)
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)

            result = f"Episode: {episode}, Steps: {n_steps}, Reward: {reward}, Best reward: {best_reward}, Avg reward: {avg_reward}"
            self.writer.write_to_file(result)
            print(result)
            
            # Recording.
            if record and episode % 50 == 0:
                self.recorder.save(episode)
                
            # Saving Best Model
            if reward > best_reward: 
                best_reward = reward
                self.agent.save_models()
                
            # Telegram bot
            if self.is_tg and episode % self.tg_bot_freq_epi == 0: 
                info_msg(episode+1, self.noe, reward, best_reward, "d")
                
            # Eatly Stopping
            if episode > 100 and np.mean(episode_rewards[-20:]) >= self.target_score: 
                break
                
        return episode_rewards, avg_rewards, best_reward
    