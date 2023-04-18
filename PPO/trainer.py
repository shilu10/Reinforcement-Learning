import gymnasium as gym 


class Trainer: 
    def __init__(self, env, input_dims, out_dims, video_prefix, is_tg, 
                         noe, max_steps, record, alpha, beta, gamma, action_bound, 
                         std_bound, agent_update_interval, lmbda): 
        self.env = env
        self.noe = noe 
        self.max_steps = max_steps 
        self.input_dims = input_dims 
        self.out_dims = out_dims
        self.gamma = gamma 
        self.lmbda = lmbda
        self.agent_update_interval = agent_update_interval
        self.target_val = 190

        self.recorder = RecordVideos(video_prefix)
        self.is_tg = is_tg 
        self.record = record
        self.agent = PPOAgent(input_dims, out_dims, gamma, alpha, beta, action_bound, std_bound)
        
    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.gamma * forward_val - v_values[k]
            gae_cumulative = self.gamma * self.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets
    
    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self): 

        ep_rewards = []
        avg_rewards = []
        best_reward = float("-inf")
        

        for episode in range(self.noe): 
            episodic_rewards = 0
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []
            
            state, _ = self.env.reset()
            rewards = 0 

            if self.record and episode % 50 == 0: 
                img = self.env.render()
                self.recorder.add_image(img)

            for step in range(self.max_steps):
                log_old_policy, action = self.agent.get_action(state)
                next_info = self.env.step(action)

                next_state, reward_prob, terminated, truncated, _ = next_info 
                done = terminated or truncated 
                episodic_rewards += reward_prob
                
                state = np.reshape(state, [1, self.input_dims[0]])
                action = np.reshape(action, [1, self.out_dims])
                next_state = np.reshape(next_state, [1, self.input_dims[0]])
                reward = np.reshape(reward_prob, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])
                
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)
                old_policy_batch.append(log_old_policy)
                
                if len(state_batch) >= self.agent_update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.agent.critic_network(states)
                    next_v_value = self.agent.critic_network(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)
                    
                #    print(gaes, 'gaes')
                    for _ in range(3): 
                        self.agent.learn(states, actions, gaes, old_policys, td_targets)
                    state_batch = []
                    action_batch = []
                    old_policy_batch = []
                    reward_batch = []

                state = next_state

                if self.record and episode % 50 == 0:
                    img = self.env.render()
                    self.recorder.add_image(img)

                if done: 
                    break 

            if self.record and episode % 50 == 0:
                self.recorder.save(episode)

            ep_rewards.append(episodic_rewards)
            avg_reward = np.mean(ep_rewards[-100:])
            avg_rewards.append(avg_reward)
            print(f"Episode: {episode} Reward: {episodic_rewards} Best Score: {best_reward}, Average Reward: {avg_reward}")
            
            if self.target_val <= avg_reward:
                break
            
            if episodic_rewards > best_reward: 
                self.agent.save_models()
                best_reward = episodic_rewards

        return ep_rewards, avg_rewards

