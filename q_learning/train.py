import gym
import numpy as np 
from policy import *
from video_recorder import *

recorder = RecordVideo("q_learning")

def initialize_q_table(nos_space, noa_space): 
    q_table = np.zeros((nos_space, noa_space))
    return q_table


def check_2d_array(observation): 
    if np.array(observation, dtype="object").shape == (2, ): 
        return True 
    return False 


def train_model(env, nos_space, noa_space, noe, epsilon, 
                    max_epsilon, alpha, gamma, eps_decay_rate=0.005, 
                    min_epsilon=0.05, max_steps=100): 
    action_space = [_ for _ in range(noa_space)] 
    q_table = initialize_q_table(nos_space, noa_space)

    episode_rewards = []
    avg_rewards = []
    epsilon_history = []
    best_score = float("-inf")

    for episode in range(noe): 
        step = 0
        state = env.reset()
        done = False 
        tot_rew = 0    
        
        if episode % 50 == 0:
            img = env.render()
            recorder.add_image(ing)

        if check_2d_array(state): 
            state = state[0]

        action = epsilon_greedy_policy(state, 
                                action_space, 
                                epsilon,
                                q_table,
                                env
                            )  

        while not done: 
            next_state_info = env.step(action)
            next_state, reward_prob, done, info, _ = next_state_info

            next_action = epsilon_greedy_policy(next_state,
                                            action_space,
                                            epsilon,
                                            q_table,
                                            env
                                    )
            
            old_q_val = q_table[state][action]
            td_target = reward_prob + (gamma * np.max(q_table[next_state]))
            td_error = alpha * (td_target - old_q_val)
            new_q_val = old_q_val + td_error
            q_table[state][action] = new_q_val

            state = next_state
            action = next_action
            tot_rew += reward_prob
            step += 1 

            if episode % 50 == 0:
                img = env.render()
                recorder.add_image(ing)

            if done or step >= max_steps: 
                break
        
        episode_rewards.append(tot_rew)
        epsilon_history.append(epsilon)
        avg_reward = np,mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-eps_decay_rate*episode) 

        if tot_rew > best_score:
            best_score = tot_rew

        if episode % 50 == 0:
            recorder.save(episode)

        print(f"[+] Episode: {episode}, reward: {tot_rew} Epsilon: {epsilon} Best Score {best_score} Avg Reward: {avg_reward}")
    

    return q_table, episode_rewards, epsilon_history, avg_rewards


