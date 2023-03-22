import numpy as np 

class ExperienceReplayBuffer: 
    def __init__(self, max_memory, input_shape, batch_size, cer=False): 
        self.mem_size = max_memory
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.batch_size = batch_size
        self.cer = cer

    def store_experience(self, state, action, reward, next_state, done): 
        index = self.mem_counter % self.mem_size 

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_counter += 1

    def sample_experience(self, batch_size):
        # used to get the last transition
        offset = 1 if self.cer else 0

        max_mem = min(self.mem_counter, self.mem_size) - offset
        batch_index = np.random.choice(max_mem, batch_size - offset, replace=False)

        states = self.state_memory[batch_index]
        next_states = self.next_state_memory[batch_index]
        rewards = self.reward_memory[batch_index]
        actions = self.action_memory[batch_index]
        terminals = self.terminal_memory[batch_index]

        if self.cer: 
            last_index = self.mem_counter % self.mem_size - 1
            last_state = self.state_memory[last_index]
            last_action = self.action_memory[last_index]
            last_terminal = self.terminal_memory[last_index]
            last_next_state = self.next_state_memory[last_index]
            last_reward = self.reward_memory[last_index]

            # for 2d and 3d use vstack to append, for 1d array use append() to append the data
            states = np.vstack((self.state_memory[batch_index], last_state))
            next_states = np.vstack((self.next_state_memory[batch_index], last_next_state))

            actions = np.append(actions, last_action)
            terminals = np.append(terminals, last_terminal)
            rewards = np.append(rewards, last_reward)
    
        return states, actions, rewards, next_states, terminals
    
    
    def is_sufficient(self): 
        return self.mem_counter > self.batch_size
        