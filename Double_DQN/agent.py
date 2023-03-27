import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, RMSprop

#https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DQN/tf2/agent.py
class Agent: 
    def __init__(self, agent_params):
        # Parameters
        self.gamma = agent_params.get("gamma")
        self.lr = agent_params.get("lr")
        self.input_dims = agent_params.get("input_dims")
        self.batch_size = agent_params.get("batch_size")
        self.replace_target_weight_counter = agent_params.get("replace")
        self.algo = agent_params.get("algo")
        self.env_name = agent_params.get("env_name")
        self.chkpt_dir = agent_params.get("chkpt_dir")
        self.n_actions = agent_params.get("n_actions")
        self.action_space = agent_params.get('actions')
        
        self.eps = agent_params.get("eps")
        self.min_eps = agent_params.get("min_eps")
        self.eps_decay_rate = agent_params.get("eps_decay_rate")
        
        self.learn_step_counter = 0
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'
        self.mem_size = agent_params.get("mem_size")
        
        self.TAU = agent_params.get("tau")
        self.soft_update = agent_params.get("soft_update")

        # networks and replaybuffer
        self.memory = ExperienceReplayBuffer(self.mem_size, self.input_dims, self.n_actions)
        self.q_value_network = DeepQNetwork2D(self.input_dims, self.n_actions) if len(self.input_dims) < 3 else \
                                                        DeepQNetwork3D(self.input_dims, self.n_actions)
        self.q_value_network.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))
        self.target_q_network = DeepQNetwork2D(self.input_dims, self.n_actions) if len(self.input_dims) < 3 else \
                                                        DeepQNetwork3D(self.input_dims, self.n_actions)
        self.target_q_network.compile(optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01))

    def save_models(self):
        self.q_value_network.save(self.fname+'q_value')
        self.target_q_network.save(self.fname+'target_q')
        print('... models saved successfully ...')

    def load_models(self):
        self.q_value_network = keras.models.load_model(self.fname+'q_value')
        self.target_q_network = keras.models.load_model(self.fname+'target_q')
        print('... models loaded successfully ...')

    def store_experience(self, state, action, reward, state_, done):
        self.memory.store_experience(state, action, reward, state_, done)

    def sample_experience(self):
        state, action, reward, new_state, done = \
                                  self.memory.sample_experience(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    def choose_action(self, observation):
        if np.random.random() > self.eps:
            state = tf.convert_to_tensor([observation])
            actions = self.q_value_network(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        if not self.soft_update:
            self.target_q_network.set_weights(self.q_value_network.get_weights())
            return
        else: 
            q_model_theta = self.q_value_network.get_weights()
            target_model_theta = self.target_q_network.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_q_network.set_weights(target_model_theta)
    
    def decrement_epsilon(self): 
        self.eps -= self.eps_decay_rate
        self.eps = max(self.eps, self.min_eps)

    def learn(self):

        if self.memory.mem_counter < self.batch_size:
            return

        self.replace_target_network()
        
        states, actions, rewards, states_, dones = self.sample_experience()

        with tf.GradientTape() as tape:
        
            q_pred = self.q_value_network(states, verbose=False)
            q_next_val = self.q_value_network(states_, verbose=False)
            
            q_pred, q_next_val = q_pred.numpy(), q_next_val.numpy()
            
            q_target = q_pred.copy()

            q_eval = self.target_q_network(states_, verbose=False)
            max_action_ids = np.argmax(q_next_val, axis=1)

            q_target[[np.arange(self.batch_size)], max_action_ids] = rewards[np.arange(self.batch_size)] + self.gamma * q_eval[[np.arange(self.batch_size)], max_action_ids]  * (1 - dones[np.arange(self.batch_size)])
            
            loss = keras.losses.MSE(tf.to_tensor(q_pred), tf.to_tensor(q_target))
        
        params = self.q_value_network.trainable_variables
        grads = tape.gradient(loss, params)
        self.q_value_network.optimizer.apply_gradients(zip(grads, params))
        
        self.learn_step_counter += 1
        self.decrement_epsilon()       