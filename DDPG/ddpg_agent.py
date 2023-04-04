from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np

class DDPGAgent:
  
    def __init__(self, input_dims, n_actions, gamma, alpha, beta, 
                                batch_size, mem_size, soft_update, 
                                tau, min_action, max_action, noise): 
        self.gamma = gamma 
        self.noise = noise
        self.n_actions = n_actions
        self.soft_update = soft_update
        self.tau = tau
        self.fname = "models/ddpg/"
        self.min_action = min_action
        self.max_action = max_action
        self.batch_size = batch_size

        self.memory = ExperienceReplayBuffer(mem_size, input_dims, batch_size, n_actions, cer=False)
        self.actor = ActorNetwork(input_dims, max_action, n_actions, "actor")
        self.target_actor = ActorNetwork(input_dims, max_action, n_actions, "target_actor")
        self.critic = CriticNetwork(input_dims, 1, "critic")
        self.target_critic = CriticNetwork(input_dims, 1, "target_critic_")
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))
        self.bg_noise = np.zeros(n_actions)

        self.update_target_networks()
        
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
        
    def get_action(self, state, evaluate): 
        # adding noise, makes us to do the exploration,
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        noise = self.ou_noise(self.bg_noise, dim=self.n_actions) 
        if not evaluate:
            actions = actions + noise

        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        self.bg_noise = noise
        return actions[0]

    def store_experience(self, state, action, reward, state_, done):
        self.memory.store_experience(state, action, reward, state_, done)

    def sample_experience(self):
        state, action, reward, new_state, done = \
                                  self.memory.sample_experience(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones
 
    def save_models(self):
        self.actor.save(self.fname + "ddpg_actor_network")
        self.target_actor.save(self.fname + "ddpg_target_actor_network")
        self.critic.save(self.fname  + "ddpg_critic_network")
        self.target_critic.save(self.fname  + "ddpg_target_critic_network")
        print("[+] Saving the models") 

    def load_models(self):
        self.actor = tf.keras.models.load_model(self.fname + "_" + "ddpg_actor_network") 
        self.target_actor = tf.keras.models.load_model(self.fname + "_" + "ddpg_target-actor_network") 
        self.critic = tf.keras.models.load_model(self.fname + "_" + "ddpg_critic_network") 
        self.target_critic = tf.keras.models.load_model(self.fname + "_" + "ddpg_target_critic_network") 
        print("[+] Loading the models")
  
    def learn(self): 
        if not self.memory.is_sufficient():
            return
        states, actions, rewards, next_states, dones = self.sample_experience()
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            one_step_lookahead_vals = tf.squeeze(self.target_critic(next_states, target_actions), 1)
            q_vals = tf.squeeze(self.critic(states, actions), 1)
            target_q_vals = rewards + self.gamma * one_step_lookahead_vals * ([1 - int(d) for d in dones])
            critic_loss = self.critic_loss(q_vals, target_q_vals)
      
        critic_params = self.critic.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_params)
        self.critic.optimizer.apply_gradients(zip(critic_grads, critic_params))

        with tf.GradientTape() as tape:
            pred_actions = self.actor(states)
            q_vals = -self.critic(states, pred_actions)
            actor_loss = self.actor_loss(q_vals)

        actor_params = self.actor.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        self.actor.optimizer.apply_gradients(zip(actor_grads, actor_params))

        self.update_target_networks()
    
    def critic_loss(self, q_vals, target_q_vals): 
        loss = tf.keras.losses.MSE(q_vals, target_q_vals)
        return loss
    
    def actor_loss(self, q_vals):
        return tf.math.reduce_mean(q_vals)

    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        t_actor_weights = self.target_actor.get_weights()
        critic_weights = self.critic.get_weights()
        t_critic_weights = self.target_critic.get_weights()
        if self.soft_update: 
            for i in range(len(actor_weights)):
                t_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * t_actor_weights[i]

            for i in range(len(critic_weights)):
                t_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * t_critic_weights[i]

            self.target_actor.set_weights(t_actor_weights)
            self.target_critic.set_weights(t_critic_weights)
            
        else: 
            self.target_actor.set_weights(t_actor_weights)
            self.target_critic.set_weights(t_critic_weights)
  