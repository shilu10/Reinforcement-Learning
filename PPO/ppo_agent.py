from tensorflow.keras.optimizers import Adam

class PPOAgent:
    def __init__(self, input_dims, out_dims, gamma, alpha, beta, action_bound, std_bound): 
        self.input_dims = input_dims 
        self.out_dims = out_dims 
        self.gamma = gamma
        self.alpha = alpha 
        self.beta = beta 
        self.action_bound = action_bound 
        self.std_bound = std_bound
        self.clip_ratio = 0.2
        
        self.actor_network = ActorNetwork(self.input_dims, self.out_dims, self.action_bound)
        self.critic_network = CriticNetwork(self.input_dims, 1)
        
        self.actor_network.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic_network.compile(optimizer=Adam(learning_rate=self.beta))
        
    def get_action(self, state):
        state = np.reshape(state, [1, self.input_dims[0]])
        mu, std = self.actor_network(state)
        action = np.random.normal(mu[0], std[0], size=self.out_dims)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action
    
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
    
    def save_models(self): 
        self.actor_network.save("models/" + "ppo_actor")
        self.critic_network.save("models/" + "ppo_critic")
        print("Saved the models successfully")
        
    def load_models(self):
        self.actor_network = tf.keras.models.load_model("models/" + "actor")
        self.critic_network = tf.keras.models.load_model("models/" + "critic")
        print("Loaded the models successfully")
        
    def actor_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)
    
    def critic_loss(self, q_val, target_q_val): 
        mse = tf.keras.losses.MeanSquaredError()
        return mse(target_q_val, q_val)
    
    def learn(self, states, actions, gaes, log_old_policy, td_targets):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        gaes = tf.convert_to_tensor(gaes, dtype=tf.float32)
        log_old_policy = tf.convert_to_tensor(log_old_policy, dtype=tf.float32)
        td_targets = tf.convert_to_tensor(td_targets, dtype=tf.float32)
        with tf.GradientTape() as tape:
            mu, std = self.actor_network(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            actor_loss = self.actor_loss(
                log_old_policy, log_new_policy, actions, gaes)
        
        actor_params = self.actor_network.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        self.actor_network.optimizer.apply_gradients(zip(actor_grads, actor_params))
        
        with tf.GradientTape() as tape:
            v_pred = self.critic_network(states, training=True)
            assert v_pred.shape == td_targets.shape
            critic_loss = self.critic_loss(v_pred, tf.stop_gradient(td_targets))
        
        critic_params = self.critic_network.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_params)
        self.critic_network.optimizer.apply_gradients(zip(critic_grads, critic_params))
