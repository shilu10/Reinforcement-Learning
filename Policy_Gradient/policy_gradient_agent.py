from tensorflow.keras.optimizers import Adam
import numpy as np 
import tensorflow_probability as tfp 

class PolicyGradientAgent: 
  
  def __init__(self, input_dims, out_dims, lr, action_space, gamma, chpkt, algo_name): 
    self.input_dims = input_dims
    self.out_dims = out_dims
    self.lr = lr
    self.action_space = action_space
    self.gamma = gamma
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    self.policy_network = PolicyNetwork2D(self.out_dims)
    self.policy_network.compile(optimizer=Adam(learning_rate=self.lr))
    self.fname = chpkt + algo_name + "_"


  def save_model(self): 
    self.policy_network.save(self.fname)
    print("[+] Saved the model!!")


  def load_model(self):
    self.policy_network = tf.models.load_model(self.fname)
    print("[+] Loaded the mode!!")


  def policy(self, state): 
    state = state.reshape(1, -1)
    state = tf.convert_to_tensor(state)
    action_logits = self.policy_network(state)
    action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
    return action

  def get_action(self, state): 
    action = self.policy(state).numpy()
    return action.squeeze()
  
  def learn(self, rewards, actions, states): 
    discounted_rewards = []
    discounted_reward = 0
    for reward in rewards[::-1]: 
      discounted_reward = reward + self.gamma * discounted_reward 
      discounted_rewards.append(discounted_reward)
    discounted_rewards = discounted_rewards[::-1]

    for discounted_reward, state, action in zip(discounted_rewards, states, actions): 
     # discounted_rewards = tf.convert_to_tensor(discounted_rewards)
    #  states = tf.convert_to_tensor(states)
      #actions = tf.convert_to_tensor(actions)

      with tf.GradientTape() as tape: 
        action_probs = self.policy_network(np.array(state).reshape(1, -1), training=True)
        loss = self.loss(action_probs, action, discounted_reward)

      params = self.policy_network.trainable_variables
      grads = tape.gradient(loss, params)
      self.optimizer.apply_gradients(zip(grads, params)) 


  def loss(self, action_probabilities, actions, rewards): 
    dist = tfp.distributions.Categorical(
            probs=action_probabilities, dtype=tf.float32
        )
    log_prob = dist.log_prob(actions)
    loss = -log_prob * rewards
    return loss 
    