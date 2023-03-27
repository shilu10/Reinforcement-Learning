from actor_network import * 
from critic_network import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import tensorflow_probability as tfp
import numpy as np

class ActorCriticAgent:
  
  def __init__(self, input_dims, out_dims, gamma, lr1, lr2, action_space, batch_size, chkpt, algo_name): 
    self.gamma = gamma 
    self.batch_size = batch_size 
    self.action_space = action_space 
    self.action = None  
    self.fname = chkpt + '_' + algo_name 
    self.actor_network = ActorNetwork(input_dims, out_dims)
    self.actor_network.compile(optimizer=Adam(learning_rate=lr1))
    self.critic_network = CriticNetwork(input_dims, out_dims)
    self.critic_network.compile(optimizer=Adam(learning_rate=lr2))


  def get_action(self, state): 
    state = tf.convert_to_tensor(np.array(state).reshape(1, -1))
    action_probs = self.actor_network(state)[0]
    action_probs = tfp.distributions.Categorical(probs=action_probs)
    action = action_probs.sample()
    self.action = action
    return action.numpy()
 

  def save_models(self):
    self.actor_network.save(self.fname + "_" + "actor_network")
    self.critic_network.save(self.fname + "_" + "critic_network")
    print("[+] Saving the model")


  def load_models(self):
    self.actor_network = tf.keras.models.load_model(self.fname + "_" + "actor_network") 
    self.critic_network = tf.keras.models.load_model(self.fname + "_" + "critic_network") 
    print("[+] Loading the model")
  
  
  def learn(self, state, action, reward, next_state, done): 
    
    state = tf.convert_to_tensor(np.array(state).reshape(1, -1))
    next_state = tf.convert_to_tensor(np.array(next_state).reshape(1, -1))    
    td_target = None
    
    with tf.GradientTape() as tape: 
      value = self.critic_network(state)
      next_state_value = self.critic_network(next_state, training=True)
      critic_loss, td = self.critic_loss(value, reward, next_state_value, self.gamma, done)

    td_target = td
    critic_params = self.critic_network.trainable_variables 
    critic_grads = tape.gradient(critic_loss, critic_params)
    self.critic_network.optimizer.apply_gradients(zip(critic_grads, critic_params))

    with tf.GradientTape() as tape: 
      action_probs = self.actor_network(state)
      actor_loss = self.actor_loss(action_probs, action, td_target)

    actor_params = self.actor_network.trainable_variables
    actor_grads = tape.gradient(actor_loss, actor_params)

    self.actor_network.optimizer.apply_gradients(zip(actor_grads, actor_params))  
  

  def actor_loss(self, action_probs, action, td): 
    dist = tfp.distributions.Categorical(probs=action_probs, dtype=tf.float32)
    log_prob = dist.log_prob(action)
    loss = -log_prob * td
    return loss
  

  def critic_loss(self, value, reward, next_state_value, gamma, done): 
    td_target = reward + gamma * next_state_value * (1-int(done))
    delta = (td_target - value) ** 2 
    return delta, td_target
  