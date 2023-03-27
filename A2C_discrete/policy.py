import tensorflow as tf 
import numpy as np 
import random
import tensorflow_probability as tfp

def get_action(state):
    state = tf.convert_to_tensor(np.array(state).reshape(1, -1))
    action_probs = self.actor_network(state)[0]
    action_probs = tfp.distributions.Categorical(probs=action_probs)
    action = action_probs.sample()
    self.action = action
    return action.numpy()