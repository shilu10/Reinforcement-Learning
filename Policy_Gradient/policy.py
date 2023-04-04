import numpy as np 
import tensorflow as tf  

def policy(state, policy_network): 
    state = state.reshape(1, -1)
    state = tf.convert_to_tensor(state)
    action_logits = policy_network(state)
    action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
    return action

def get_action(state, policy_network): 
    action = policy(state, policy_network).numpy()
    return action.squeeze()