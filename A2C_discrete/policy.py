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

"""
    def get_action(self, state, greedy=False):
        _logits = self.model(np.array([state]))
        _probs = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_probs.ravel())
        return tl.rein.choice_action_by_probs(_probs.ravel())  # sam

"""