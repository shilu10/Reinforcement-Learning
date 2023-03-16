import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Input
 
class PolicyNetwork2D(tf.keras.Model):
    def __init__(self, action_dim=1):
        super(PolicyNetwork2D, self).__init__()
        self.fc1 = Dense(24, activation="relu")
        self.fc2 = Dense(36, activation="relu")
        self.fc3 = Dense(action_dim, activation="softmax")

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
