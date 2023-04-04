import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Input, Lambda, concatenate


class CriticNetwork(tf.keras.Model):
    def __init__(self, input_dims, action_dims, name): 
        super(CriticNetwork, self).__init__()
        self.model_name = name
        self.fc1 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        self.fc2 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.fc3 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.out = Dense(1, activation='linear')

    def call(self, state, action):
        x = self.fc1(tf.concat([state, action], axis=1))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x 