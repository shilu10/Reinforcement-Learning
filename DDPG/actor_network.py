import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Input, Lambda, concatenate
 
class ActorNetwork(tf.keras.Model):
    def __init__(self, input_dims, action_bound, action_dims, name):
        super(ActorNetwork, self).__init__()
        self.model_name = name
        self.fc1 = Dense(64, activation="relu", input_shape=input_dims, kernel_initializer="he_uniform")
        self.fc2 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.fc3 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        
        self.out = Dense(action_dims, activation='tanh')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x 
