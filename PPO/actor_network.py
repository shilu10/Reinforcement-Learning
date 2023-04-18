import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Input, Lambda
 
class ActorNetwork(tf.keras.Model):
    def __init__(self, input_didims, action_dim, action_bound):
        super(ActorNetwork, self).__init__()
        self.action_bound = action_bound
        self.fc1 = Dense(64, activation="relu", input_shape=input_dims, kernel_initializer="he_uniform")
        self.fc2 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        self.fc3 = Dense(32, activation="relu", kernel_initializer="he_uniform")
        self.mu = Dense(action_dim, activation='tanh')
        self.std = Dense(action_dim, activation='softplus')
        self.mu_out = Lambda(lambda x: x * self.action_bound)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc3(x)
        mu = self.mu(x)
        mu = self.mu_out(mu)
        
        std = self.std(x)
        return mu, std  
