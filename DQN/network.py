import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D, Input


class DeepQNetwork2D(keras.Model):

    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork2D, self).__init__()
     #   self.fc1 = Dense(64, activation='relu')
        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(n_actions, activation=None)

    def call(self, state):

        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class DeepQNetwork3D(keras.Model): 
    
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork3D, self).__init__()

        self.conv1 = Conv2D(32, 8, strides=(4, 4), activation='relu', data_format="channels_first")
        self.conv2 = Conv2D(32, 4, strides=(2, 2), activation='relu', data_format="channels_first")
        self.conv3 = Conv2D(64, 3, strides=(1, 1), activation='relu', data_format="channels_first")
        self.flatten = Flatten()

        self.fc2 = Dense(128, activation='relu')
        self.fc3 = Dense(n_actions, activation=None)

    def call(self, state):

        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        
        x = self.fc2(x)
        x = self.fc3(x)
        return x