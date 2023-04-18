class CriticNetwork(tf.keras.Model):
    def __init__(self, input_dims, action_dim=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(512, activation="relu", input_shape=input_dims, kernel_initializer="he_uniform")
        self.fc2 = Dense(256, activation="relu", kernel_initializer="he_uniform")
        self.fc2 = Dense(64, activation="relu", kernel_initializer="he_uniform")
        self.fc3 = Dense(1)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x  
