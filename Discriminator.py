from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

class Discriminator:
    
    ''' Build Discriminator '''
    def __init__(self):
        
        self.arch = Sequential()
        
        # Fix input_shape later

        self.arch.add(Conv2D(32, 3, padding='same', strides=2, input_shape=(256, 256, 3), data_format='channels_last'))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Conv2D(64, 3, padding='same', strides=2, data_format='channels_last'))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Conv2D(128, 3, padding='same', strides=2, data_format='channels_last'))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Conv2D(256, 3, padding='same', strides=2, data_format='channels_last'))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Flatten())

        # self.arch.add(Dense(128, activation='relu'))
        self.arch.add(Dense(2, activation='sigmoid'))
        
        adam_lr = 0.0002
        adam_beta_1 = 0.5
        
        self.arch.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        
        return

    def __repr__(self):
        return str(self.arch.summary())
    
    def fit_data(self, train_data, train_labels, test_data, test_labels):
        train_history = self.arch.fit(train_data, train_labels,
                                      epochs=5, steps_per_epoch=train_data.shape[0], validation_steps=test_data.shape[0],
                                      validation_data=(test_data, test_labels),
                                      shuffle=True
                                      )
        return train_history


