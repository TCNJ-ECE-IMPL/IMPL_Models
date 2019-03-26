from keras.models import Model, load_model
from keras.applications.mobilenet import MobileNet, preprocess_input

#from ClassificationModel import ClassificationModel

class MobileNetModel:#(ClassificationModel):
    def __init__(self):
        #super(MobileNetModel, self).__init___(model_name='MobileNet')
        self.num_classes = 2
        self.build_model()

        return

    def build_model(self):
        # Initializing the model with random wights
        self.arch = MobileNet(weights=None, input_shape=(256,256,3), classes=self.num_classes)

        # Compiling model with optimization of RSM and cross entropy loss
        self.arch.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return

    def __repr__(self):
        return str(self.arch.summary())

    def fit_data(self, train_images, train_labels, val_images, val_labels, initial_epoch=None):
        train_history = self.arch.fit(train_images, train_labels,
                                      epochs=5, steps_per_epoch=train_images.shape[0], validation_steps=val_images.shape[0],
                                      validation_data=(val_images, val_labels),
                                      shuffle=True
                                      )
        return train_history

    def save_model(self, model_path):
        self.arch.save(model_path)
        return

    def load_model(self, model_path):
        self.arch = load_model(model_path)
        return
