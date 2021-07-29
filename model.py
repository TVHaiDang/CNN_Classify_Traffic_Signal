from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

class Model:

    def __init__(self):
        self.model = Sequential()
        self.width = 32
        self.height = 32
        self.classes = 43
        self.shape = (self.width, self.height, 3)
        self.learning_rate = 0.01
        self.epochs = 10
        self.batch_size = 64


    def run_model(self, X_train, Y_train, X_val, Y_val):
        self.model.add(Conv2D(32, (3, 3), padding="same", input_shape=self.shape))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, (3,3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3,3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3,3), padding="same"))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())

        self.model.add(Dense(self.classes))
        self.model.add(Activation("softmax"))

        self.model.summary()

        aug = ImageDataGenerator(rotation_range=0.18, zoom_range=0.15, width_shift_range=0.2, horizontal_flip=True)

        opt = SGD(learning_rate=self.learning_rate, momentum=0.9)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
        complete_model = self.model.fit_generator(aug.flow(X_train, Y_train, batch_size=self.batch_size), validation_data=(X_val, Y_val),
        steps_per_epoch=X_train.shape[0]//self.batch_size, epochs=self.epochs, verbose=1)
        
        return self.model