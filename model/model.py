import tensorflow as tf
import keras

from data.data import Data

class Model:

    def __init__(self):

        self.model = tf.keras.models.Sequential(
            [
                keras.layers.Conv2D(filters=28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(filters=56, kernel_size=(3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(56, activation='relu'),
                keras.layers.Dense(10),
            ]
        )

        self.model.compile(
            optimizer='Adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.data = self.load_data()

    def load_data(self):
        return Data()

    def train(self):
        self.model.fit(x=self.data.x_train, y=self.data.y_train, epochs=10, validation_data=(self.data.x_test, self.data.y_test))

    def test(self):
        self.model.evaluate(x=self.data.x_test, y=self.data.y_test)

    def save(self):
        keras.saving.save_model(self.model, './model.keras')

    def load(self):
        self.model = keras.saving.load_model('./model.keras')
