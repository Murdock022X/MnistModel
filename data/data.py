import keras
import matplotlib.pyplot as plt
import numpy as np

class Data:

    def __init__(self) -> None:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()

        self.x_train: np = self.x_train / 255.0
        self.x_test: np = self.x_test / 255.0

        self.x_train: np = self.x_train.reshape(len(self.x_train), 28, 28, 1)
        self.x_test: np = self.x_test.reshape(len(self.x_test), 28, 28, 1)

    def show_imgs(self):
        fig, axes = plt.subplots(1, 5, figsize=(10, 10))

        for axe, img in zip(axes, zip(self.x_train[:5], self.y_train[:5])):
            axe.imshow(img[0])
            axe.set_title(str(img[1]))

        plt.show()
