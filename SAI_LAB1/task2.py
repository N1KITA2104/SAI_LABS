import numpy as np


class Neuron:
    def __init__(self, num_inputs):
        # Ініціалізуємо ваги випадковими значеннями
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    @staticmethod
    def activation_function(x):
        # Використовуємо сигмоїдальну функцію активації
        return 1 / (1 + np.exp(-x))

    def predict(self, input_data):
        # Обчислюємо зважену суму вхідних значень та застосовуємо функцію активації
        z = np.dot(input_data, self.weights) + self.bias
        return self.activation_function(z)

    def train(self, X_el, y_el, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            for i, j in zip(X_el, y_el):
                prediction_el = self.predict(i)
                error = j - prediction_el
                self.weights += learning_rate * error * prediction_el * (1 - prediction_el) * i
                self.bias += learning_rate * error * prediction_el * (1 - prediction_el)
