import numpy as np


class NeuralNetwork:
    def __init__(self):
        # Ініціалізація ваг та зміщення
        np.random.seed(1)
        self.weights = 2 * np.random.random((2, 1)) - 1
        self.bias = 2 * np.random.random(1) - 1

    # Функція активації (сигмоїда)
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Похідна функції активації
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    # Навчання нейронної мережі
    def train(self, X, y, learning_rate=0.1, epochs=10000):
        for epoch in range(epochs):
            # Пряме поширення (forward propagation)
            inputs = X
            outputs = self.sigmoid(np.dot(inputs, self.weights) + self.bias)

            # Обчислення помилки
            error = y.reshape(-1, 1) - outputs

            # Зворотнє поширення (backpropagation)
            adjustments = error * self.sigmoid_derivative(outputs)
            self.weights += np.dot(inputs.T, adjustments) * learning_rate
            self.bias += np.sum(adjustments) * learning_rate

    # Передбачення
    def predict(self, X):
        return np.round(self.sigmoid(np.dot(X, self.weights) + self.bias)).astype(int)
