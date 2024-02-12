from artificial_neuron import *
from time_series_predictor import *
from neural_network import *

# Приклад використання логічних функцій.
print(f"Результат логічної функції AND: {logical_and(1, 0)}")  # Виведе 0
print(f"Результат логічної функції OR: {logical_or(0, 0)}")  # Виведе 0
print(f"Результат логічної функції NOT: {logical_not(0)}")  # Виведе 1
print(f"Результат логічної функції XOR: {logical_xor(1, 0)}")  # Виведе 1
print()

# Приклад використання
data = [2.5, 4.2, 1.6, 4.2, 1.1, 4.4, 0.8, 4.1, 0.0, 4.7, 1.9, 4.1, 0.0, 5.0, 1.4]
predictor = TimeSeriesPredictor(data)
print(f"Передбачене наступне значення: {predictor.predict_next()}")  # Передбачене наступне значення
print()

# Ініціалізація нейронної мережі
nn = NeuralNetwork()

# Вхідні дані та вихідні дані з таблиці істинності
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([1, 1, 0, 1])

# Навчання нейронної мережі
nn.train(X, y)

# Передбачення та виведення результатів
print("Ваги після навчання:")
print(*nn.weights)
print("Зміщення після навчання:")
print(*nn.bias)
print("Результати після навчання (округлені до цілих чисел):")
print(*nn.predict(X))
