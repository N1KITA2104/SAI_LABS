from task1_1 import *
from task1_2 import *
from task2 import *

print("Task 1.1")
# Приклад використання логічних функцій.
print(f"Результат логічної функції AND: {logical_and(0, 1)}")  # Виведе 0
print(f"Результат логічної функції OR: {logical_or(0, 0)}")  # Виведе 0
print(f"Результат логічної функції NOT: {logical_not(0)}")  # Виведе 1
print(f"Результат логічної функції XOR: {logical_xor(1, 0)}")  # Виведе 1
print()


print("Task 1.2")
# Вхідні дані
data = [2.5, 4.2, 1.6, 4.2, 1.1, 4.4, 0.8, 4.1, 0.0, 4.7, 1.9, 4.1, 0.0, 5.0, 1.4]

# Ініціалізуємо та навчаємо нейрон
neuron = ArtificialNeuron()
neuron.train(data)
print(f"Вагові коефіцієнти після навчання: w1 = {neuron.w1: .8f}, w2 = {neuron.w2: .8f}, w3 = {neuron.w3: .8f}")

# Передбачення для четвертого значення
print("Вхідні дані:", data[-3], data[-2], data[-1])
prediction_first = neuron.predict(data[-3], data[-2], data[-1])
print("Передбачення наступного значення: %.1f" % prediction_first)

print("Вхідні дані:", data[-2], data[-1], "%.1f" % prediction_first)
prediction_second = neuron.predict(data[-2], data[-1], prediction_first)
print("Передбачення наступного значення: %.1f\n" % prediction_second)

print("Task 2")
# Вхідні дані
X = np.array([[0, 0, 0],
              [0, 1, 0],
              [1, 0, 0],
              [1, 1, 1]])

# Очікувані вихідні дані
y = np.array([1, 1, 0, 1])

# Створюємо екземпляр нейрона
neuron = Neuron(num_inputs=3)

# Навчання нейрона
neuron.train(X, y)

# Тестування навченої моделі
for inputs, output in zip(X, y):
    prediction = neuron.predict(inputs)
    print("Input:", inputs, "Predicted Output:", prediction, "Expected Output:", output)
