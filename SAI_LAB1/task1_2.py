import math


class ArtificialNeuron:
    def __init__(self, learning_rate=0.01):
        self.w1 = 0.1  # вагові коефіцієнти
        self.w2 = 0.1
        self.w3 = 0.1
        self.learning_rate = learning_rate

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def predict(self, x1, x2, x3):
        Si = x1 * self.w1 + x2 * self.w2 + x3 * self.w3
        Yi = self.sigmoid(Si) * 10
        return Yi

    def train(self, train_data, epochs=1000, tolerance=0.0001):
        prev_error = float('inf')
        for epoch in range(epochs):
            total_error = 0
            for i in range(3, len(train_data)):
                x1, x2, x3 = train_data[i - 3], train_data[i - 2], train_data[i - 1]
                yi = train_data[i]

                # Прогнозуємо значення
                Si = x1 * self.w1 + x2 * self.w2 + x3 * self.w3
                predicted_value = self.sigmoid(Si) * 10

                # Обчислюємо помилку
                error = (predicted_value - yi) ** 2
                total_error += error

                # Обчислюємо похідні
                derivative = (predicted_value - yi) * (math.exp(-Si) / (1 + math.exp(-Si)) ** 2)

                # Обчислюємо виправлення вагових коефіцієнтів
                delta_w1 = derivative * x1
                delta_w2 = derivative * x2
                delta_w3 = derivative * x3

                # Коригуємо вагові коефіцієнти
                self.w1 -= self.learning_rate * delta_w1
                self.w2 -= self.learning_rate * delta_w2
                self.w3 -= self.learning_rate * delta_w3

            # Зупиняємо навчання, якщо досягнуто необхідної точності
            if epoch > 0 and abs(total_error - prev_error) <= tolerance:
                print(f"Навчання завершено на епосі {epoch}. Загальна помилка: {total_error}")
                break

            prev_error = total_error
