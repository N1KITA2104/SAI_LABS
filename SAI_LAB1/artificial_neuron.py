from math import exp


class Neuron:
    def __init__(self, weights, threshold=None, sigmoid=False):
        self.weights = weights
        self.threshold = threshold
        self.sigmoid = sigmoid

    def activate(self, inputs):
        weighted_sum = sum([i * w for i, w in zip(inputs, self.weights)])
        if self.sigmoid:
            return 1 / (1 + exp(-weighted_sum))
        else:
            return 1 if weighted_sum >= self.threshold else 0


# Логічна функція NOT
def logical_not(a):
    neuron = Neuron([-1.5], -1)
    return neuron.activate([a])


# Логічна функція AND
def logical_and(a, b):
    neuron = Neuron([1, 1], 1.5)
    return neuron.activate([a, b])


# Логічна функція OR
def logical_or(a, b):
    neuron = Neuron([1, 1], 0.5)
    return neuron.activate([a, b])


# Логічна функція XOR
def logical_xor(a, b):
    return logical_and(logical_or(a, b), logical_not(logical_and(a, b)))
