from neurolab import *
from pylab import *

"""
    Single Layer Perceptron
"""

input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Logical and
target_and = [[0], [0], [0], [1]]
net_and = net.newp([[0, 1], [0, 1]], 1)  # Create net with 2 inputs and 1 neuron
error_and = net_and.train(input_data, target_and, epochs=100, show=10, lr=0.1)  # train with delta rule

# Logical or
target_or = [[0], [1], [1], [1]]
net_or = net.newp([[0, 1], [0, 1]], 1)  # Create net with 2 inputs and 1 neuron
error_or = net_or.train(input_data, target_or, epochs=100, show=10, lr=0.1)  # train with delta rule

# Plot results
plot(error_and, label='AND Gate')
plot(error_or, label='OR Gate')
xlabel('Epoch number')
ylabel('Train error')
legend()
grid()
show()

