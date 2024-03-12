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

test_input = [[0, 0], [0, 1], [1, 0], [1, 1]]

# Test logical AND
print("Testing Logical AND:")
for i in range(len(test_input)):
    output = net_and.sim([test_input[i]])
    print(f"Input: {test_input[i]}, Output: {output}")

# Test logical OR
print("\nTesting Logical OR:")
for i in range(len(test_input)):
    output = net_or.sim([test_input[i]])
    print(f"Input: {test_input[i]}, Output: {output}")

# Plot results
plot(error_and, label='AND Gate')
plot(error_or, label='OR Gate')
xlabel('Epoch number')
ylabel('Train error')
legend()
grid()
show()
