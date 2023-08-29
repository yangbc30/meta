import numpy as np
import torch
import matplotlib.pyplot as plt

# Define values from -pi to pi
x_values = np.linspace(-np.pi, np.pi, 10000)
p_values = x_values  # Assuming p(x) = x for demonstration, adjust this as needed

x = torch.tensor(x_values, dtype=torch.float32)
p = torch.tensor(p_values, dtype=torch.complex64)
p.requires_grad = True


def fourier_coefficient(p, x, n):
    delta_x = 2 * np.pi / 10000
    f_x = torch.exp(1j * p)  # This is f(x) = exp(1j * p(x))

    # Fourier integration approximation for nth order coefficient
    sum_val = torch.tensor([0 + 1j * 0], dtype=torch.complex64)
    for f_val, x_val in zip(f_x, x):
        sum_val += f_val * torch.exp(-1j * n * x_val) * delta_x

    return sum_val / (2 * np.pi)


def energy_efficiency(p, x):
    total_energy = torch.tensor([0.0], dtype=torch.float32)
    for i in np.arange(-100, 101, 1):
        coefficient = fourier_coefficient(p, x, i)
        total_energy += abs(coefficient) ** 2

    abs_square_coefficient_pos1 = abs(fourier_coefficient(p, x, 1)) ** 2
    abs_square_coefficient_neg1 = abs(fourier_coefficient(p, x, -1)) ** 2

    efficiency_ratio = (abs_square_coefficient_pos1 + abs_square_coefficient_neg1) / total_energy
    return efficiency_ratio


def eval(p, x, alpha=0.5):
    abs_square_coefficient_pos1 = abs(fourier_coefficient(p, x, 1)) ** 2
    abs_square_coefficient_neg1 = abs(fourier_coefficient(p, x, -1)) ** 2
    balance_term = abs(abs_square_coefficient_pos1 - abs_square_coefficient_neg1)
    return (1 - alpha) * (1 - energy_efficiency(p, x)) + alpha * balance_term


learning_rate = 0.01

for epoch in range(100):
    e = eval(p, x)
    e.backward()
    with torch.no_grad():
        p -= learning_rate * p.grad
        p.grad.zero_()
    print('Epoch:', epoch, 'Eval:', e.item())

print("Result:", p)

abs_square_coefficient_pos1 = abs(fourier_coefficient(p, x, 1)) ** 2
abs_square_coefficient_neg1 = abs(fourier_coefficient(p, x, -1)) ** 2
print("coefficient_pos1:", abs_square_coefficient_pos1)
print("coefficient_neg1:", abs_square_coefficient_neg1)

print("energy_efficiency:", energy_efficiency(p, x))

