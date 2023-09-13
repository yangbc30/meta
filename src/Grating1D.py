import torch
import matplotlib.pyplot as plt


def even_case(M):
    return list(range(-2 * M + 1, 2 * M, 2))


def odd_case(M):
    return list(range(-2 * M + 2, 2 * M - 1, 2))


def exp_then_FT(p_vec):
    """exp p_vec to get modulated field, then do FFT to get coefficience of each order
    Args:
        p_vec (tenser): phase of the gating indroduce
    Returns:
        tensor: fourier coefficiency, shape of p_vec
    """
    exp_p = torch.exp(1j * p_vec)  # p_vec do not need to use torch.complex64
    # question: if p_vec use torch.complex64, will autograd compute imagine part
    # return exp_p
    return torch.fft.fft(exp_p) / len(exp_p)


class Grating1D(object):
    def __init__(self, orders) -> None:
        self.params = {}
        self.orders = torch.tensor(orders)

    def energy_of_orders(self):
        """print energy of target order

        Returns:
            tensor: energy list of target order
        """
        coefficient = exp_then_FT(self.get_p_vec())
        energy_list = torch.abs(coefficient[self.orders]) ** 2
        return energy_list

    def forward(self):
        pass

    def get_p_vec(self):
        return self.params["p_vec"]

    def visualize(self):
        """visualize the phase function and fourier coefficient"""
        p_vec = self.get_p_vec().detach()

        coefficient = exp_then_FT(p_vec).detach()

        plt.subplot(1, 2, 1)
        plt.stem(p_vec.numpy())
        plt.title("phase function")

        plt.subplot(1, 2, 2)
        plt.stem(torch.abs(coefficient).numpy())
        plt.title("Fourier coefficient")

        # test energy conserve
        # frequency_energy = torch.sum(torch.abs(coefficient) ** 2).item()
        # phase_energy = torch.sum(torch.abs(torch.exp(1j * p_vec)) ** 2).item()
        # print("energy of frequncy domain", frequency_energy)
        # print("energy of phase domain", phase_energy)

        print("Calculate Energy Efficiency:")
        print(
            "Energy Efficiency =",
            torch.sum(self.energy_of_orders()).detach().item(),
        )
        print("\n")

        print("energy of some order:")
        coeff_indices = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6]
        for index in coeff_indices:
            coeff_value = coefficient[index]
            magnitude = torch.abs(coeff_value).detach().item()
            magnitude_squared = magnitude**2
            # print(f"coefficient {index}: {coeff_value}, Magnitude: {magnitude}")
            print(f"order {index}: {magnitude_squared}")
        print("\n")


class ConstrainedOptimGrating1D(Grating1D):
    def __init__(self, orders=[-1, 1], segment=10000, alpha=0.5) -> None:
        """initialize the grating, with hyperparameter alpha

        Args:
            segment (int): divide the grating into how many segment
            alpha (float): between 0 and 1, trade off between energy efficiency and uniformity
        """
        super().__init__(orders)

        self.alpha = alpha
        p_vec = torch.randn(segment) * 0.1
        p_vec.requires_grad = True

        self.params["p_vec"] = p_vec

    def forward(self):
        """forward function, compute loss

        Returns:
            scale: loss
        """

        loss = 0

        total_energy = 1  # phase domian magnitude is all 1, energy conserve

        target_energy_list = self.energy_of_orders()
        target_energy = torch.sum(target_energy_list)

        energy_efficiency = target_energy / total_energy
        energy_variance = torch.var(target_energy_list)

        loss += (self.alpha) * (1 - energy_efficiency)
        loss += (1 - self.alpha) * energy_variance

        return loss


class LeastSquaresOptimGrating1D(Grating1D):
    def __init__(self, orders=[-1, 1], segment=10000) -> None:
        """initialize the grating

        Args:
            segment (int): divide the grating into how many segment
        """
        super().__init__(orders)

        p_vec = torch.randn(segment) * 0.1
        p_vec.requires_grad = True
        # split two beams +1, -1
        a_vec = torch.randn(self.orders.shape[0]) * 0.1
        a_vec.requires_grad = True

        self.params["p_vec"] = p_vec
        self.params["a_vec"] = a_vec
        self.segment = segment

    def sum_of_fourier_orders(self):
        x_vec = torch.linspace(-torch.pi, torch.pi, self.segment)

        _phase = x_vec.unsqueeze(1) + self.params["a_vec"]
        _exp = torch.exp(_phase * self.orders * 1j)
        approx = torch.sum(_exp, dim=1) / torch.sqrt(torch.tensor(self.orders.shape[0]))
        return approx

    def forward(self):
        """forward function, compute loss

        Returns:
            scale: loss
        """
        loss = 0
        p_vec = self.params["p_vec"]

        approx = self.sum_of_fourier_orders()

        exp_p = torch.exp(1j * p_vec)
        norm2 = torch.linalg.vector_norm(exp_p - approx)

        loss = norm2**2 / self.segment
        return loss

    def visualize(self):

        print("learned a_k of target order")
        for index, order in enumerate(self.orders):
            print(f"a_{order} = {self.params['a_vec'][index].item()}")
        print("\n")
        return super().visualize()


class MinimumVarianceOptimGrating1D(Grating1D):
    def __init__(self, orders, segment=10000) -> None:
        super().__init__(orders)

        a_vec = torch.randn(self.orders.shape[0]) * 1
        a_vec.requires_grad = True

        self.params["a_vec"] = a_vec
        self.segment = segment

    def sum_of_fourier_orders(self):
        x_vec = torch.linspace(-torch.pi, torch.pi, self.segment)

        _phase = x_vec.unsqueeze(1) + self.params["a_vec"]
        _exp = torch.exp(_phase * self.orders * 1j)
        approx = torch.sum(_exp, dim=1) / torch.sqrt(torch.tensor(self.orders.shape[0]))
        return approx

    def forward(self):
        """forward function, compute loss

        Returns:
            scale: loss
        """

        loss = 0

        intensity = torch.abs(self.sum_of_fourier_orders()) ** 2

        loss = torch.var(intensity)

        return loss

    def get_p_vec(self):
        """construct p_vec from a_vec

        Returns:
            p_vec: p_vec
        """

        p_vec = torch.angle(self.sum_of_fourier_orders())

        return p_vec.detach()

    def visualize(self):

        print("learned a_k of target order")
        for index, order in enumerate(self.orders):
            print(f"a_{order} = {self.params['a_vec'][index].item()}")
        print("\n")

        return super().visualize()



class Solver(object):
    def __init__(self, model):
        """load the model to optimize

        Args:
            model (grating): model.params has all the parameter to optimize, model.forward yield loss
        """
        self.model = model

    def set_optim(self, optim_func, **kwargs):
        self.optim = optim_func(self.model.params.values(), **kwargs)

    def train(self, step, verbose):
        self.loss_history = []
        for epoch in range(step):
            self.optim.zero_grad()
            loss = self.model.forward()
            loss.backward()
            self.optim.step()
            self.loss_history.append(loss.item())
            if verbose:
                print("Epoch:", epoch, "Loss:", loss.item())

        if verbose:
            plt.plot(self.loss_history, label="loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()

        # return loss_history
