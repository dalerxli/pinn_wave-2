import timeit
import torch
import torch.nn as nn
import numpy as np
from layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the wave equation.

    Attributes:
        network: pytorch network model with input (t, x) and output u(t, x).
        c: wave velocity.
        grads: gradient layer.
    """

    def __init__(self, network, inputs, outputs, c=1):
        """
        Args:
            network: pytorch network model with input (t, x) and output u(t, x).
            c: wave velocity. Default is 1.
        """
        self.network = network
        self.c = c
        self.grads = GradientLayer(self.network)
        self.inputs = inputs 
        self.ouputs = outputs
        self.num_epoch = 0
        self.optimizer = torch.optim.LBFGS(
            network.parameters(), 
            lr=1.0, 
            max_iter=15000, 
            max_eval=15000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )
        self.loss_list = list()


    def loss(self):
        # compute gradients
        _, _, _, d2u_dt2, d2u_dx2 = self.grads(self.inputs[0])

        # equation output being zero
        u_eqn = d2u_dt2 - self.c*self.c * d2u_dx2
        # initial condition output
        u_ini, du_dt_ini, _, _, _ = self.grads(self.inputs[1])
        # boundary condition output
        u_bnd, _, _, _, _ = self.grads(self.inputs[2])  # dirichlet
        #_, _, u_bnd, _, _ = self.grads(tx_bnd)  # neumann

        loss_val = torch.mean(torch.square(u_eqn - self.ouputs[0])) \
                + torch.mean(torch.square(u_ini - self.ouputs[1])) \
                + torch.mean(torch.square(du_dt_ini - self.ouputs[2])) \
                + torch.mean(torch.square(u_bnd - self.ouputs[3])) 
        self.optimizer.zero_grad()
        loss_val.backward()
        if self.num_epoch % 100 == 0:
            # pre_time = self.time 
            self.stop = timeit.default_timer()
            self.loss_list.append(loss_val.item())
            print(f'Epoch: {self.num_epoch}'.ljust(12), f'Loss: {loss_val.item():.8f}', f'Time: {self.stop-self.start:.1f}')
        self.num_epoch += 1
        return loss_val

    def train(self):
        self.network.train()
        self.start = timeit.default_timer()
        self.optimizer.step(self.loss)

