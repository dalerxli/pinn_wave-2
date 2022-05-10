import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pinn import PINN
from network import Network
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def u0(t, x, c=1, k=2, sd=0.3):
    """
    Initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.
        c: wave velocity.
        k: wave number.
        sd: standard deviation.

    Returns:
        u(t, x) as tf.Tensor.
    """
    z = k*x - (c*k)*t
    return torch.sin(z) * torch.exp(-(0.5*z/sd)**2)

def du0_dt(t, x):
    """
    First derivative of t for the initial wave form.

    Args:
        tx: variables (t, x) as tf.Tensor.

    Returns:
        du(t, x)/dt as tf.Tensor.
    """
    u = u0(t, x)
    du_dt = torch.autograd.grad(
        u, t, 
        grad_outputs=torch.ones_like(u),
        retain_graph=False, 
        create_graph=False
    )[0]
    return du_dt.detach()

if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for the wave equation.
    """

    # number of training samples
    num_train_samples = 1000#10000
    # number of test samples
    num_test_samples = 1000

    # build a core network model
    network = Network.build()
    summary(network, (2, ), device='cpu')
    network.to(torch.float64).to(device)


    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)
    tx_eqn[..., 0] = 4*tx_eqn[..., 0]                # t =  0 ~ +4
    tx_eqn[..., 1] = 2*tx_eqn[..., 1] - 1            # x = -1 ~ +1
    tx_ini = np.random.rand(num_train_samples, 2)
    tx_ini[..., 0] = 0                               # t = 0
    tx_ini[..., 1] = 2*tx_ini[..., 1] - 1            # x = -1 ~ +1
    tx_bnd = np.random.rand(num_train_samples, 2)
    tx_bnd[..., 0] = 4*tx_bnd[..., 0]                # t =  0 ~ +4
    tx_bnd[..., 1] = 2*np.round(tx_bnd[..., 1]) - 1  # x = -1 or +1

    inputs = [tx_eqn, tx_ini, tx_bnd]



    # create training output
    u_zero = torch.zeros((num_train_samples, 1), dtype=torch.float64, device=device)
    t0 = torch.tensor(tx_ini[:, 0:1], device=device, dtype=torch.float64, requires_grad=True)
    x0 = torch.tensor(tx_ini[:, 1:2], device=device, dtype=torch.float64)
    u_ini = u0(t0, x0).detach()
    du_dt_ini = du0_dt(t0, x0)
    outputs = [u_zero, u_ini, du_dt_ini, u_zero.clone()]

    # build a PINN model
    pinn = PINN(network, inputs, outputs)
    pinn.train()


    # predict u(t,x) distribution
    t_flat = np.linspace(0, 4, num_test_samples)
    x_flat = np.linspace(-1, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    tx = torch.tensor(tx, device=device, dtype=torch.float64)
    u = network(tx).detach().cpu().numpy()
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    vmin, vmax = -0.5, +0.5
    plt.pcolormesh(t, x, u, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(vmin, vmax)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0.5, 1, 2]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        tx = torch.tensor(tx, device=device, dtype=torch.float64)
        u = network(tx).detach().cpu().numpy()
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.savefig('result_img_dirichlet.png', transparent=True)
    plt.show()
