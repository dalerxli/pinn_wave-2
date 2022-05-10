import torch 
import torch.nn as nn


class GradientLayer(nn.Module):
    """
    Custom layer to compute 1st and 2nd derivatives for the wave equation.

    Attributes:
        model: pytorch network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """
        super().__init__(**kwargs)
        self.model = model
        self.device = next(model.parameters()).device


    def forward(self, tx):
        """
        Computing 1st and 2nd derivatives for the wave equation.

        Args:
            tx: input variables (t, x).

        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
        """
        t = tx[:, 0:1]
        x = tx[:, 1:2]
        t = torch.tensor(t, requires_grad=True, dtype=torch.float64, device=self.device)
        x = torch.tensor(x, requires_grad=True, dtype=torch.float64, device=self.device)
        u = self.model(torch.cat((t, x), dim=1))
        du_dt = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        d2u_dt2 = torch.autograd.grad(
            du_dt, t, 
            grad_outputs=torch.ones_like(du_dt),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dx = torch.autograd.grad(
            u, x, 
            torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        d2u_dx2 = torch.autograd.grad(
            du_dx, x, 
            torch.ones_like(du_dx),
            retain_graph=True,
            create_graph=True
        )[0]

        return u, du_dt, du_dx, d2u_dt2, d2u_dx2
