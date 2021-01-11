import torch
from torch.autograd import gradcheck

from diffeqtorch import DiffEq


def test_gradient_simple_ode_analytical():
    de = DiffEq(
        f="""
    function f(du,u,p,t)
        du[1] = p[1] * u[1]
    end
    """
    )

    u0 = torch.tensor([1.0])
    tspan = torch.tensor([0.0, 3.0])
    p = torch.tensor([2.0], requires_grad=True)
    u, t = de(u0, tspan, p)

    loss = (u ** 2).sum()
    loss.backward()

    analytical = (2 * t * torch.exp(2 * p * t)).sum()

    torch.testing.assert_allclose(p.grad, analytical.float())


def test_gradient_simple_ode_numerical():
    de = DiffEq(
        f="""
    function f(du,u,p,t)
        du[1] = p[1] * u[1]
    end
    """
    )

    u0 = torch.tensor([1.0])
    tspan = torch.tensor([0.0, 3.0])
    p = torch.tensor([2.0], requires_grad=True)
    u, t = de(u0, tspan, p)

    def loss(p):
        u, t = de(u0, tspan, p)
        return (u ** 2).sum()

    gradcheck(loss, p, eps=1e-4, atol=1e-4)


def test_gradient_sir_numerical():
    N = 1000.0
    de = DiffEq(
        f"""
    function f(du,u,p,t)
        S,I,R = u
        b,g = p
        du[1] = -b * S * I / {N}
        du[2] = b * S * I / {N} - g * I
        du[3] = g * I
    end
    """,
        saveat=1.0,
    )

    I0 = 1.0
    R0 = 0.0
    S0 = N - I0 - R0
    u0 = torch.tensor([S0, I0, R0])
    tspan = torch.tensor([0.0, 160.0])
    p = torch.tensor([0.3, 0.1], requires_grad=True)
    u, t = de(u0, tspan, p)

    def loss(p):
        u, t = de(u0, tspan, p)
        return (u ** 2).sum()

    gradcheck(loss, p, eps=1e-4, atol=1e-4)
