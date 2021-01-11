[![PyPI version](https://badge.fury.io/py/diffeqtorch.svg)](https://badge.fury.io/py/diffeqtorch)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/jan-matthis/diffeqtorch/blob/master/CONTRIBUTING.md)

# diffeqtorch

Bridges `DifferentialEquations.jl` with PyTorch. Besides benefitting from the huge range of solvers available in `DifferentialEquations.jl`, this allows taking gradients through solvers using [local sensitivity analysis/auto-diff](https://docs.sciml.ai/stable/analysis/sensitivity/). The package has only been tested with ODE problems, and in particular, automatic differentiation is only supported for ODEs using `ForwardDiff.jl`. This can be extended in the future, [contributions are welcome](https://github.com/jan-matthis/diffeqtorch/blob/master/CONTRIBUTING.md).


### Examples

- [Simple ODE problem to demonstrate the interface and confirm gradients with analytical solution](https://github.com/jan-matthis/diffeqtorch/blob/master/notebooks/01_simple_ode.ipynb)
- [SIR model for a slighlty more complicated model with numerical gradient checking](https://github.com/jan-matthis/diffeqtorch/blob/master/notebooks/02_sir_model.ipynb)
- [Hodgkin-Huxley model for a realistic example from Neuroscience](https://github.com/jan-matthis/diffeqtorch/blob/master/notebooks/03_hh_model.ipynb)


## Installation

Prerequisites for using `diffeqtorch` are installation of Julia and Python. Note that the binary directory of `julia` needs to be in your `PATH`.

Install `diffeqtorch`:
```commandline
$ pip install diffeqtorch
$ export JULIA_SYSIMAGE_DIFFEQTORCH="$HOME/.julia_sysimage_diffeqtorch.so"
$ python -c "from diffeqtorch.install import install_and_test; install_and_test()"
```

We recommend using a custom Julia system image containing dependencies. By setting the environment variable `JULIA_SYSIMAGE_DIFFEQTORCH`, an image will be created and used automatically. This may take a while but will improve speed afterwards.


## Usage

```python
from diffeqtorch import DiffEq

f = """
function f(du,u,p,t)
    du[1] = p[1] * u[1]
end
"""
de = DiffEq(f)

u0 = torch.tensor([1.])
tspan = torch.tensor([0., 3.])
p = torch.tensor([1.01])

u, t = de(u0, tspan, p)
```

See also `help(DiffEq)` and examples provided in `notebooks/`.


## License

MIT
