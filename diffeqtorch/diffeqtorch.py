from __future__ import annotations

import os
import time
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Union
from warnings import warn

import torch
from julia.api import Julia
from opt_einsum import contract

from diffeqtorch.logging import get_logger

JULIA_PROJECT = str(Path(__file__).parent / "julia")
os.environ["JULIA_PROJECT"] = JULIA_PROJECT


def find_sysimage():
    if "JULIA_SYSIMAGE_DIFFEQTORCH" in os.environ:
        environ_path = Path(os.environ["JULIA_SYSIMAGE_DIFFEQTORCH"])
        if environ_path.exists():
            return str(environ_path)
        else:
            warn("JULIA_SYSIMAGE_DIFFEQTORCH is set but image does not exist")
            return None
    else:
        warn("JULIA_SYSIMAGE_DIFFEQTORCH not set")
        default_path = Path("~/.julia_sysimage_diffeqtorch.so").expanduser()
        if default_path.exists():
            warn(f"Defaulting to {default_path}")
            return str(default_path)
        else:
            return None


class DiffEq(torch.nn.Module):
    """Solve differential equation using DifferentialEquations.jl

    Usage examples are provided in notebooks folder of `diffeqtorch`.
    """

    def __init__(
        self,
        f: str,
        problem_type_grad: str = "ODEForwardSensitivityProblem",
        problem_type_no_grad: str = "ODEProblem",
        solver: str = "Tsit5",
        reltol: float = 1e-8,
        abstol: float = 1e-8,
        saveat: float = 0.1,
        using: List[str] = ["DifferentialEquations, DiffEqSensitivity"],
        pyjulia_opts: Dict[str, Any] = {
            "compiled_modules": False,
            "sysimage": find_sysimage(),
            "runtime": "julia",
        },
        debug: Union[bool, int] = 0,
    ):
        """Initialize solver module

        Args:
            f: Function definition in Julia, see [1] for details
            problem_type_grad: The DifferentialEquations.jl problem type used when
                gradients calculation is required, see [1] for details
            problem_type_no_grad: The DifferentialEquations.jl problem type used when
                no gradients calculation is required, see [1] for details
            solver: The solver that is used, see [1] for details
            reltol: Relative tolerance, see [1] for details
            abstol: Absolute tolerance, see [1] for details
            saveat: At which timesteps outputs are saved, see [1] for details
            using: Packages that are imported to Julia session
            pyjulia_opts: Keyword dict passed to pyjulia at initialisation, see [2]
                for details
            debug: Amount of debug info (from nothing to a lot), options are:
                - 0: No info, also used when `debug=False`
                - 1: General info including Julia commands, also used when `debug=True`
                - 2: Additionally logging PyTorch shapes
                - 3: Additionally logging PyTorch values
                - 4: Additionally turning on PyJulia debug mode

        [1]: https://docs.sciml.ai/stable/
        [2]: https://pyjulia.readthedocs.io/en/stable/api.html#julia.api.Julia
        """
        super().__init__()

        self.opts = {
            "problem_type_grad": problem_type_grad,
            "problem_type_no_grad": problem_type_no_grad,
            "solver": solver,
            "reltol": reltol,
            "abstol": abstol,
            "saveat": saveat,
            "using": using,
            "debug": debug,
        }

        # Logging
        self.log = get_logger(__name__)
        self.debug = debug
        if self.debug > 0:
            self.log.debug("Initializing `DiffEq` with keywords:")
            for key, value in self.opts.items():
                self.log.debug(f"    {key}: {value}")

        # Start pyjulia
        if self.debug > 3:
            pyjulia_opts["debug"] = True
        tic = time.time()
        self.jl = Julia(**pyjulia_opts)
        if self.debug > 0:
            self.log.debug(
                f"\nStarted Julia through `PyJulia`, took {time.time()-tic:.2f}sec"
            )

        # Import packages
        cmd = f"using {','.join(using)}"
        if self.debug > 0:
            self.log.debug(f"\nJulia >>>\n{cmd}\n<<<\n")
        self.jl.eval(cmd)

        # Evalulate function definition
        cmd = f
        if self.debug > 0:
            self.log.debug(f"\nJulia >>>\n{cmd}\n<<<\n")
        self.jl.eval(cmd)

    def forward(
        self, u0: torch.Tensor, tspan: torch.Tensor, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through solver

        Args:
            u0: Initial values as a 1d tensor
            tspan: Start and end points for integration
            p: Parameters as a 1d tensor

        Returns:
            Returns solutions `u` and timesteps `t`
        """
        return DifferentialEquationsJuliaFunction.apply(
            p, u0, tspan, self.jl, self.opts
        )

    @property
    def debug(self):
        return self.opts["debug"]

    @debug.setter
    def debug(self, state):
        self.opts["debug"] = int(state)


class DifferentialEquationsJuliaFunction(torch.autograd.Function):
    """Forward and backward pass through solver

    Gradients are calculated using `DiffEqSensitivity.jl`. In the backward pass,
    these gradients need to be multiplied with the incoming gradients according
    to the chain rule (see [1] for documentation of the `torch.autograd.Function`
    interface and examples). This multiplication is performed using `einsum` [2],
    where we use an optimized implementation [3] for speed.

    [1]: https://pytorch.org/docs/stable/notes/extending.html
    [2]: https://rockt.github.io/2018/04/30/einsum
    [3]: http://optimized-einsum.readthedocs.io
    """

    @staticmethod
    def forward(ctx, p, u0, tspan, jl, opts):
        debug = opts["debug"]
        if debug > 0:
            log = get_logger(__name__)

        if ctx.needs_input_grad[0]:
            compute_grad_p = True
        else:
            compute_grad_p = False

        if ctx.needs_input_grad[1]:
            raise NotImplementedError("Gradient w.r.t. u0 not supported")

        if ctx.needs_input_grad[2]:
            raise NotImplementedError("Gradient w.r.t. tspan not supported")

        if debug > 1:
            log.debug(f"p.shape: {p.shape}")
            log.debug(f"u0.shape: {u0.shape})")
            log.debug(f"tspan.shape: {tspan.shape}")

        if debug > 2:
            log.debug(f"p: {p}")
            log.debug(f"u0: {u0})")
            log.debug(f"tspan: {tspan}")
            log.debug(f"compute_grad_p: {compute_grad_p}")

        cmd = dedent(
            f"""
        u0 = {u0.tolist()}
        tspan = {tuple(tspan.tolist())}
        p = {p.tolist()}
        """
        )

        if compute_grad_p:
            cmd += dedent(
                f"""
            prob = {opts['problem_type_grad']}(f,u0,tspan,p)

            solution = solve(prob,{opts['solver']}(),reltol={opts['reltol']},abstol={opts['abstol']},saveat={opts['saveat']})

            u, du = extract_local_sensitivities(solution)

            u, du, solution.t
            """  # noqa: E501
            )
            if debug > 0:
                log.debug(f"\nJulia >>>\n{cmd}\n<<<\n")

            u, du, t = jl.eval(cmd)

            u = torch.tensor(u)
            du = torch.tensor(du)
            t = torch.tensor(t, requires_grad=False)

            if debug > 1:
                log.debug(f"u.shape: {u.shape})")
                log.debug(f"du.shape: {du.shape})")
                log.debug(f"t.shape: {t.shape})")

            if debug > 2:
                log.debug(f"u: {u})")
                log.debug(f"du: {du})")
                log.debug(f"t: {t})")

            ctx.save_for_backward(du, torch.tensor(debug))

            return u, t

        else:  # No gradient computation
            cmd += dedent(
                f"""
            prob = {opts['problem_type_no_grad']}(f,u0,tspan,p)

            solution = solve(prob,{opts['solver']}(),reltol={opts['reltol']},abstol={opts['abstol']},saveat={opts['saveat']})

            solution.u, solution.t
            """  # noqa: E501
            )
            if debug > 0:
                log.debug(f"\nJulia >>>\n{cmd}\n<<<\n")

            u, t = jl.eval(cmd)
            t = torch.tensor(t, requires_grad=False)
            u = torch.tensor(u).T

            # NOTE: The transpose on u is to ensure that shapes are
            # consistent with the case above in which gradients are computed
            # and another interface is used.

            if debug > 1:
                log.debug(f"u.shape: {u.shape})")
                log.debug(f"t.shape: {t.shape})")

            if debug > 2:
                log.debug(f"u: {u})")
                log.debug(f"t: {t})")

            return u, t

    @staticmethod
    def backward(ctx, grad_output_u, grad_output_t):
        du = ctx.saved_tensors[0]

        debug = ctx.saved_tensors[1].item()
        if debug > 0:
            log = get_logger(__name__)

        if debug > 1:
            log.debug(f"du.shape: {du.shape})")
            log.debug(f"grad_output_u.shape: {grad_output_u.shape})")

        if debug > 2:
            log.debug(f"du: {du})")
            log.debug(f"grad_output_u: {grad_output_u})")

        grad_p = contract("pot,ot->p", du, grad_output_u)

        if debug > 1:
            log.debug(f"grad_p.shape: {grad_p.shape})")

        if debug > 2:
            log.debug(f"grad_p: {grad_p})")

        return grad_p, None, None, None, None
