import math


def test_julia_call(julia):
    julia._call("1 + 1")
    julia._call("sqrt(2.0)")


def test_julia_call_packages(julia):
    julia._call("using DifferentialEquations")
    julia._call("using DiffEqSensitivity")


def test_julia_eval(julia):
    assert julia.eval("1 + 1") == 2
    assert julia.eval("sqrt(2.0)") == math.sqrt(2.0)
    assert julia.eval("PyObject(1)") == 1
    assert julia.eval("PyObject(1000)") == 1000
    assert julia.eval("PyObject((1, 2, 3))") == (1, 2, 3)
