import numpy as np
import pytest

from antupy import Var
from antupy.utils.htc import (
    h_horizontal_surface_upper_hot,
    h_ext_flat_plate,
    h_ext_flat_plane_constant_flux,
    h_ext_cylinder,
    h_ext_sphere,
    temp_sky_simplest,
)

def test_horizontal_surface_upper_hot():
    T_s = 400.
    T_inf = 300.
    L = 1.
    assert (
        np.round(h_horizontal_surface_upper_hot(T_s,T_inf,L,correlation="Holman"), 1)
        == 7.1
        )
    assert (
        np.round(h_horizontal_surface_upper_hot(T_s,T_inf,L,correlation="NellisKlein"), 1)
        == 6.7
        )


def test_horizontal_surface_invalid_correlation_raises():
    with pytest.raises(ValueError):
        h_horizontal_surface_upper_hot(400.0, 300.0, 1.0, correlation="invalid")


def test_temp_sky_simplest():
    assert temp_sky_simplest(300.0) == 285.0


def test_h_ext_flat_plate_with_float_inputs():
    h = h_ext_flat_plate(
        temp_surf=350.0,
        temp_fluid=300.0,
        length=1.0,
        u_inf=5.0,
    )
    assert isinstance(h, Var)
    assert h.u == "W/m2-K"
    assert h.gv("W/m2-K") > 0


def test_h_ext_flat_plate_with_var_inputs():
    h = h_ext_flat_plate(
        temp_surf=Var(350.0, "K"),
        temp_fluid=Var(300.0, "K"),
        length=Var(1.0, "m"),
        u_inf=Var(5.0, "m/s"),
    )
    assert isinstance(h, Var)
    assert h.u == "W/m2-K"
    assert h.gv("W/m2-K") > 0


def test_h_ext_flat_plate_invalid_type_raises():
    with pytest.raises(ValueError):
        h_ext_flat_plate(temp_surf="bad", temp_fluid=300.0, length=1.0, u_inf=5.0)


def test_external_placeholder_functions_return_var_none():
    h1 = h_ext_flat_plane_constant_flux()
    h2 = h_ext_cylinder()
    h3 = h_ext_sphere()

    assert isinstance(h1, Var)
    assert isinstance(h2, Var)
    assert isinstance(h3, Var)
    assert h1.u == "W/m2-K"
    assert h2.u == "W/m2-K"
    assert h3.u == "W/m2-K"
    assert np.isnan(h1.v)
    assert np.isnan(h2.v)
    assert np.isnan(h3.v)