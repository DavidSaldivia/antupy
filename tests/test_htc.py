import numpy as np
import pytest

import antupy as ap

def test_horizontal_surface_upper_hot():
    T_s = 400.
    T_inf = 300.
    L = 1.
    assert (
        np.round(ap.htc.h_horizontal_surface_upper_hot(T_s,T_inf,L,correlation="Holman"), 1)
        == 7.1
        )
    assert (
        np.round(ap.htc.h_horizontal_surface_upper_hot(T_s,T_inf,L,correlation="NellisKlein"), 1)
        == 6.7
        )


def test_horizontal_surface_invalid_correlation_raises():
    with pytest.raises(ValueError):
        ap.htc.h_horizontal_surface_upper_hot(400.0, 300.0, 1.0, correlation="invalid")


def test_temp_sky_simplest():
    assert ap.htc.temp_sky_simplest(300.0) == 285.0


def test_h_ext_flat_plate_with_float_inputs():
    h = ap.htc.h_ext_flat_plate(
        temp_surf=350.0,
        temp_fluid=300.0,
        length=1.0,
        u_inf=5.0,
    )
    assert isinstance(h, ap.Var)
    assert h.u == "W/m2-K"
    assert h.gv("W/m2-K") > 0


def test_h_ext_flat_plate_with_var_inputs():
    h = ap.htc.h_ext_flat_plate(
        temp_surf=ap.Var(350.0, "K"),
        temp_fluid=ap.Var(300.0, "K"),
        length=ap.Var(1.0, "m"),
        u_inf=ap.Var(5.0, "m/s"),
    )
    assert isinstance(h, ap.Var)
    assert h.u == "W/m2-K"
    assert h.gv("W/m2-K") > 0


def test_h_ext_flat_plate_invalid_type_raises():
    with pytest.raises(ValueError):
        ap.htc.h_ext_flat_plate(temp_surf="bad", temp_fluid=300.0, length=1.0, u_inf=5.0)


def test_external_placeholder_functions_return_var_none():
    h1 = ap.htc.h_ext_flat_plane_constant_flux()
    h2 = ap.htc.h_ext_cylinder()
    h3 = ap.htc.h_ext_sphere()

    assert isinstance(h1, ap.Var)
    assert isinstance(h2, ap.Var)
    assert isinstance(h3, ap.Var)
    assert h1.u == "W/m2-K"
    assert h2.u == "W/m2-K"
    assert h3.u == "W/m2-K"
    assert np.isnan(h1.v)
    assert np.isnan(h2.v)
    assert np.isnan(h3.v)