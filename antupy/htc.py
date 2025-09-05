import numpy as np

from antupy import Var
from antupy.props import Fluid, Air, Water

SIGMA_CONSTANT = 5.67e-8

def temp_sky_simplest(temp_amb: float) -> float:
    """simplest function to estimate sky temperature. It is just temp_amb-15.[K]

    Args:
        temp_amb (float): temperature in K

    Returns:
        float: sky temperature
    """
    return (temp_amb - 15.)

def h_horizontal_surface_upper_hot(
        T_s: float,
        T_inf: float,
        L: float,
        P: float = 101325,
        fluid: Air = Air(),
        correlation: str = "NellisKlein"
    ) -> float:
    """
    Correlation for natural convection in upper hot surface horizontal plate
    T_s, T_inf          : surface and free fluid temperatures [K]
    L                   : characteristic length [m]
    """
    T_av = ( T_s + T_inf )/2
    mu = fluid.viscosity(T_av, P).v
    k = fluid.k(T_av, P).v
    rho = fluid.rho(T_av, P).v
    cp = fluid.cp(T_av, P).v
    alpha = k/(rho*cp)
    beta = 1./T_s
    visc = mu/rho
    Pr = visc/alpha
    g = 9.81
    Ra = g * beta * abs(T_s - T_inf) * L**3 * Pr / visc**2
    if correlation == "Holman":
        if Ra > 1e4 and Ra < 1e7:
            Nu = 0.54*Ra**0.25
            h = (k*Nu/L)
        elif Ra>= 1e7 and Ra < 1e9:
            Nu = 0.15*Ra**(1./3.)
            h = (k*Nu/L)
        else:
            h = 1.52*(T_s-T_inf)**(1./3.)
        return h
    elif correlation == "NellisKlein":
        C_lam  = 0.671 / ( 1+ (0.492/Pr)**(9/16) )**(4/9)
        Nu_lam = float(1.4/ np.log(1 + 1.4 / (0.835*C_lam*Ra**0.25) ) )
        C_tur  = 0.14*(1 + 0.0107*Pr)/(1+0.01*Pr)
        Nu_tur = C_tur * Ra**(1/3)
        Nu = (Nu_lam**10 + Nu_tur**10)**(1/10)
        h = (k*Nu/L)
        return h
    else:
        raise ValueError(f"label {correlation} is not a valid correlation label.")
    

def h_ext_flat_plate(
        temp_surf: float | Var = Var(300, "K"),
        temp_fluid: float | Var = Var(400, "K"),
        length: float | Var = Var(1, "m"),
        u_inf: float | Var = Var(10, "m/s"),
        fluid: Fluid = Water(),
        Re_crit: float = 4e5
) -> Var:
    if isinstance(temp_surf, Var):
        Ts = temp_surf.gv("K")
    elif isinstance(temp_surf, (int, float)):
        Ts = temp_surf
    else:
        raise ValueError(f"{type(temp_surf)=} is not a valid type")
    if isinstance(temp_fluid, Var):
        Tf = temp_fluid.gv("K")
    elif isinstance(temp_fluid, (int, float)):
        Tf = temp_fluid
    else:
        raise ValueError(f"{type(temp_fluid)=} is not a valid type")
    if isinstance(length, Var):
        L = length.gv("m")
    elif isinstance(length, (int, float)):
        L = length
    else:
        raise ValueError(f"{type(length)=} is not a valid type")
    if isinstance(u_inf, Var):
        u = u_inf.gv("m/s")
    elif isinstance(u_inf, (int, float)):
        u = u_inf
    else:
        raise ValueError(f"{type(u_inf)=} is not a valid type")
    rho = fluid.rho(Tf).gv("kg/m3")
    cp = fluid.cp(Tf).gv("J/kg-K")
    mu = fluid.viscosity(Tf).gv("Pa-s")
    k = fluid.k(Tf).gv("W/m-K")
    Re_L = rho * u * L / mu
    Pr = cp * mu / k
    Nu_L = (
        0.6774 * Pr**(1/3) * Re_crit**(1./2)
        / (1 + (0.0468/Pr)**(2/3))**(1./4)
        + 0.037*Pr**(1/3) * (Re_L**0.8 - Re_crit**0.8)
        )
    return Var(Nu_L * k / L, "W/m2-K")


def h_ext_flat_plane_constant_flux() -> Var:
    pass

def h_ext_cylinder() -> Var:
    # See page 614 from Nellis&Klein
    pass

def h_ext_sphere() -> Var:
    # See page 619 from Nellis&Klein 
    pass

