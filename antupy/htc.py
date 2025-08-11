import numpy as np

from antupy.props import Air

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