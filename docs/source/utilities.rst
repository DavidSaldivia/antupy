Utility modules
==================

The ``props`` module
------------------------

The ``props`` module in `antupy` provides a library of thermophysical properties for various fluids and materials. For all materials it allows users to easily access and utilize properties such as density, specific heat, and thermal conductivity. For fluids, specifically, it also includes properties like viscosity, enthalpy, etc.

All properties are returned as :py:class:`~antupy.Var` objects with appropriate units, making them compatible with the unit-aware calculation framework. The module includes materials such as molten salts (SolarSalt, Carbo), metals (Aluminium, Copper, StainlessSteel), and various fluids (water, air, CO2, thermal oils).

Using Material Properties
^^^^^^^^^^^^^^^^^^^^^^^^^

Materials provide basic thermophysical properties like density, specific heat, and thermal conductivity. Here's an example using copper:

.. code-block:: python

    from antupy.utils.props import Copper
    from antupy import Var

    # Create material instance
    copper = Copper()
    
    # Get properties at a specific temperature
    T = Var(400, "K")
    density = copper.rho(T)
    specific_heat = copper.cp(T)
    conductivity = copper.k(T)
    
    print(density)        # e.g., 8900.0 [kg/m3]
    print(specific_heat)  # e.g., 385.0 [J/kg-K]
    print(conductivity)   # e.g., 398.0 [W/m-K]

Using Fluid Properties
^^^^^^^^^^^^^^^^^^^^^^

Fluids offer additional properties like viscosity, enthalpy, and Prandtl number. Example with saturated water:

.. code-block:: python

    from antupy.utils.props import SaturatedWater
    from antupy import Var

    # Create fluid instance
    water = SaturatedWater()
    
    # Get properties at saturation conditions
    T = Var(373.15, "K")  # 100°C
    
    # Thermophysical properties
    density = water.rho(T)
    specific_heat = water.cp(T)
    viscosity = water.mu(T)
    thermal_conductivity = water.k(T)
    
    # Dimensionless numbers
    prandtl = (specific_heat * viscosity / thermal_conductivity)
    
    print(f"Density: {density:.1f}")
    print(f"Enthalpy: {specific_heat:.1f}")
    print(f"Prandtl number: {prandtl:.3f}")


For complete API documentation of all available materials and fluids, see the :doc:`api` reference.

The ``htc`` library
-----------------------

.. warning::
	This HTC module documentation reflects an incomplete initial version. Several functions are placeholders and important correlations and use-cases still need to be implemented and documented in future expansions.

The ``htc`` module provides simple helpers to estimate radiative and convective heat transfer terms. It includes a sky temperature approximation and basic correlations for natural and external forced convection. Results use SI units; where returned as :py:class:`~antupy.Var`, the unit is ``W/m2-K``. The available functions are:

- :py:func:`antupy.utils.htc.temp_sky_simplest` — Approximate sky temperature as ``T_sky = T_amb - 15 K``.
- :py:func:`antupy.utils.htc.h_horizontal_surface_upper_hot` — Natural convection on an upper hot horizontal plate (Holman or Nellis-Klein correlations). Returns ``h`` as a float ``[W/m2-K]``.
- :py:func:`antupy.utils.htc.h_ext_flat_plate` — External forced convection over a flat plate (transition-aware). Returns a :py:class:`~antupy.Var` with unit ``W/m2-K``.

Examples of usage:

.. code-block:: python

	# Sky temperature approximation
	from antupy.utils.htc import temp_sky_simplest
	T_amb = 293.15  # K
	T_sky = temp_sky_simplest(T_amb)
	print(f"Sky temperature: {T_sky:.2f} K")

	# Natural convection: upper hot horizontal surface (air)
	from antupy.utils.htc import h_horizontal_surface_upper_hot
	h_nat = h_horizontal_surface_upper_hot(T_s=333.15, T_inf=293.15, L=0.5, correlation="NellisKlein")
	print(f"Natural convection h: {h_nat:.2f} W/m2-K")

	# External forced convection over a flat plate (water)
	from antupy.utils.htc import h_ext_flat_plate
	from antupy import Var
	from antupy.utils.props import Water

	h_forced = h_ext_flat_plate(
		temp_surf=Var(300, "K"),
		temp_fluid=Var(350, "K"),
		length=Var(1.0, "m"),
		u_inf=Var(2.0, "m/s"),
		fluid=Water()
	)
	print(h_forced)            # -> e.g., 123.4 [W/m2-K]
	print(h_forced.gv("W/m2-K"))  # numeric value only

