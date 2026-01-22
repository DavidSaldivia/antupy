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

    from antupy.props import Copper
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

Fluids offer additional properties like viscosity. Example with saturated water:

.. code-block:: python

    from antupy.props import SaturatedWater
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
	This a very limited initial version of this module. Only a couple of correlations are presented as example. It is expected to implemented a wider range of correlations in future expansions.

The ``htc`` module provides a collection of functions to estimate convective heat transfer coefficients. The source are diverse of heat transfer textbooks. It includes other functions for useful correlations (such as sky temperature approximation). Results are in SI units, returned as :py:class:`~antupy.Var`, with its unit as ``W/m2-K``. The available functions so far are:

- :py:func:`antupy.htc.temp_sky_simplest` — Approximate sky temperature as ``T_sky = T_amb - 15 K``.
- :py:func:`antupy.htc.h_horizontal_surface_upper_hot` — Natural convection on an upper hot horizontal plate (Holman or Nellis-Klein correlations).
- :py:func:`antupy.htc.h_ext_flat_plate` — External forced convection over a flat plate (transition-aware).

Examples of usage:

.. code-block:: python
	# External forced convection over a flat plate (water)
	from antupy import Var
	from antupy.htc import h_ext_flat_plate
	from antupy.props import Water

	h_forced = h_ext_flat_plate(
		temp_surf=Var(300, "K"),
		temp_fluid=Var(350, "K"),
		length=Var(1.0, "m"),
		u_inf=Var(2.0, "m/s"),
		fluid=Water()
	)
	print(h_forced)            # -> e.g., 10.4 [W/m2-K]
	print(h_forced.gv("W/m2-K"))  # numeric value only

