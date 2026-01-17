System Classes
==============

Antupy provides ``Plant`` and ``Parametric`` classes to build and analyze thermal systems. The :py:class:`~antupy.Plant` class is the main simulation container where you define components and their interactions in the ``run_simulation()`` method. The :py:class:`~antupy.Parametric` class allows running multiple simulations while varying one or more parameters to perform parametric studies.

Using the Plant class
---------------------

The :py:class:`~antupy.Plant` class lets you build system simulations by defining components as class attributes and implementing the ``run_simulation()`` method. Here's a simple solar water heating system:

.. code-block:: python

    from dataclasses import dataclass
    from antupy import SimulationOutput, Var, Plant
    from antupy.tsg.weather import TMY
    from antupy.utils.props import Water

    @dataclass
    class SolarCollector():
        area = Var(2, "m2")
        Fr_ta = Var(0.7, "-")
        Fr_Ul = Var(0.5, "W/m2-K")
        
        def run_model(self, temp_amb: Var, temp_in: Var, solar_rad: Var):
            x = (temp_in - temp_amb) / solar_rad if solar_rad.gv("W/m2") > 0 else Var(0,"W/m2")
            eta_out = self.Fr_ta - self.Fr_Ul * x
            rho, cp = Water().rho(temp_in), Water().cp(temp_in)
            temp_out = temp_in + eta_out * solar_rad / (self.area * cp * rho)
            return {"temp_out": temp_out, "eta_out": eta_out}

    @dataclass
    class HotWaterTank():
        volume = Var(200, "L")
        U = Var(0.4, "W/m2-K")
        
        def run_model(self):
            return {}

    class SolarWaterHeating(Plant):
        heater = SolarCollector()
        tank = HotWaterTank()
        weather = TMY(location="Antofagasta")
        
        def run_simulation(self, verbose: bool = False) -> SimulationOutput:
            df_sim = self.weather.load_data()
            for _, row in df_sim.iterrows():
                out_collector = self.heater.run_model(
                    row["temp_amb"], row["temp_in"], row["GHI"]
                )
                out_tank = self.tank.run_model()
            return super().run_simulation(verbose)

    # Run the simulation
    plant = SolarWaterHeating()
    results = plant.run_simulation()

Using the Parametric class
---------------------------

The :py:class:`~antupy.Parametric` class runs multiple simulations varying one or more parameters. You provide a base case plant and a dictionary of parameter ranges as :py:class:`~antupy.Array` objects:

.. code-block:: python

    import numpy as np
    from antupy import Array, Var, Parametric

    # Define base case
    base_case = SolarWaterHeating()
    
    # Define parameter ranges
    params_in = {
        "heater.area": Array(np.arange(1.5, 3.1, 0.5), "m2"),
        "heater.Fr_ta": Array(np.linspace(0.6, 0.8, 3), "-"),
    }
    
    # Create and run parametric study
    study = Parametric(
        base_case=base_case,
        params_in=params_in,
        save_results_detailed=True,
        dir_output="./parametric_results",
        path_results="./parametric_results/results.csv",
    )
    
    df_results = study.run_analysis()
    print(df_results.head())

The parametric study runs all combinations of the input parameters and saves results to CSV files. The ``save_results_detailed=True`` option stores individual simulation outputs for further analysis.
