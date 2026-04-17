from dataclasses import dataclass
from antupy import SimulationOutput, Var, Plant, Parametric
from antupy.tsg.weather import TMY, HWD
from antupy.utils.props import Water

@dataclass
class SolarCollector():
    area = Var(2, "m2")
    Fr_ta = Var(0.7, "-")
    Fr_Ul = Var(0.5, "W/m2-K")
    fluid = Water()

    def run_model(self, temp_amb: Var, temp_in: Var, solar_rad: Var) -> SimulationOutput:
        x = (temp_in - temp_amb) / solar_rad if solar_rad > Var(0, "W/m2") else Var(0,"W/m2")
        eta_out = self.Fr_ta - self.Fr_Ul * x
        rho = self.fluid.rho(temp_in)
        cp = self.fluid.cp(temp_in)
        temp_out = temp_in + eta_out * solar_rad / (self.area * cp * rho)
        return {
            "temp_out": temp_out.su("K"),
            "eta_out": eta_out.su("-"),
        }

@dataclass
class HotWaterTank():
    volume = Var(200, "L")
    U = Var(0.4, "W/m2-K")
    area = Var(4.0, "m2")
    def run_model(self,) -> SimulationOutput:
        out = {

        }
        return out
    
@dataclass
class WaterPump():
    flow_rate = Var(0.1, "kg/s")

    def run_model(self,) -> SimulationOutput:
        out = {

        }
        return out

class SolarWaterHeating(Plant):
    heater = SolarCollector()
    tank = HotWaterTank()
    pump = WaterPump()
    weather = TMY(location="Antofagasta")
    hwd = HWD()
    
    def run_simulation(self, verbose: bool = False) -> SimulationOutput:
        df_weather = self.weather.load_data()

        for _, row in df_weather.iterrows():
            out_collector = self.heater.run_model(
                row["temp_amb"], row["temp_in"], row["GHI"]
            )
            out_tank = self.tank.run_model()
            out_pump = self.pump.run_model()
        return {}
