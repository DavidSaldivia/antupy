from dataclasses import dataclass
from antupy import SimulationOutput, Var, Plant, Parametric
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
        return {
            "temp_out": temp_out,
            "eta_out": eta_out,
        }

@dataclass
class HotWaterTank():
    volume = Var(200, "L")
    U = Var(0.4, "W/m2-K")
    def run_model(self,):
        out = {

        }
        return out

class SolarWaterHeating(Plant):
    heater = SolarCollector()
    tank = HotWaterTank()
    weather = TMY(location="Antofagasta", country="CL")
    
    def run_simulation(self, verbose: bool = False) -> SimulationOutput:
        for _, row in self.weather.load_data().iterrows():
            out_collector = self.heater.run_model(
                row["temp_amb"], row["temp_in"], row["GHI"]
            )
            out_tank = self.tank.run_model()
        return super().run_simulation(verbose)
