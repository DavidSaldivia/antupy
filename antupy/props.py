from dataclasses import dataclass
from antupy.core import Var
from typing import Protocol

import cantera as ct
import numpy as np

class Fluid(Protocol):
    def rho (self, T: float|Var) -> Var:
        ...
    def cp  (self, T: float|Var) -> Var:
        ...
    def k (self, T: float|Var) -> Var:
        ...

class Material(Protocol):
    def rho (self, T: float|Var) -> Var:
        ...
    def cp  (self, T: float|Var) -> Var:
        ...
    def k (self, T: float|Var) -> Var:
        ...


@dataclass
class SolarSalt(Fluid):
    def rho(self, T: float|Var) -> Var:
        return Var(1900., "kg/m3")
    
    def cp(self, T: float|Var) -> Var:
        return Var(1100., "J/kg-K")
    
    def k(self, T: float | Var) -> Var:
        return Var(0.55, "W/m-K")

    def __repr__(self) -> str:
        return "Solar salt (NaNO3-KNO3 mixture)"


class Carbo():
    def rho(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        return Var(1810, "kg/m3")
    
    def cp(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        return Var(148 * temp**0.3093, "J/kg-K")

    def k(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        return Var(0.7, "W/m-K")

    def absortivity(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        return Var(0.91, "-")

    def emissivity(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        return Var(0.85, "-")    
    

class Aluminium():
    def rho(self, T: float|Var) -> Var:
        return Var(2698.4, "kg/m3")
    
    def cp(self, T: float|Var) -> Var:
        return Var(900., "J/kg-K")
    
    def k(self, T: float|Var) -> Var:
        return Var(237., "W/m-K")


class Copper():
    def rho(self, T: float|Var) -> Var:
        return Var(8960., "kg/m3")
    
    def cp(self, T: float|Var) -> Var:
        return Var(385., "J/kg-K")
    
    def k(self, T: float|Var) -> Var:
        return Var(401., "W/m-K")


class CopperNickel():
    def rho(self, T: float|Var) -> Var:
        return Var(8900., "kg/m3")
    
    def cp(self, T: float|Var) -> Var:
        return Var(376.6, "J/kg-K")
    
    def k(self, T: float|Var) -> Var:
        return Var(50.2, "W/m-K")


class StainlessSteel():
    def rho(self, T: float|Var) -> Var:
        return Var(7850., "kg/m3")
    
    def cp(self, T: float|Var) -> Var:
        return Var(510., "J/kg-K")
    
    def k(self, T: float|Var) -> Var:
        return Var(15., "W/m-K")


class Glass():
    def rho(self, T: float|Var) -> Var:
        return Var(2490., "kg/m3")
    
    def cp(self, T: float|Var) -> Var:
        return Var(837.4, "J/kg-K")
    
    def k(self, T: float|Var) -> Var:
        return Var(0.8374, "W/m-K")
    
    def absortivity(self, T: float|Var) -> Var:
        return Var(0.02, "-")
    
    def emissivity(self, T: float|Var) -> Var:
        return Var(0.86, "-")
    
    def transmisivity(self, T: float|Var) -> Var:
        return Var(0.935, "-")


class SaturatedWater():
    def rho(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("°C")
        elif isinstance(T, (int, float)):
            temp = T
        A = (9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8)
        aux = sum([A[i]*temp**i for i in range(len(A))])
        return Var(aux, "kg/m3")
    
    def cp(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = 8.15599e3 - 2.80627e1*temp + 5.11283e-2*temp**2 - 2.17582e-13*temp**6
        return Var(aux, "J/kg-K")
    
    def k(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (0.80201, -0.25992, 0.10024, -0.032005)
        B = (-0.32, -5.7, -12.0, -15.0)
        aux = sum([
            A[i]*(temp/300.)**B[i] for i in range(len(A))
        ])
        return Var(aux, "W/m-K")
    
    def viscosity(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = 4.2844e-5 + 1 / ( 0.157*(temp+64.994)**2 - 91.296 )
        return Var(aux, "Pa-s")
    
    def surface_tension(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = temp / 647.096
        return Var(
            0.2358 * (1 - aux)**1.256 * (1 - 0.625*aux),
            "N/m"
        )
    
    def latent_heat(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (2.501e6, -2.369e3, 2.678e-1, -8.103e-3, -2.079e-5)
        aux = sum([ A[i] * temp**i for i in range(len(A)) ])
        return Var(aux / 1000.,"kJ/kg")
    
    def saturation_pressure(
            self,
            T: float|Var = Var(273.15, "K")
    ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        Pc = 22089.
        Tc = 647.286
        factors = (-7.419242, 0.29721, -0.1155286,
                   0.008685635, 0.001094098, -0.00439993,
                   0.002520658, -0.000521868)
        aux = sum([
            Fi*(0.01*(temp-338.15))**i 
            for (i,Fi) in enumerate(factors)
        ])
        return Var(Pc * np.exp(aux * (Tc/temp - 1 )), "kPa")

    def vapor_pressure(
            self,
            T: float|Var = Var(273.15, "K")
    ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = [-5800., 1.391, -4.846e-2, 4.176e-5, -1.445e-8, 6.545]
        aux = (
            sum([ A[i] * temp**(i-1) for i in np.arange(4) ])
            + A[-1] * np.log(temp)
        )
        return Var(np.exp(aux) / 1000., "kPa")
    
    def saturation_temperature(
            self,
            P: float|Var = Var(101.325, "kPa")
    ) -> Var:
        if isinstance(P, Var):
            pressure = P.gv("kPa")
        elif isinstance(P, (int, float)):
            pressure = P
        return Var(
            42.6776 - 3892.7 / (np.log(pressure/1000) - 9.48654) - 273.15,
            "K"
        )

class SaturatedSteam():
    def rho(
            self,
            T: float|Var = Var(273.15, "K")
    ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("°C")
        elif isinstance(T, (int, float)):
            temp = T
        A = (-4.062329056, 0.10277044, -9.76300388e-4,
             4.475240795e-6, -1.004596894e-8, 8.9154895e-12)
        return Var(sum([A[i] * temp**i for i in range(len(A))]), "kg/m3")

    def cp(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = (1.3605e3 + 2.31334*temp - 2.46784e-10*temp**5 + 5.91332e-13*temp**6)
        return Var(aux, "J/kg-K")
    
    def k(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (1.3046e-2, -3.756191e-5, 2.217964e-7, -1.111562e-10)
        aux = sum([ A[i] * temp**i for i in range(len(A)) ])
        return Var(aux, "W/m-K")

    def viscosity(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (2.562435e-6, 1.816683e-8, 2.579066e-11, -1.067299e-14)
        aux = sum([ A[i] * temp**i for i in range(len(A)) ])
        return Var(aux, "Pa-s")
    

class CompressedWater():
    ...

class SeaWater():
    def rho(
            self,
            T: float|Var = Var(273.15, "K"),
            X: float|Var = Var(35000, "ppm")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Var):
            salinity = X.gv("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        A1 = (2*temp - 200) / 160.
        B1 = (2*salinity/1000 - 150) / 150.
        G = (0.5, B1, 2*B1**2-1)
        F = (0.5, A1, 2*A1**2-1, 4*A1**3-3*A1)
        A = (
            4.032219*G[0] + 0.115313*G[1] + 3.26e-4*G[2],
            -0.108199*G[0] + 1.571e-3*G[1] + 4.23e-4*G[2],
            -0.012247*G[0] + 1.74e-3*G[1] + 9e-6*G[2],
            6.92e-4*G[0] - 8.7e-5*G[1] - 5.3e-5*G[2]
        )
        aux = sum([ A[i] * F[i] for i in range(len(A)) ])
        return Var(aux * 1e3, "kg/m3")
    
    def cp(
            self,
            T: float|Var = Var(273.15, "K"),
            X: float|Var = Var(35000, "ppm")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Var):
            salinity = X.gv("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        s = salinity / 1000.
        A = 4206.8 - 6.6197*s + 1.2288e-2*s**2
        B = -1.1262 + 5.4178e-2*s - 2.2719e-4*s**2
        C = 1.2026e-2 - 5.3566e-4*s + 1.8906e-6*s**2
        D = 6.8777e-7 + 1.517e-6*s - 4.4268e-9*s**2
        aux = A + B*temp + C*temp**2 + D*temp**3
        return Var(aux, "J/kg-K")

    def k(
            self,
            T: float|Var = Var(273.15, "K"),
            X: float|Var = Var(35000, "ppm")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Var):
            salinity = X.gv("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        s = salinity / 1000.
        A = 2e-4
        B = 3.7e-2
        C = 3e-2
        aux = (1 - temp/( 647.3 + C*s ))**(1/3)
        aux = aux * 0.434 * ( 2.3 - ( 343.5 + B*s )/temp)
        aux = - 6 + aux + np.log10(240 + A*s)
        return Var(10.**aux * 1000., "W/m-K")
    
    def viscosity(
            self,
            T: float|Var = Var(273.15, "K"),
            X: float|Var = Var(35000, "ppm")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Var):
            salinity = X.gv("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        s = salinity / 1000.
        A = 1.474e-3 + 1.5e-5*temp - 3.927e-8*temp**2
        B = 1.0734e-5 - 8.5e-8*temp + 2.23e-10*temp**2
        mur = 1 + A*s + B*s**2
        muw = np.exp( -3.79418 + 604.129 / (139.18 + temp) )
        return Var(mur*muw*1e-3, "Pa-s")
    
    def surface_tension(
            self,
            T: float|Var = Var(273.15, "K"),
            X: float|Var = Var(35000, "ppm")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Var):
            salinity = X.gv("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        surface_tension_l = SaturatedWater().surface_tension(T).gv("N/m")
        s = salinity / 1000.
        if (temp>40.):
            aux = surface_tension_l
        else:
            aux = surface_tension_l * (1 + (2.26e-4*temp + 9.46e-3) * np.log(1 + 3.31e-2*s) )
        return Var(aux, "N/m")

class DryAir():
    def rho(
            self,
            T: float|Var = Var(273.15, "K"),
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        temps = np.concatenate((
            [223,], np.arange(243, 374, 10), np.arange(423, 574, 50)
        ))
        values = np.array([1.582, 1.452, 1.394, 1.342,
                           1.292, 1.247, 1.204, 1.164,
                           1.127, 1.092, 1.060, 1.030,
                           1.000, 0.973, 0.946, 0.835,
                           0.746, 0.675, 0.616])
        return Var(float(np.interp(temp, temps, values)), "kg/m3")

    def cp(
            self,
            T: float|Var = Var(273.15, "K"),
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        temps = np.arange(250, 851, 50)
        values = np.array([1.003, 1.005, 1.008,
                           1.013, 1.020, 1.029,
                           1.040, 1.051, 1.063,
                           1.075, 1.087, 1.099,
                           1.121])
        return Var(float(np.interp(temp, temps, values)), "kJ/kg-K")

    def k(
            self,
            T: float|Var = Var(273.15, "K"),
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (-4.937787e-4, 1.018087e-4, -4.627937e-8, 1.250603e-11)
        aux = sum([ A[i] * temp**i for i in range(len(A)) ])
        return Var(aux, "W/m-K")

    def viscosity(
            self,
            T: float|Var = Var(273.15, "K"),
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        temps = np.concatenate((
            [223,], np.arange(243, 374, 10), np.arange(423, 574, 50)
        ))
        values = np.array([0.921, 1.08, 1.16, 1.24,
                           1.33, 1.42, 1.51, 1.60,
                           1.69, 1.79, 1.89, 1.99, 
                           2.09, 2.19, 2.30, 2.85,
                           3.45, 4.08, 4.75])
        return Var(
            float(np.interp(temp, temps, values)) * 1e-5,
            "m2/s"
        )
    
    def humidity(
            self,
            T: float|Var = Var(273.15, "K"),
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        temps = np.concatenate((
            np.arange(253, 324, 2), np.arange(328, 369, 5)
        ))
        values = np.array([ 0.63, 0.77, 0.93, 1.11,
                           1.34, 1.60, 1.91, 2.27,
                           2.69, 3.19, 3.78, 4.37,
                           5.03, 5.79, 6.65, 7.63, 
                           8.75, 9.97, 11.4, 12.9, 
                           14.7, 16.6, 18.8, 21.4, 
                           24.0, 27.2, 30.6, 34.4, 
                           38.8, 43.5, 48.8, 54.8, 
                           61.3, 68.9, 77.0, 86.2, 
                           114., 152., 204., 276.,
                           382., 545., 828., 1400., 3120.])
        return Var(float(np.interp(temp, temps, values))*1.e-3, "-")


class HumidAir():
    def rho(
            self,
            T: float|Var = Var(273.15, "K"),
            P: float|Var = Var(101.325, "kPa"),
            AH: float|Var = Var(0.001, "-")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        else:
            raise ValueError(f"{type(T)=} is not a valid type")
        if isinstance(P, Var):
            pressure = P.gv("Pa")
        elif isinstance(P, (int, float)):
            pressure = P
        else:
            raise ValueError(f"{type(P)=} is not a valid type")
        if isinstance(AH, Var):
            abshum = AH.gv("-")
        elif isinstance(AH, (int, float)):
            abshum = AH
        else:
            raise ValueError(f"{type(AH)=} is not a valid type")
        aux = (1 - abshum) * (1 - abshum / ( abshum + 0.62198 ) ) * pressure / (287.08 * temp)
        return Var(float(aux), "kg/m3")

    def cp(
            self,
            T: float|Var = Var(273.15, "K"),
            AH: float|Var = Var(0.001, "-")
        ) -> Var:
        if isinstance(AH, Var):
            abshum = AH.gv("-")
        elif isinstance(AH, (int, float)):
            abshum = AH
        else:
            raise TypeError("AH must be a Var or a number.")
        return DryAir().cp(T) + abshum * SaturatedSteam().cp(T)

    def k(
            self,
            T: float|Var = Var(273.15, "K"),
            AH: float|Var = Var(0.001, "-")
        ) -> Var:
        if isinstance(AH, Var):
            abshum = AH.gv("-")
        elif isinstance(AH, (int, float)):
            abshum = AH
        Ma = 28.97
        Mv = 18.016
        Xa = 1 / (1 + 1.608*abshum)
        Xv = abshum / (abshum + 0.622)
        k_a = DryAir().k(T)
        k_v = SaturatedSteam().k(T)
        return (Xa*Ma**0.33*k_a + Xv*Mv**0.33*k_v) / (Xa*Ma**0.33 + Xv*Mv**0.33)

    def viscosity(
            self,
            T: float|Var = Var(273.15, "K"),
            AH: float|Var = Var(0.001, "-")
        ) -> Var:
        if isinstance(AH, Var):
            abshum = AH.gv("-")
        elif isinstance(AH, (int, float)):
            abshum = AH
        Ma = 28.97
        Mv = 18.016
        Xa = 1 / (1 + 1.608*abshum)
        Xv = abshum / (abshum + 0.622)
        mu_a = DryAir().viscosity(T)
        mu_v = SaturatedSteam().viscosity(T)
        return (Xa*Ma**0.5*mu_a + Xv*Mv**0.5*mu_v) / (Xa*Ma**0.5 + Xv*Mv**0.5)

    def enthalpy(
            self,
            T: float|Var = Var(273.15, "K"),
            AH: float|Var = Var(0.001, "-")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("K")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(AH, Var):
            abshum = AH.gv("-")
        elif isinstance(AH, (int, float)):
            abshum = AH
        if temp < 273.15-50:
            aux = 1.005*temp + abshum * (2500.9 + 1.82*temp)
        else:
            aux = 1.005*temp + abshum * (2507.523 + 1.69*temp)

        return Var(aux*1000., "kJ/kg")

class CO2():
    _ct_solution = ct.Solution('gri30.yaml','gri30') # type: ignore

    def rho(self, temp: float, pressure: float) -> float:
        # Calculate specific heat capacity of CO2 at given temperature and pressure
        self._ct_solution.TPY = temp, pressure, 'CO2:1.00'
        return self._ct_solution.density_mass
    
    def cp(self, temp: float, pressure: float) -> float:
        # Calculate specific heat capacity of CO2 at given temperature and pressure
        self._ct_solution.TPY = temp, pressure, 'CO2:1.00'
        return self._ct_solution.cp
    
    def k(self, temp: float, pressure: float) -> float:
        # Calculate thermal conductivity of CO2 at given temperature and pressure
        self._ct_solution.TPY = temp, pressure, 'CO2:1.00'
        return self._ct_solution.thermal_conductivity
    
    def viscosity(self, temp: float, pressure: float) -> float:
        # Calculate viscosity of CO2 at given temperature and pressure
        self._ct_solution.TPY = temp, pressure, 'CO2:1.00'
        return self._ct_solution.viscosity


class TherminolVP1():
    def rho(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = (16, 38, 60, 82, 
                 104, 127, 149, 171, 
                 193, 216, 238, 257, 
                 271, 293, 316, 338, 
                 360, 382, 399, 416)
        values = (1068, 1050, 1032, 1014,
                  995, 977, 958, 939, 
                  919, 899, 879, 860,
                  847, 824, 800, 775,
                  749, 720, 696, 670)
        return Var(float(np.interp(temp, temps, values)), "kg/m3")

    def cp(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = (16, 38, 60, 82, 
                 104, 127, 149, 171, 
                 193, 216, 238, 257, 
                 271, 293, 316, 338, 
                 360, 382, 399, 416)
        values = (1530, 1600, 1660, 1730,
                  1790, 1850, 1910, 1970,
                  2030, 2090, 2150, 2200,
                  2240, 2300, 2360, 2420,
                  2480, 2560, 2620, 2700)
        return Var(np.interp(temp, temps, values), "J/kg-K")

    def k(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = (16, 38, 60, 82, 
                 104, 127, 149, 171, 
                 193, 216, 238, 257, 
                 271, 293, 316, 338, 
                 360, 382, 399, 416)
        values = (0.1367, 0.1346, 0.1323, 0.1298,
                  0.1271, 0.1243, 0.1213, 0.1181,
                  0.1148, 0.1113, 0.1076, 0.1043,
                  0.1018, 0.0977, 0.0934, 0.0890,
                  0.0844, 0.0796, 0.0759, 0.0721)
        return Var(np.interp(temp, temps, values), "W/m-K")
    
    def viscosity(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = (16, 38, 60, 82, 
                 104, 127, 149, 171, 
                 193, 216, 238, 257, 
                 271, 293, 316, 338, 
                 360, 382, 399, 416)
        values = (4.89, 2.73, 1.761, 1.244,
                  0.934, 0.731, 0.591, 0.490,
                  0.414, 0.355, 0.309, 0.276,
                  0.256, 0.229, 0.206, 0.1866,
                  0.1703, 0.1562, 0.1470, 0.1387)
        return Var(np.interp(temp, temps, values)*1e-6, "Pa-s")


class Syltherm800():
    def rho(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = np.arange(-40, 401, 40)
        values = (990.61, 953.16, 917.07, 881.68, 
                  846.35, 810.45, 773.33, 734.35, 
                  692.87, 648.24, 599.83, 547.00)
        return Var(np.interp(temp, temps, values), "kg/m3")

    def cp(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = np.arange(-40, 401, 40)
        values = (1506, 1574, 1643, 1711, 
                  1779, 1847, 1916, 1984, 
                  2052, 2121, 2189, 2257)
        return Var(np.interp(temp, temps, values), "J/kg-K")

    def k(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = np.arange(-40, 401, 40)
        values = (0.1463, 0.1388, 0.1312, 0.1237, 
                  0.1162, 0.1087, 0.1012, 0.0936, 
                  0.0861, 0.0786, 0.0711, 0.0635)
        return Var(np.interp(temp, temps, values), "W/m-K")
    
    def viscosity(
            self,
            T: float|Var = Var(273.15, "K")
        ) -> Var:
        if isinstance(T, Var):
            temp = T.gv("C")
        elif isinstance(T, (int, float)):
            temp = T - 273.15
        temps = np.arange(-40, 401, 40)
        values = (51.05, 15.33, 7.00, 3.86, 
                  2.36, 1.54, 1.05, 0.74, 
                  0.54, 0.41, 0.31, 0.25)
        return Var(np.interp(temp, temps, values)*1e-6, "Pa-s")