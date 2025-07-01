from dataclasses import dataclass
from antupy.units import Variable
from typing import Protocol

import numpy as np

class Fluid(Protocol):
    def rho (self, T: float|Variable) -> Variable:
        ...
    def cp  (self, T: float|Variable) -> Variable:
        ...
    def k (self, T: float|Variable) -> Variable:
        ...

class Material(Protocol):
    def rho (self, T: float|Variable) -> Variable:
        ...
    def cp  (self, T: float|Variable) -> Variable:
        ...
    def k (self, T: float|Variable) -> Variable:
        ...


@dataclass
class SolarSalt(Fluid):
    def rho(self, T: float|Variable) -> Variable:
        return Variable(1900., "kg/m3")
    
    def cp(self, T: float|Variable) -> Variable:
        return Variable(1100., "J/kg-K")
    
    def k(self, T: float | Variable) -> Variable:
        return Variable(0.55, "W/m-K")

    def __repr__(self) -> str:
        return "Solar salt (NaNO3-KNO3 mixture)"


class Carbo():
    def rho(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        return Variable(1810, "kg/m3")
    
    def cp(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        return Variable(148 * temp**0.3093, "J/kg-K")

    def k(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        return Variable(0.7, "W/m-K")

    def absortivity(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        return Variable(0.91, "-")

    def emissivity(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        return Variable(0.85, "-")    
    

class Aluminium():
    def rho(self, T: float|Variable) -> Variable:
        return Variable(2698.4, "kg/m3")
    
    def cp(self, T: float|Variable) -> Variable:
        return Variable(900., "J/kg-K")
    
    def k(self, T: float|Variable) -> Variable:
        return Variable(237., "W/m-K")


class Copper():
    def rho(self, T: float|Variable) -> Variable:
        return Variable(8960., "kg/m3")
    
    def cp(self, T: float|Variable) -> Variable:
        return Variable(385., "J/kg-K")
    
    def k(self, T: float|Variable) -> Variable:
        return Variable(401., "W/m-K")


class CopperNickel():
    def rho(self, T: float|Variable) -> Variable:
        return Variable(8900., "kg/m3")
    
    def cp(self, T: float|Variable) -> Variable:
        return Variable(376.6, "J/kg-K")
    
    def k(self, T: float|Variable) -> Variable:
        return Variable(50.2, "W/m-K")


class StainlessSteel():
    def rho(self, T: float|Variable) -> Variable:
        return Variable(7850., "kg/m3")
    
    def cp(self, T: float|Variable) -> Variable:
        return Variable(510., "J/kg-K")
    
    def k(self, T: float|Variable) -> Variable:
        return Variable(15., "W/m-K")


class Glass():
    def rho(self, T: float|Variable) -> Variable:
        return Variable(2490., "kg/m3")
    
    def cp(self, T: float|Variable) -> Variable:
        return Variable(837.4, "J/kg-K")
    
    def k(self, T: float|Variable) -> Variable:
        return Variable(0.8374, "W/m-K")
    
    def absortivity(self, T: float|Variable) -> Variable:
        return Variable(0.02, "-")
    
    def emissivity(self, T: float|Variable) -> Variable:
        return Variable(0.86, "-")
    
    def transmisivity(self, T: float|Variable) -> Variable:
        return Variable(0.935, "-")


class SaturatedWater():
    def rho(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (9.999e2, 2.034e-2, -6.162e-3, 2.261e-5, -4.657e-8)
        aux = sum([
            A[i]*temp**i for i in range(len(A))
        ])
        return Variable(aux, "kg/m3")
    
    def cp(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = 8.15599e3 - 2.80627e1*temp + 5.11283e-2*temp**2 - 2.17582e-13*temp**6
        return Variable(aux, "J/kg-K")
    
    def k(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (0.80201, -0.25992, 0.10024, -0.032005)
        B = (-0.32, -5.7, -12.0, -15.0)
        aux = sum([
            A[i]*(temp/300.)**B[i] for i in range(len(A))
        ])
        return Variable(aux, "W/m-K")
    
    def viscosity(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = 4.2844e-5 + 1 / ( 0.157*(temp+64.994)**2 - 91.296 )
        return Variable(aux, "Pa-s")
    
    def surface_tension(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = temp / 647.096
        return Variable(
            0.2358 * (1 - aux)**1.256 * (1 - 0.625*aux),
            "N/m"
        )
    
    def latent_heat(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        return Variable(
            (2.501e6 - 2.369e3*temp + 2.678e-1*temp**2 - 8.103e-3*temp**3 - 2.079e-5*temp**4) / 1000.,
            "kJ/kg"
        )
    
    def saturation_pressure(
            self,
            T: float|Variable = Variable(273.15, "K")
    ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
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
        return Variable(Pc * np.exp(aux * (Tc/temp - 1 )), "kPa")

    def vapor_pressure(
            self,
            T: float|Variable = Variable(273.15, "K")
    ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = [-5800., 1.391, -4.846e-2, 4.176e-5, -1.445e-8, 6.545]
        aux = (
            sum([ A[i] * temp**(i-1) for i in np.arange(4) ])
            + A[-1] * np.log(temp)
        )
        return Variable(np.exp(aux) / 1000., "kPa")
    
    def saturation_temperature(
            self,
            P: float|Variable = Variable(101.325, "kPa")
    ) -> Variable:
        if isinstance(P, Variable):
            pressure = P.u("kPa")
        elif isinstance(P, (int, float)):
            pressure = P
        return Variable(
            42.6776 - 3892.7 / (np.log(pressure/1000) - 9.48654) - 273.15,
            "K"
        )

class SaturatedVapor():
    def rho(
            self,
            T: float|Variable = Variable(273.15, "K")
    ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (-4.062329056, 0.10277044, -9.76300388e-4,
             4.475240795e-6, -1.004596894e-8, 8.9154895e-12)
        aux = sum([
            A[i] * temp**i for i in range(len(A))
        ])
        return Variable(aux, "kg/m3")

    def cp(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        aux = (1.3605e3 + 2.31334*temp - 2.46784e-10*temp**5 + 5.91332e-13*temp**6)
        return Variable(aux, "J/kg-K")
    
    def k(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (1.3046e-2, -3.756191e-5, 2.217964e-7, -1.111562e-10)
        aux = sum([
            A[i] * temp**i for i in range(len(A))
        ])
        return Variable(aux, "W/m-K")

    def viscosity(
            self,
            T: float|Variable = Variable(273.15, "K")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("K")
        elif isinstance(T, (int, float)):
            temp = T
        A = (2.562435e-6, 1.816683e-8, 2.579066e-11, -1.067299e-14)
        aux = sum([
            A[i] * temp**i for i in range(len(A))
        ])
        return Variable(aux, "Pa-s")
    

class CompressedWater():
    ...

class SeaWater():
    def rho(
            self,
            T: float|Variable = Variable(273.15, "K"),
            X: float|Variable = Variable(35000, "ppm")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Variable):
            salinity = X.u("ppm")
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
        aux = sum([
            A[i] * F[i] for i in range(len(A))
        ])
        return Variable(aux * 1e3, "kg/m3")
    
    def cp(
            self,
            T: float|Variable = Variable(273.15, "K"),
            X: float|Variable = Variable(35000, "ppm")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Variable):
            salinity = X.u("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        s = salinity / 1000.
        A = 4206.8 - 6.6197*s + 1.2288e-2*s**2
        B = -1.1262 + 5.4178e-2*s - 2.2719e-4*s**2
        C = 1.2026e-2 - 5.3566e-4*s + 1.8906e-6*s**2
        D = 6.8777e-7 + 1.517e-6*s - 4.4268e-9*s**2
        aux = A + B*temp + C*temp**2 + D*temp**3
        return Variable(aux, "J/kg-K")

    def k(
            self,
            T: float|Variable = Variable(273.15, "K"),
            X: float|Variable = Variable(35000, "ppm")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Variable):
            salinity = X.u("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        s = salinity / 1000.
        A = 2e-4
        B = 3.7e-2
        C = 3e-2
        aux = (1 - temp/( 647.3 + C*s ))**(1./3.)
        aux = aux * 0.434 * ( 2.3 - ( 343.5 + B*s )/temp)
        aux = - 6 + aux + np.log10(240 + A*s)
        return Variable(10.**aux * 1000., "W/m-K")
    
    def viscosity(
            self,
            T: float|Variable = Variable(273.15, "K"),
            X: float|Variable = Variable(35000, "ppm")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Variable):
            salinity = X.u("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        s = salinity / 1000.
        A = 1.474e-3 + 1.5e-5*temp - 3.927e-8*temp**2
        B = 1.0734e-5 - 8.5e-8*temp + 2.23e-10*temp**2
        mur = 1 + A*s + B*s**2
        muw = np.exp( -3.79418 + 604.129 / (139.18 + temp) )
        return Variable(mur*muw*1e-3, "Pa-s")
    
    def surface_tension(
            self,
            T: float|Variable = Variable(273.15, "K"),
            X: float|Variable = Variable(35000, "ppm")
        ) -> Variable:
        if isinstance(T, Variable):
            temp = T.u("C")
        elif isinstance(T, (int, float)):
            temp = T
        if isinstance(X, Variable):
            salinity = X.u("ppm")
        elif isinstance(X, (int, float)):
            salinity = X
        surface_tension_l = SaturatedWater().surface_tension(T).u("N/m")
        s = salinity / 1000.
        if (temp>40.):
            aux = surface_tension_l
        else:
            aux = surface_tension_l * (1 + (2.26e-4*temp + 9.46e-3) * np.log(1 + 3.31e-2*s) )
        return Variable(aux, "N/m")

class DryAir():
    ...

class HumidAir():
    ...

class TherminolVP1():
    ...

class Syltherm800():
    ...
