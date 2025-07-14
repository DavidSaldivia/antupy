from antupy.units import Variable, conversion_factor    

def test_conversion_factor():
    assert conversion_factor("m3/s", "L/min") == 60000.
    assert conversion_factor("W", "kJ/hr") == 3.6

def test_conversion_temp():
    temp1 = Variable(300., "K")
    temp2 = Variable(20., "C")
    assert temp1.u("C") == 300-273.15
    assert temp1.u("K") == 300.
    assert temp2.u("C") == 20.
    assert temp2.u("K") == 20+273.15

def test_conversion_time():
    time_sim = Variable(365, "d")
    assert time_sim.u('d') == 365
    assert time_sim.u('hr') == 8760.
    assert time_sim.u('s') == 31536000.0

def test_add_variables():
    time_sim1 = Variable(365,"d")
    time_sim2 = Variable(1, "d")
    assert (time_sim1 + time_sim2).u("d") == 366

def test_mul_variables():
    time_sim = Variable(1, "d")
    time_sim.set_unit("hr")
    nom_power = Variable(100, "kW")
    nom_power.set_unit("kJ/hr")
    energy = nom_power * time_sim

    assert energy.u("kJ") == 8640000.0
    assert (8*energy/2 - energy*2).u("kWh") == 4800.