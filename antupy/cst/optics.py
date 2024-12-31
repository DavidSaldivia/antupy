from dataclasses import dataclass
from antupy.protocols import Model

@dataclass
class SolarField(Model):
    # Characteristics of Solar Field
    eta_sfr = 0.97*0.95*0.95           # Solar field reflectivity
    eta_rfl = 0.95                     # Includes mirror refl, soiling and refl. surf. ratio. Used for HB and CPC
    A_h1    = 2.97**2                  # Area of one heliostat
    N_pan   = 1                        # Number of panels per heliostat
    err_x   = 0.001                    # [rad] Reflected error mirror in X direction
    err_y   = 0.001                    # [rad] Reflected error mirror in X direction
    pass

    def 



@dataclass
class BeamDownReceiver(Model):
    # Characteristics of BDR and Tower
    zf = 50.               # Focal point (where the rays are pointing originally) will be (0,0,zf)
    fzv      = 0.83              # Position of HB vertix (fraction of zf)
    eta_hbi  = 0.95              # Desired hbi efficiency
    
    Type     = 'PB'              # Type of TOD
    Array    = 'A'               # Array of polygonal TODs
    xrc      = 0.                # Second focal point (TOD & receiver)
    yrc      = 0.                # Second focal point (TOD & receiver)
    fzc      = 0.20              # Second focal point (Height of TOD Aperture, fraction of zf)
    
    Q_av     = 0.5               # [MW/m2] Desired average radiation flux on receiver
    Q_mx     = 2.0               # [MW/m2] Maximum radiation flux on receiver
    
    # if 'zrc' in CST:                            # Second focal point (CPC receiver)
    #     CST['fzc'] = CST['zrc']/ CST['zf']
    # else:
    #     CST['zrc']  = CST['fzc']*CST['zf']
    
    # if 'zv'  in CST:                            # Hyperboloid vertix height
    #     CST['fzv'] = CST['zv'] / CST['zf']
    # else:
    #     CST['zv']   = CST['fzv']*CST['zf']
    pass

@dataclass
class HyperboloidMirror(Model):
    pass

@dataclass
class CPC(Model):
    pass

@dataclass
class ParaboloidMirror(Model):
    pass

@dataclass
class PowerBlock(Model):
    # Receiver and Power Block
    P_el    = 10.0               #[MW] Target for Net Electrical Power
    eta_pb  = 0.50               #[-] Power Block efficiency target 
    eta_sg  = 0.95               #[-] Storage efficiency target
    eta_rcv = 0.75               #[-] Receiver efficiency target


@dataclass
class LocWeather(Model):
    # Environment conditions
    Gbn   = 950                # Design-point DNI [W/m2]
    day   = 80                 # Design-point day [-]
    omega = 0.0                # Design-point hour angle [rad]
    lat   = -23.               # Latitude [°]
    lng   = 115.9              # Longitude [°]
    T_amb = 300.               # Ambient Temperature [K]



def CST_BaseCase_Thesis(**kwargs):
    """
    Subroutine to create a dictionary with the main parameters of a BDR CST plant.
    The parameters here are the default ones. Anyone can be changed if are sent as variable.
    e.g. CST_BaseCase(P_el=15) will create a basecase CST but with P_el=15MWe
    
    Parameters
    ----------
    **kwargs : the parameters that will be different than the basecase

    Returns
    -------
    CST : Dictionary with all the parameters of BDR-CST plant.

    """
    CST = dict()
    
    ############### STANDARD CASE ####################
    ##################################################
    
    
    ##### CHANGING SPECIFIC VARIABLES ###########
    for key, value in kwargs.items():
        CST[key] = value
        
        
    if 'P_SF' in CST:                           #[MW] Required power energy
        CST['P_el'] = CST['P_SF'] * ( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )
    else:
        CST['P_SF'] = CST['P_el']/( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )


    return CST

# TODO: move the keys on these dictionaries as classes attributes
def CST_BaseCase_Paper(**kwargs):
    """
    Subroutine to create a dictionary with the main parameters of a BDR CST plant.
    The parameters here are the default ones. Anyone can be changed if are sent as variable.
    e.g. CST_BaseCase(P_el=15) will create a basecase CST but with P_el=15MWe
    
    Parameters
    ----------
    **kwargs : the parameters that will be different than the basecase

    Returns
    -------
    CST : Dictionary with all the parameters of BDR-CST plant.

    """
    CST = dict()
    
    ############### STANDARD CASE ####################
    ##################################################
    # Environment conditions
    CST['Gbn']   = 950                # Design-point DNI [W/m2]
    CST['day']   = 80                 # Design-point day [-]
    CST['omega'] = 0.0                # Design-point hour angle [rad]
    CST['lat']   = -23.               # Latitude [°]
    CST['lng']   = 115.9              # Longitude [°]
    CST['T_amb'] = 300.               # Ambient Temperature [K]
    ##################################################
    # Receiver and Power Block
    CST['P_el']    = 10.0               #[MW] Target for Net Electrical Power
    CST['eta_pb']  = 0.50               #[-] Power Block efficiency target 
    CST['eta_sg']  = 0.95               #[-] Storage efficiency target
    CST['eta_rcv'] = 0.75               #[-] Receiver efficiency target
    ##################################################
    # Characteristics of Solar Field
    CST['eta_sfr'] = 0.97*0.95*0.95                # Solar field reflectivity
    CST['eta_rfl'] = 0.95                          # Includes mirror refl, soiling and refl. surf. ratio. Used for HB and CPC
    CST['A_h1']    = 7.07*7.07                     # Area of one heliostat
    CST['N_pan']   = 16                            # Number of panels per heliostat
    CST['err_x']   = 0.001                    # [rad] Reflected error mirror in X direction
    CST['err_y']   = 0.001                    # [rad] Reflected error mirror in X direction
    
    ##################################################
    # Characteristics of BDR and Tower
    CST['zf']       = 50.               # Focal point (where the rays are pointing originally) will be (0,0,zf)
    CST['fzv']      = 0.83              # Position of HB vertix (fraction of zf)
    CST['eta_hbi']  = 0.95              # Desired hbi efficiency
    
    CST['Type']     = 'PB'              # Type of TOD
    CST['Array']    = 'A'               # Array of polygonal TODs
    CST['xrc']      = 0.                # Second focal point (TOD & receiver)
    CST['yrc']      = 0.                # Second focal point (TOD & receiver)
    CST['fzc']      = 0.20              # Second focal point (Height of TOD Aperture, fraction of zf)
    
    CST['Q_av']     = 0.5               # [MW/m2] Desired average radiation flux on receiver
    CST['Q_mx']     = 2.0               # [MW/m2] Maximum radiation flux on receiver
    
    ##### CHANGING SPECIFIC VARIABLES ###########
    for key, value in kwargs.items():
        CST[key] = value
        
    ####### Variables from calculations #########
    if 'zrc' in CST:                            # Second focal point (CPC receiver)
        CST['fzc'] = CST['zrc']/ CST['zf']
    else:
        CST['zrc']  = CST['fzc']*CST['zf']
    
    if 'zv'  in CST:                            # Hyperboloid vertix height
        CST['fzv'] = CST['zv'] / CST['zf']
    else:
        CST['zv']   = CST['fzv']*CST['zf']
        
    if 'P_SF' in CST:                           #[MW] Required power energy
        CST['P_el'] = CST['P_SF'] * ( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )
    else:
        CST['P_SF'] = CST['P_el']/( CST['eta_pb']*CST['eta_sg']*CST['eta_rcv'] )


    return CST
