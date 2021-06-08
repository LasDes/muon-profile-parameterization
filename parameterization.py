"""
Calculate the longitudinal production profile of high-energy muons in air showers.

This module implements the parametrization of muon production described in the paper.
Given a primary cosmic ray energy, mass and zenith angle, a muon threshold energy and
an atmospheric temperature profile, the average longitudinal production profile
of muons above threshold (differential in slant depth) is obtained.
"""

import numpy as np

# values from Table 1 and 2 in the paper
parameters = {
        'IceCube' : {'Nmax' : {'q'  : 2.677,
                               'c1' : 0.124,
                               'p1' : 1.012,
                               'c2' : 0.244,
                               'p2' : 0.902},
                     'Xmax' : {'q'  : 3.117,
                               'a1' : 366.2,
                               'b1' : 139.5,
                               'a2' : 642.2,
                               'b2' :  51.0},
                     'lamb' : {'q'  : 2.074,
                               'a1' : 266.0,
                               'b1' :  42.1,
                               'a2' : 398.8,
                               'b2' : -21.9},
                     'X0'   : {'q'  : 4.025,
                               'a1' :  -2.9,
                               'b1' :  -2.6,
                               'a2' : -15.8,
                               'b2' :   0.6},
                     'f'    : {'q'  :  2.72,
                               'a1' :    1.,
                               'b1' :  0.53,
                               'a2' :  2.45,
                               'b2' :    0.},
                     'elbert' : {'K'      :  12.4,
                                 'alpha1' : 0.787,
                                 'alpha2' :  5.99}
                     },
        'NOvA'    : {'Nmax' : {'q'  : 2.557,
                               'c1' : 0.144,
                               'p1' : 0.972,
                               'c2' : 0.213,
                               'p2' : 0.905},
                     'Xmax' : {'q'  : 3.476,
                               'a1' : 260.9,
                               'b1' : 176.4,
                               'a2' : 665.1,
                               'b2' :  60.1},
                     'lamb' : {'q'  : 1.526,
                               'a1' : 289.4,
                               'b1' :  95.0,
                               'a2' : 483.0,
                               'b2' : -31.8},
                     'X0'   : {'q'  : 2.778,
                               'a1' : -28.7,
                               'b1' :  -2.3,
                               'a2' : -48.0,
                               'b2' :   4.6},
                     'f'    : {'q'  :  2.80,
                               'a1' :    1.,
                               'b1' :  0.69,
                               'a2' :  3.06,
                               'b2' : -0.05},
                     'elbert' : {'K'      : 6.034,
                                 'alpha1' :  0.80,
                                 'alpha2' :  5.99}
                     }
}

def GH_derivative(X, Nmax, Xmax, lamb, X0):
    """
    Derivative of the Gaisser-Hillas function.
    """
    return Nmax * np.exp((Xmax-X)/lamb) * ((X0-X)/(X0-Xmax))**((Xmax-X0)/lamb) * (Xmax-X)/(X-X0)/lamb

def threshold(E0, A, E_mu, alpha):
    """
    threshold factor of the Elbert formula.
    """
    E_N = E0/A
    if E_N < E_mu:
        return 0
    return (1-E_mu/E_N)**alpha

def log_function(E0, A, E_mu, param_dict):
    """
    Function to describe parametrized Xmax, lamb, X0, f as function of log10(E0/A/E_mu)
    """
    log_x = np.log10(E0/A/E_mu)
    i = "1" if (log_x <= param_dict['q']) else "2"
    return param_dict['a'+i] + param_dict['b'+i] * log_x

def power_function(E0, A, E_mu, param_dict):
    """
    Function to describe parametrized Nmax as function of (E0/A/E_mu)
    """
    log_x = np.log10(E0/A/E_mu)
    i = "1" if (log_x <= param_dict['q']) else "2"
    return param_dict['c'+i] * A * (E0/A/E_mu)**param_dict['p'+i]

def decay_fraction(X, T, theta, E_mu, f, kind='pion'):
    """
    Fraction of mesons that decays to muons vs reinteracting.

    :param X:     slant depth (g/cm^2)
    :param T:     atmospheric temperature at depth X (K)
    :param theta: zenith angle (degrees)
    :param E_mu:  minimum muon energy (GeV)
    :param f:     ratio between mean (above minimum) and minimum muon energy
    :param kind:  type of mesons that decay into muons: "pion" or "kaon"
    """
    if kind == 'pion':
        # critical energy
        eps = 0.524 * T
        # interaction length
        L = 111.
        # ratio between muon and meson energy: E_mu = r * E_pi
        R = 0.79
    elif kind == 'kaon':
        eps = 3.897 * T
        L = 122.
        R = 0.52

    # decay fraction written as a function of the ratio between interaction length and decay length
    ratio = R*L*eps/f/E_mu/np.cos(np.deg2rad(theta))/X
    return ratio*(1/(1+ratio))

def production_profile(X, T, E0, A, theta, E_mu, params=parameters['IceCube']):
    """
    Calculate amount of muons produced per (g/cm^2).
    
    :param X: slant depth (g/cm^2)
    :param T: atmospheric temperature at depth X
    :param E0:    primary cosmic ray energy (GeV)
    :param A:     primary cosmic ray mass
    :param theta: primary cosmic ray zenith angle (degrees)
    :param E_mu:  minimum muon energy (GeV)
    :param params: dictionary with parameter values for a certain parameterization
    """

    # values for four parameters in Gaisser-Hillas derivative
    Nmax = power_function(E0, A, E_mu, params['Nmax'])
    Xmax = log_function(E0, A, E_mu, params['Xmax'])
    lamb = log_function(E0, A, E_mu, params['lamb'])
    X0 = log_function(E0, A, E_mu, params['X0'])
    # ratio between mean muon energy (above minimum) and minimum muon energy
    f = log_function(E0, A, E_mu, params['f'])
    # exponent for threshold factor
    alpha2 = params['elbert']['alpha2']
    # muon production: Gaisser-Hillas derivative * decay fraction (pion and kaon) * threshold factor
    # step function ensures production does not become negative for X > Xmax
    return (GH_derivative(X, Nmax, Xmax, lamb, X0) * np.heaviside(Xmax-X, 0) *
            (0.92 * decay_fraction(X, T, theta, E_mu, f, kind='pion') +
             0.08 * decay_fraction(X, T, theta, E_mu, f, kind='kaon')) *
            threshold(E0, A, E_mu, alpha2))

if __name__ == "__main__":
    # simple usage example
    import matplotlib.pyplot as plt

    # depth grid (g/cm^2) and temperature profiles (K) from AIRS data
    depth = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 70.0, 100.0,
                      150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0]) * 1.019
    temp_summer = np.array([291.44, 284.90, 278.24, 270.21, 259.60, 252.24, 246.05, 239.98, 237.27,
                            235.24, 234.71, 233.63, 229.88, 229.09, 226.68, 220.26, 221.60, 234.45,
                            244.32, 250.52, 253.31])
    temp_winter = np.array([264.83, 249.95, 235.96, 222.82, 219.77, 206.54, 190.51, 185.55, 185.36,
                            180.73, 178.72, 184.05, 185.15, 191.88, 195.05, 198.83, 204.95, 217.83,
                            227.52, 233.27, 233.27])

    # get profile for muons >400 GeV in vertical air showers from 10 PeV primary cosmic rays
    profile_p_summer = production_profile(depth, temp_summer, 1e7, 1, 0., 400.,
                                          params=parameters['IceCube'])
    profile_p_winter = production_profile(depth, temp_winter, 1e7, 1, 0., 400.,
                                          params=parameters['IceCube'])
    profile_Fe_summer = production_profile(depth, temp_summer, 1e7, 56, 0., 400.,
                                           params=parameters['IceCube'])
    profile_Fe_winter = production_profile(depth, temp_winter, 1e7, 56, 0., 400.,
                                           params=parameters['IceCube'])
    # instead of an array of temperatures, a single value can be given for an isothermal atmosphere
    profile_p_iso = production_profile(depth, 217., 1e7, 1, 0., 400.,
                                       params=parameters['IceCube'])
    profile_Fe_iso = production_profile(depth, 217., 1e7, 56, 0., 400.,
                                        params=parameters['IceCube'])

    # plot
    plt.figure()
    plt.plot(depth, profile_p_summer, 'r--')
    plt.plot(depth, profile_p_winter, 'r:')
    plt.plot(depth, profile_p_iso, 'r-', label='p')
    plt.plot(depth, profile_Fe_summer, 'b--')
    plt.plot(depth, profile_Fe_winter, 'b:')
    plt.plot(depth, profile_Fe_iso, 'b-', label='Fe')
    mlegend = plt.legend(loc='lower center')
    plt.legend([plt.plot([], [], color='k', linestyle=ls)[0] for ls in ('-', '--', ':')],
               ('isothermal 217K', 'summer', 'winter'), loc='upper right')
    plt.gca().add_artist(mlegend)
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel(r"X (g/cm$^2$)")
    plt.ylabel(r"dN/dX (/(g/cm$^2$))")
    plt.title("Muon production (> 400 GeV) in 10 PeV vertical showers")
    plt.show()
