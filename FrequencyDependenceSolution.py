import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.font_manager as font_manager
import warnings
warnings.filterwarnings("ignore")

def S(z_values, xt, gt, epst, N):
    """ Evaluate the series solution S(z) for multiple z-values.
        The function also returns the coefficients s_k, which are
        needed for the other linearly independent solution P(z).

        Arguments:

        z_values: Values of z (array)
        xt:       Dimensionless variable for recoil
        gt:       Dimensionless variable for expansion/contraction
        epst:     Dimensionless variable for dust absorption. """
    
    # Generate the first N coefficients s_k for S(z)
    # (NOTE: DON'T CHANGE THESE RELATIONS!):
    
    s    = np.zeros(N)
    s[0] = 1.0
    s[1] = -(1/4)*xt*s[0]
    s[2] = -(1/5)*xt*s[1]
    s[3] = -(1/6)*xt*s[2]-(1/6)*gt*s[0]
    s[4] = -(1/7)*xt*s[3]-(1/7)*gt*s[1] + epst*s[0]/(7*4)
    s[5] = -(1/8)*xt*s[4] - (1/8)*gt*s[2] + epst*s[1]/(8*5)
    
    for k in range(N - 6):
        # Recursive relation for higher k:
        s[k+6] = ( (s[k]+epst*s[k+2])/((k+9)*(k+6))
                  - xt*s[k+5]/(k+9) - gt*s[k+3]/(k+9) )
    
    # Compute the partial sums of S(z), all the way up to the N:th term:
    
    k_values     = np.arange(N)
    z_powers     = np.power(z_values[:, np.newaxis], k_values + 3)
    partial_sums = np.zeros((N, len(z_values)))

    for n in range(1, N + 1):
        partial_sums[n-1] = np.dot(z_powers[:, :n], s[:n])
    
    return partial_sums


def ShanksTransform(series, z_values, xt, gt, epst, N):

    # Read in all the needed partial sums:

    s = series(z_values, xt, gt, epst, N+1)

    s_Np1 = s[N+1]
    s_N   = s[N]
    s_Nm1 = s[N-1]

    denominator = s_Np1 - 2*s_N + s_Nm1
    if np.any(denominator == 0):
        series_shanks = s_N
    else:
        series_shanks = s_Np1 - ((s_Np1 - s_N)**2)/(s_Np1 - 2*s_N + s_Nm1)

    return series_shanks

def ShanksTransform_it(series, z_values, xt, gt, epst, N):

    s_Np1 = ShanksTransform(series, z_values, xt, gt, epst, N+1)
    s_N   = ShanksTransform(series, z_values, xt, gt, epst, N)
    s_Nm1 = ShanksTransform(series, z_values, xt, gt, epst, N-1)

    denominator = s_Np1 - 2*s_N + s_Nm1
    if np.any(denominator == 0):
        series_shanks = s_N
    else:
        series_shanks = s_Np1 - ((s_Np1 - s_N)**2)/(s_Np1 - 2*s_N + s_Nm1)

    return series_shanks

xt = 7.0; gt = 0.0; epst = 0.0; N = 170
z_values = np.array([4.0])
Values0 = S(z_values, xt, gt, epst, N)
Values = ShanksTransform_it(S, z_values, xt, gt, epst, N)
print( Values0[-1] )
print( Values )



def P_andmore(z_values, xt, gt, epst, N, limit = 'pos'):
    """ Evaluate the second solution P(z) for multiple z-values.
        Also returns S(z) and the ratio lim_z->+/-infty P(z)/S(z)
        for convenience. """

    # Read in S, dS/dz, and its coefficients:

    S_values, Sder_values, s = S(z_values, xt, gt, epst, N)

    # Generate the first N coefficients p_k for the series
    # solution part of P(z), as well as the constant A.
    # (NOTE: DON'T CHANGE THESE RELATIONS!):

    p    = np.zeros(N)
    p[0] = 1.0
    p[1] = -xt
    p[2] = -xt*p[1]/2
    p[3] = 0.0
    p[4] = -(1/4)*gt*p[1] + (1/4)*epst*p[0]
    p[5] = -(1/5)*xt*p[4] - (1/5)*gt*p[2] + (1/10)*epst*p[1]

    for k in range(N - 6):
        # Recursive relation for higher k:
        p[k + 6] = ( (p[k]+epst*p[k+2])/((k+6)*(k+3))

                 - xt*p[k+5]/(k+6) - gt*p[k+3]/(k+6) )

    # Compute P(z) for multiple z-values:
    
    k_values        = np.arange(N)
    z_powers        = np.power( z_values[:, np.newaxis], k_values )
    P_values        = np.dot( z_powers, p )

    # Also compute (dP/dz)/(z^2):

    zder_powers     = np.power( z_values[:, np.newaxis], k_values - 3 ) 
    Pder_values     = np.dot( zder_powers, p*k_values )

    # Compute the ratio at the limitng value:

    if limit == 'pos':
        LimitRatio = P_values[-1]/S_values[-1]
    elif limit == 'neg':
        LimitRatio = P_values[0]/S_values[0]


    return P_values, S_values, LimitRatio



def Solution(z_values, xt, gt, epst, N):
    """ Full solution f(z) that satisfy f -> 0 at abs(z) -> infty. """


    # Get positive & negative z-values:

    z_values_pos = z_values[z_values > 0]
    z_values_neg = z_values[z_values < 0]

    # Get the solution for positive and negative z:

    ( P_valuespos, S_valuespos, LimitRatiopos) = (
      P_andmore(z_values_pos, xt, gt, epst, N, limit = 'pos') )
    
    ( P_valuesneg, S_valuesneg, LimitRationeg) = (
      P_andmore(z_values_neg, xt, gt, epst, N, limit = 'neg') )

    Solution_pos = P_valuespos - S_valuespos*LimitRatiopos
    Solution_neg = P_valuesneg - S_valuesneg*LimitRationeg

    # Put them together to get f(z):

    Solution = np.concatenate((Solution_neg, Solution_pos))

    # Get (z^(-2))*df/dz(+) - (z^(-2))*df/dz(-) near the origin:

    DiffDerivative = -3.0*( LimitRatiopos - LimitRationeg )

    # Compute the integral of f(z) over z:

    Integral = np.trapz( Solution, z_values )

    return Solution, DiffDerivative, Integral/(0.858599*(1.0 - DiffDerivative))
    
def fit(z, xt):

    fit = np.exp( - (1/3)*(abs(z)**3) - xt*z)

    return fit

###############################################################
#######                                                 #######
#######           P L O T   C O S M E T I C S           #######
#######                                                 #######
###############################################################

# REMOVE THIS IF YOU DON'T HAVE THE REQUIRED FONTS:

#plt.rc('font',**{'sans-serif':['RomanD'],
                 #'size': 27} )
#plt.rc('font',**{'sans-serif':['AVHershey Simplex'],
                #'size': 27} )
plt.rc('font',**{'sans-serif':['AVHershey Complex'],
                'size': 40} )
#plt.rc('font',**{'sans-serif':['Complex'],
                # 'size': 27} )


params = {'mathtext.default': 'regular' }
plt.rcParams.update(params)

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'GreekC'

# The stuff below need not be removed:

params = {'legend.fontsize': 32 }
plt.rcParams.update(params)

plt.rcParams["legend.markerscale"] = 0.75

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.major.width'] = 2.2
plt.rcParams['xtick.minor.width'] = 2.2
plt.rcParams['xtick.major.size'] = 20
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.width'] = 2.2
plt.rcParams['ytick.minor.width'] = 2.2
plt.rcParams['ytick.major.size'] = 20
plt.rcParams['ytick.minor.size'] = 8
plt.rcParams['xtick.major.pad']='15'
plt.rcParams['ytick.major.pad']='15'

plt.rcParams['legend.frameon'] = 'False'

plt.rcParams['axes.linewidth'] = 2.7
plt.rcParams['grid.linewidth'] = 2.7
plt.rcParams['grid.linestyle'] = '--'

"""xt = 7.0; gt = 0.0; epst = 0.0
z_values = np.array([4.0])
N = np.arange( 10, 200, step = 1 )

Trunc  = np.array([ abs(S(z_values, xt, gt, epst, n)[-1]) for n in N ])
#Shanks = np.array([ abs(ShanksTransform(S, z_values, xt, gt, epst, n)) for n in N ])
Shanks2 = np.array([ abs(ShankTransform_it(S, z_values, xt, gt, epst, N)) for n in N ])

plt.semilogy( N, Trunc, color = 'red' )
#plt.semilogy( N, Shanks, color = 'blue' )
plt.semilogy( N, Shanks2, color = 'black' )
plt.show()"""
