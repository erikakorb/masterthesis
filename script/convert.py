import numpy as np

# conversions
AU = float(150e9) # m   150 milions of km
Rsun = float(7e8)  #m   700 000 km
Msun = float(2e30) #kg
pc = float(3e16)   # m
Mpc = float(3e22)  #m

G = float(6.67e-11) #mks
G4pi2 = float(G/(4*np.pi*np.pi))  # G/4*pi^2

c= float(3e8)  #m/s

hour = 60*60
day = 60*60*24      # seconds
Myr = day * 365 * 1e6  #seconds

# data
M1 = 7.5 * Msun
M2 = 3 * Msun
P = 4.5 * day * 365  # period



# third Kepler law
# a^3/P^2 = (M1+M2)*G/4*pi^2

a = (P*P * (M1+M2)*G4pi2)**(1./3.)
print("a = ", a/Rsun, " Rsun")
print("a = ", a/AU, " AU")
print("1 AU = ", AU/Rsun, " Rsun")





# star property on MS
M = 60 * Msun
taunuc = 1./(M*M)    # nuclear timescale on MS
print("taunuc = ",   taunuc/Myr, " Myr")


# Schwarzschild
MBH = 1 * Msun   # kg
Rsch = 2*G*MBH/(c*c)

print("Rsch = ", Rsch/1000., "km")



# GW analysis
# numax = 1/pisqrt8 c**3/GMtot
Mtot = 6 * Msun

pisqrt8 = np.pi * np.sqrt(8)
numax = (1./pisqrt8) * (c**3)/(G*Mtot)
print('numax ', numax, ' Hz')

# Mtot = pisqrt8 c**3/Gnumax
numax = 100 # Hz

pisqrt8 = np.pi * np.sqrt(8)
Mtot = (1./pisqrt8) * (c**3)/(G*numax)

print('Mtot = ', Mtot/Msun, ' Msun')

Mtot1 = 70*Msun
Rcoal = Mtot1 * 2.*G/(c*c)

print('Rcoal ', Rcoal/1000, ' km')


# h = 4/r (Gmchrip/c**2)**5/3  (pinugw/c)**2/3
r = 410 * Mpc
mchirp = 30 * Msun
nugw = 300 # Hz


h = (4/r) * (G *mchirp/(c*c))**(5/3) *  (np.pi * nugw/c)**(2/3)
print('h ',h, ' Hz**0.5')

# r = 4/h (Gmchrip/c**2)**5/3  (pinugw/c)**2/3
h = float(1e-21)
mchirp = 30 * Msun
nugw = 300 # Hz

r = (4/h) * (G *mchirp/(c*c))**(5/3) *  (np.pi * nugw/c)**(2/3)
print('r ',r/Mpc, ' Mpc')
