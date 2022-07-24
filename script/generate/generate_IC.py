from ic4popsyn import populations as pop
from ic4popsyn import tools

# Number of systems (10 systems are for backup)
Nbin = 1100000
#Nbin = 1000
# create a population of binaries
binSanaMDS = pop.Binaries(Nbin, model='sana_eccM&DS', mass_ranges=[10.,150.], qmin=0.1, alphas=[-2.3]) # available: sana12 / sana_eccm&ds
#SinglePop = pop.Binaries(Nbin, model='sana_eccM&DS', single_pop=True, mass_ranges=[5.,150.], alphas=[-2.3]) # available: sana12 / sana_eccm&ds


# save the population as input for SEVN
binSanaMDS.save_sevn_input('listBin.dat', 0.02, 0.02, 0.0, 0.0, 'end', 'zams', 'compact', 'compact', 'all')
#binSanaMDS.save_sevn_input('SEVNIC_BSE_placeholder','True')
#SinglePop.save_sevn_input('SEVNIC_SSE', 'True', 0.002, 0.002, 0.0, 0.0, 'end', 'zams', 'compact', 'compact', 'all')
