#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import dask.dataframe as dd


# In[3]:


### Functions to select a subsection of dataframes of interest from the WRBH dataframe ###
#
# this functions are written for the read.py file
# which contains the additional parameters
#
# Hsup,thres_RL,thres_wind,observed = 0.3,1.,0.8,'cyg_x-3'         # for functions inside WRBH_select.py
#
#
# nevertheless, these functions can work also as standalone
# but we first have load the WRBH dataframe with dask
# for instance:
#
# path_df =  './1mln_Z02_com/dataframes/'
# n = 4
#
#
# note that every function contain these parameter as argument:
#
# [path_df] is the path to the WRBH dataframe
# [n] is the number of cores involved in the dask computation


# # First and last timestep of binary ID

# In[2]:


### Selects only first and last element of a pandas dataframe ###
#
# if in input there is a dask dataframe it is necessary to set to True the optional argument [dask]

def first_last_ID(df,dask=False):   
    first = df.drop_duplicates(subset=['ID'], keep='first',ignore_index=True)
    last = df.drop_duplicates(subset=['ID'], keep='last',ignore_index=True)
    
    if dask ==True:
        first = first.compute(num_workers = 4)
        last = last.compute(num_workers = 4)
       
    df = pd.concat([first,last]).sort_values(['ID','BWorldtime'])  
    
    return df


# # Observed candidates

# In[5]:


### Select candidate binaries to compare with observed known binary ###
#
# [observed] to select binaries similar to one observed             e.g. 'cyg_x-3'
# [firstlast] to select only first and last timestap for each ID if desired, default = False
#
# observed = 'cyg_x-3'

def obs(path_df,n,Hsup,observed,firstlast=False):
    if observed == 'cyg_x-3--Zd13':
        # read WRBH with dask
        WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)

        # period taken from Singh 2002 (cited by Zdziarski 2013 as P=0.19969 days)
        # given the fits on page 6 of Singh 2002, it could be an uncertainty of 0.00001 days
        # in doubt, we can consider a larger range from 4.5 to 5.1 hours (we expect the true value to be 4.8 hours)
        P_min, P_max = 4.5/(24*365), 5.1/(24*365)       # period converted from hours to years
        WRBH_per = WRBH.loc[(P_min <= WRBH['Period']) & (WRBH['Period']<=P_max)]

        # mass ranges from Zdziarski 2013, MNRAS, 429 
        # only BH_mass_min = 3 Msun is modified to be coherent with SEVN settings
        WR_mass_min, WR_mass_max = 7.5, 14.2     # Msun
        BH_mass_min, BH_mass_max = 3, 4.5        # Msun
        WR0_mass = ((WR_mass_min <= WRBH_per['Mass_0']) & (WRBH_per['Mass_0']<=WR_mass_max))
        WR1_mass = ((WR_mass_min <= WRBH_per['Mass_1']) & (WRBH_per['Mass_1']<=WR_mass_max))
        BH0_mass = ((BH_mass_min <= WRBH_per['Mass_0']) & (WRBH_per['Mass_0']<=BH_mass_max))
        BH1_mass = ((BH_mass_min <= WRBH_per['Mass_1']) & (WRBH_per['Mass_1']<=BH_mass_max))
        
        # make sure to put e.g. a WR constrain on a WR and not to a BH
        WR0_He = ((WRBH['PhaseBSE_0'] == 7) | (WRBH['PhaseBSE_0'] == 8) | (WRBH['PhaseBSE_0'] == 9))
        WR0_sp = ((WRBH['Hsup_0'] < Hsup) & (WRBH['RemnantType_0']== 0)  & (WRBH['PhaseBSE_0']!= 12))
        WR0 = (WR0_He | WR0_sp)
        WR1_He = ((WRBH['PhaseBSE_1'] == 7) | (WRBH['PhaseBSE_1'] == 8) | (WRBH['PhaseBSE_1'] == 9))
        WR1_sp = ((WRBH['Hsup_1'] < Hsup) & (WRBH['RemnantType_1']== 0) & (WRBH['PhaseBSE_1']!= 12))
        WR1 = (WR1_He | WR1_sp)        
        BH0,BH1 = (WRBH['PhaseBSE_0'] == 14), (WRBH['PhaseBSE_1'] == 14)
        
        # compute the dataframe
        WRBH_obs = WRBH_per.loc[(WR0 & WR0_mass & BH1_mass & BH1) | (BH0 & BH0_mass & WR1_mass & WR1)]
        WRBH_obs = WRBH_obs.compute(num_workers=n)
        
        # as optional it is possible to select only first and last timestep for each ID
        if firstlast==True:
            WRBH_obs = first_last_ID(WRBH_obs)
            
    
    
    elif observed == 'cyg_x-3--An22':
        # read WRBH with dask
        WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)

        # period taken from Antokhin 2019 (cited by Antokhin 2022 as P=0.199684.. days)
        # given the fits on page 5 of Antokhin 2019, it could be an uncertainty of 0.000001 days
        # in doubt, we can consider a larger range from 4.5 to 5.1 hours (we expect the true value to be 4.8 hours)
        P_min, P_max = 4.5/(24*365), 5.1/(24*365)       # period converted from hours to years
        WRBH_per = WRBH.loc[(P_min <= WRBH['Period']) & (WRBH['Period']<=P_max)]
 
        # mass ranges from Antokhin 2022, ApJ         
        # from mass loss rate + smooth wind they estimate MWR+MBH = 18.8 Msun
        # using M-Mdot relation from Nugis Lamers 2000 + Zdziarski 2013 they find
        # MWR = 11.6 +- 1.2 Msun therefore MBH = 7.2 +- 1.2 Msun
        WR_mass_min, WR_mass_max = 10.4, 12.8     # Msun
        BH_mass_min, BH_mass_max = 6, 8.4       # Msun
        WR0_mass = ((WR_mass_min <= WRBH_per['Mass_0']) & (WRBH_per['Mass_0']<=WR_mass_max))
        WR1_mass = ((WR_mass_min <= WRBH_per['Mass_1']) & (WRBH_per['Mass_1']<=WR_mass_max))
        BH0_mass = ((BH_mass_min <= WRBH_per['Mass_0']) & (WRBH_per['Mass_0']<=BH_mass_max))
        BH1_mass = ((BH_mass_min <= WRBH_per['Mass_1']) & (WRBH_per['Mass_1']<=BH_mass_max))
        
        # make sure to put e.g. a WR constrain on a WR and not to a BH
        WR0_He = ((WRBH['PhaseBSE_0'] == 7) | (WRBH['PhaseBSE_0'] == 8) | (WRBH['PhaseBSE_0'] == 9))
        WR0_sp = ((WRBH['Hsup_0'] < Hsup) & (WRBH['RemnantType_0']== 0)  & (WRBH['PhaseBSE_0']!= 12))
        WR0 = (WR0_He | WR0_sp)
        WR1_He = ((WRBH['PhaseBSE_1'] == 7) | (WRBH['PhaseBSE_1'] == 8) | (WRBH['PhaseBSE_1'] == 9))
        WR1_sp = ((WRBH['Hsup_1'] < Hsup) & (WRBH['RemnantType_1']== 0) & (WRBH['PhaseBSE_1']!= 12))
        WR1 = (WR1_He | WR1_sp)        
        BH0,BH1 = (WRBH['PhaseBSE_0'] == 14), (WRBH['PhaseBSE_1'] == 14)
        
        # compute the dataframe
        WRBH_obs = WRBH_per.loc[(WR0 & WR0_mass & BH1_mass & BH1) | (BH0 & BH0_mass & WR1_mass & WR1)]
        WRBH_obs = WRBH_obs.compute(num_workers=n)
        
        # as optional it is possible to select only first and last timestep for each ID
        if firstlast==True:
            WRBH_obs = first_last_ID(WRBH_obs)
            
        
    elif observed == 'cyg_x-3--Ko17':
        # read WRBH with dask
        WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)

        # period taken from Antokhin 2019 (even though also Koljonen+2017 recognize it from light curve)
        # given the fits on page 5 of Antokhin 2019, it could be an uncertainty of 0.000001 days
        # in doubt, we can consider a larger range from 4.5 to 5.1 hours (we expect the true value to be 4.8 hours)
        P_min, P_max = 4.5/(24*365), 5.1/(24*365)       # period converted from hours to years
        WRBH_per = WRBH.loc[(P_min <= WRBH['Period']) & (WRBH['Period']<=P_max)]
 
        # mass ranges from Koljonen & Maccarone 2017      
        # from luminosity-mass relation of Graefner+2011 
        # and accounting for possible distances 7.4-10.2 kpc
        # they estimate MWR = 8-10 or 11-14 i.e. overall MWR = 8-14
        # accounting for orbital inclination and wind velocity they estimate MBH < 10 Msun
        WR_mass_min, WR_mass_max = 8., 14.     # Msun
        BH_mass_min, BH_mass_max = 3., 10.       # Msun
        WR0_mass = ((WR_mass_min <= WRBH_per['Mass_0']) & (WRBH_per['Mass_0']<=WR_mass_max))
        WR1_mass = ((WR_mass_min <= WRBH_per['Mass_1']) & (WRBH_per['Mass_1']<=WR_mass_max))
        BH0_mass = ((BH_mass_min <= WRBH_per['Mass_0']) & (WRBH_per['Mass_0']<=BH_mass_max))
        BH1_mass = ((BH_mass_min <= WRBH_per['Mass_1']) & (WRBH_per['Mass_1']<=BH_mass_max))
        
        # make sure to put e.g. a WR constrain on a WR and not to a BH
        WR0_He = ((WRBH['PhaseBSE_0'] == 7) | (WRBH['PhaseBSE_0'] == 8) | (WRBH['PhaseBSE_0'] == 9))
        WR0_sp = ((WRBH['Hsup_0'] < Hsup) & (WRBH['RemnantType_0']== 0)  & (WRBH['PhaseBSE_0']!= 12))
        WR0 = (WR0_He | WR0_sp)
        WR1_He = ((WRBH['PhaseBSE_1'] == 7) | (WRBH['PhaseBSE_1'] == 8) | (WRBH['PhaseBSE_1'] == 9))
        WR1_sp = ((WRBH['Hsup_1'] < Hsup) & (WRBH['RemnantType_1']== 0) & (WRBH['PhaseBSE_1']!= 12))
        WR1 = (WR1_He | WR1_sp)        
        BH0,BH1 = (WRBH['PhaseBSE_0'] == 14), (WRBH['PhaseBSE_1'] == 14)
        
        # compute the dataframe
        WRBH_obs = WRBH_per.loc[(WR0 & WR0_mass & BH1_mass & BH1) | (BH0 & BH0_mass & WR1_mass & WR1)]
        WRBH_obs = WRBH_obs.compute(num_workers=n)
        
        # as optional it is possible to select only first and last timestep for each ID
        if firstlast==True:
            WRBH_obs = first_last_ID(WRBH_obs)
    
    else:
        col_names = WRBH.columns.to_list()
        WRBH_obs = pd.DataFrame(columns = col_names)  # create empty dataframe so the read.py file keeps working
    
    return WRBH_obs


# # Roche Lobe filling

# In[8]:


### Select RL filling systems above a given threshold ###
#
# [thres_min] is the minimum R/RL filling ratio to consider    e.g. thres_min = thres_RL = 1.
# [firstlast] to select only first and last timestap for each ID if desired, default = False

def RL_filling(path_df,n,thres_min,firstlast=False):
    # read WRBH with dask
    WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)
    
    # compute the dataframe
    WRBH_RL= WRBH.loc[(WRBH['RL0_fill'] >= thres_min) | (WRBH['RL1_fill'] >= thres_min)]
    WRBH_RL = WRBH_RL.compute(num_workers=n)
    
    # as optional it is possible to select only first and last timestep for each ID
    if firstlast==True:
        WRBH_RL = first_last_ID(WRBH_RL)
    
    return WRBH_RL


# In[9]:


### Select pure wind-fed systems (not yet completely RL_filling) ###
#
# [thres_min] is the minimum R/RL filling ratio to consider    e.g. thres_min = thres_wind = 0.8
# [thres_max] is the maximum R/RL filling ratio to consider    e.g. thres_max = thres_RL = 1.
# [firstlast] to select only first and last timestap for each ID if desired, default = False

def RL_wind(path_df,n,thres_min,thres_max,firstlast=False):
    # read WRBH with dask
    WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)
    
    # compute the dataframe
    WRBH_wind=WRBH.loc[((thres_max > WRBH['RL0_fill']) & (WRBH['RL0_fill'] >= thres_min)) | 
                   ((thres_max > WRBH['RL1_fill']) & (WRBH['RL1_fill'] >= thres_min))]
    WRBH_wind = WRBH_wind.compute(num_workers=n)
    
    # as optional it is possible to select only first and last timestep for each ID
    if firstlast==True:
        WRBH_wind = first_last_ID(WRBH_wind)
    
    return WRBH_wind


# # Pure spectroscopic WR

# In[10]:


### Pure spectroscopic WR (no He-naked yet) ###
#
# [Hsup] for surface hydrogen H<Hsup the star is a WR       e.g. Hsup = 0.3
# [firstlast] to select only first and last timestap for each ID if desired, default = False

def WR_spectro(path_df,n,Hsup,firstlast=False):
    # read WRBH with dask
    WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)
    
    # compute the dataframe
    WR0_sp_pure = ((WRBH['Hsup_0'] < Hsup) & (WRBH['RemnantType_0']== 0 & (WRBH['PhaseBSE_0']!= 12)) 
               & (WRBH['PhaseBSE_0'] != 7) & (WRBH['PhaseBSE_0'] != 8) & (WRBH['PhaseBSE_0'] != 9))
    WR1_sp_pure = ((WRBH['Hsup_1'] < Hsup) & (WRBH['RemnantType_1']== 0 & (WRBH['PhaseBSE_1']!= 12)) 
               & (WRBH['PhaseBSE_1'] != 7) & (WRBH['PhaseBSE_1'] != 8) & (WRBH['PhaseBSE_1'] != 9))
    WRBH_sp = WRBH.loc[(WR0_sp_pure | WR1_sp_pure)]
    
    WRBH_sp = WRBH_sp.compute(num_workers=n)
    
    # as optional it is possible to select only first and last timestep for each ID
    if firstlast==True:
        WRBH_sp = first_last_ID(WRBH_sp)
    
    return WRBH_sp

