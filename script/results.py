#!/usr/bin/env python
# coding: utf-8

# In[1]:


### import libraries ###
import os
import time
import dask.dataframe as dd
import pandas as pd
import numpy as np
import WRBH_select    # WRBH_select.py, in the same folder, to select subset of binaries from WRBH df
import read           # read.py, to reduce the original output dataframe


# In[28]:


### Define parameters ###
# [version] is the SEVN2 version adopted                            e.g. '3.0.0-Spindevel'
# [Nsim] is the number of simulations                               e.g. '1mln'
# [Z] is the metallicity of the stars                               e.g. '02' for Z=0.02 solar metallicity
# [SN] is the SN explosion prescription adopted                     e.g. 'del' for delayed or 'com' for compact
# [n] number of cores involved in dask computation                  e.g.  4
# [Hsup] is the threshold for spectroscopic WR                      e.g. '0.3' for WR with H < 0.3 on surface
# [thres_RL] is the threshold for Roche Lobe filling binaries
# [thres_wind] is the threshold for wind fed binaries


Nsim,Z,SN= '1mln','02', 'com'                                    # to select a set of simulations
kick = 'unified'
version = '3.0.0-Spindevel_RLO'                                      # select SEVN2 version
path_results = f'./v_{version}/{Nsim}_Z{Z}_{SN}_{kick}/'                # path to new folder with all useful results
n = 4                                                            # cores for dask computation
Hsup,thres_RL,thres_wind = 0.3,1.,0.8                            # for functions inside WRBH_select.py
observed1, observed2, observed3 = 'cyg_x-3--Zd13', 'cyg_x-3--An22', 'cyg_x-3--Ko17'  # types of mass ranges

# physical costants
H0=14000                # Hubble time in Myr


# In[3]:


### Make sure to have reduced the output_0.csv file ###
start=time.time()

read_cond = True
if read_cond == True:
    print('Called Read function from read.py')
    read.Read(version,Nsim,Z,SN,kick,Hsup,n)


# In[29]:


### Read with the reduced dataframes  ###
path_df = f'{path_results}dataframes/'

evolved = pd.read_csv(f'{path_df}evolved.csv')
WRBH = dd.read_csv(f'{path_df}WRBH.csv',blocksize= 128* 1024 * 1024)
BHBH = dd.read_csv(f'{path_df}BHBH.csv',blocksize= 128* 1024 * 1024)
BHNS = dd.read_csv(f'{path_df}BHNS.csv',blocksize= 128* 1024 * 1024)


# In[30]:


### Keep only first and last timestep for each ID to reduce ram occupation ###
# use the function inside the WRBH_select.py file
# set the optional argument dask = True to allow dask computation (disabled by default)

WRBH = WRBH_select.first_last_ID(WRBH,dask=True)
BHBH = WRBH_select.first_last_ID(BHBH,dask=True)
BHNS = WRBH_select.first_last_ID(BHNS,dask=True)


# In[11]:


##########################################

### Subsets of WRBH dataframe, selected with functions in WRBH_select.py ###
startsub = time.time()
print('Begin extracting all the sub-dataframes')

### Pure spectroscopic WR (no He-naked yet) ###
WRBH_sp = WRBH_select.WR_spectro(path_df,n,Hsup,firstlast=True)

### RL filling and wind fed WRBH binaries ###
WRBH_RL = WRBH_select.RL_filling(path_df,n,thres_min=thres_RL,firstlast=True)
WRBH_wind = WRBH_select.RL_wind(path_df,n,thres_min=thres_wind,thres_max=thres_RL,firstlast=True)

### Select candidate binaries to compare with observed known binary ###
WRBH_obs1 = WRBH_select.obs(path_df,n,Hsup=Hsup,observed=observed1,firstlast=True) 
WRBH_obs2 = WRBH_select.obs(path_df,n,Hsup=Hsup,observed=observed2,firstlast=True) 
WRBH_obs3 = WRBH_select.obs(path_df,n,Hsup=Hsup,observed=observed3,firstlast=True) 


# In[31]:


### Subsets of binaries of compact object ###

### BBH types ###
BHBH_bound=BHBH[~np.isnan(BHBH['Semimajor'])]           # BBH gravitationally bound
BHBH_GW=BHBH_bound.loc[BHBH_bound['GWtime'] <=H0]       # BBH that merge within Hubble time

### BHNS types ###
BHNS_bound=BHNS[~np.isnan(BHNS['Semimajor'])]           # BHNS gravitationally bound
BHNS_GW=BHNS_bound.loc[BHNS_bound['GWtime'] <=H0]       # BHNS that merge within Hubble time

endsub=time.time()
print(f'End extracting all the sub-dataframes in [s] ', endsub-startsub)


# In[32]:


### store values of binaries with common properties ###

### prepare lists ###
df_list1 = [BHBH,BHBH_bound,BHBH_GW,BHNS,BHNS_bound,BHNS_GW]
df_list2 = [WRBH,WRBH_sp,WRBH_RL,WRBH_wind,WRBH_obs1,WRBH_obs2,WRBH_obs3]

name_list1 = ['BHBH','BHBH_bound','BHBH_GW','BHNS','BHNS_bound','BHNS_GW']
name_list2 = ['WRBH','WRBH_sp','WRBH_RL','WRBH_wind',f'WRBH_{observed1}',f'WRBH_{observed2}',f'WRBH_{observed3}']

word_list = [' BHBH ', ' bound BBH ', ' GW-BBH ', ' BHNS ', ' bound BHNS ', ' GW-BHNS ']


# In[27]:


### write dataframes and useful results ###
print('Begin to write the useful dataframes')
startwrite=time.time()
with open(f'{path_df}results.txt', 'w') as f:
    f.write(f'Number of generated binaries: {Nsim} \n')
    f.write('Number of simulated binaries: ' + str(len(evolved.index)) + '\n')
    f.write('Number of WRBH systems: ' + str(len(WRBH.drop_duplicates(subset='ID').index)) + '\n')
    f.write('Number of WRBH_sp spectroscopic systems: ' + str(len(WRBH_sp.drop_duplicates(subset='ID').index)) + '\n')
    f.write('Number of WRBH_RL filling systems: ' + str(len(WRBH_RL.drop_duplicates(subset='ID').index)) + '\n')
    f.write('Number of WRBH_wind fed systems: ' + str(len(WRBH_wind.drop_duplicates(subset='ID').index)) + '\n')
    f.write(f'Number of {observed1} candidates: ' + str(len(WRBH_obs1.drop_duplicates(subset='ID').index)) + '\n')
    f.write(f'Number of {observed2} candidates: ' + str(len(WRBH_obs2.drop_duplicates(subset='ID').index)) + '\n')
    f.write(f'Number of {observed3} candidates: ' + str(len(WRBH_obs3.drop_duplicates(subset='ID').index)) + '\n')

    for df1,name1,word in zip(df_list1,name_list1,word_list):
        print(f'Begin to write the useful dataframes with {word}')
        f.write('----------------------------------------\n')
        f.write(f'Number of{word}systems: ' + str(len(df1.drop_duplicates(subset='ID').index)) + '\n')
        
        # select progenitors and remnants
        remnant = df1.drop_duplicates(['ID'],keep='last')
        IDs = set(remnant.ID.to_list())  # list of IDs
        progenitor = evolved.query('ID in @IDs')

        # write dataframes
        progenitor.to_csv(f'{path_df}/progenitors/p_{name1}.csv')
        remnant.to_csv(f'{path_df}/remnants/r_{name1}.csv')

        for df2,name2 in zip(df_list2,name_list2):    
            # ID of binaries that belong in both categories
            binID = set(df1['ID']).intersection(df2['ID'])

            # progenitors and remnants of such binaries
            prog = evolved.query('ID in @binID')
            remnant_timesteps = df1.query('ID in @binID')
            rem = remnant_timesteps.drop_duplicates('ID', keep='last')
            
            # initial and final timestamps of WRBH phase evolution for each subgroup
            timesteps = df2.query('ID in @binID')
            initial = timesteps.drop_duplicates('ID', keep='first')
            final = timesteps.drop_duplicates('ID', keep='last')

            # write dataframes
            prog.to_csv(f'{path_df}/progenitors/p_{name1}_{name2}.csv')
            rem.to_csv(f'{path_df}/remnants/r_{name1}_{name2}.csv')
            
            initial.to_csv(f'{path_df}/initial_WRBH/i_{name1}_{name2}.csv')
            final.to_csv(f'{path_df}/final_WRBH/f_{name1}_{name2}.csv')

            # write useful results
            f.write(f'Number of{word}systems formed after {name2}: ' + str(len(binID)) + '\n')

endwrite=time.time()
print('Time to write the useful dataframes [s]: ', endwrite-startwrite)

### output messages to check ###
end=time.time()
print('Time to elaborate all the SEVN output files [s]: ', end-start)
print(f'The SEVN output files have been correctly analyzed and stored in {path_df}')

