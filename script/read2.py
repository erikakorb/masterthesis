#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### import libraries ###
import os
import shutil
import dask.dataframe as dd
import pandas as pd
import numpy as np

# # Function to read and write dataframes
### define main function to process the dataframes obtained with results.py, read.py and WRBH_select.py ###

# as input use the parameters from the results2.py file, located in the same working directory
#
# [version] is the SEVN2 version adopted                            e.g. '3.0.0-Spindevel'
# [Nsim] is the number of simulations                               e.g. '1mln'
# [Z] is the metallicity of the stars                               e.g. '02' for Z=0.02 solar metallicity
# [kick] is the type of SN kick                                     e.g. 'unified265' has 265 as sigma of maxwellian
# [SN] is the SN explosion prescription adopted                     e.g. 'del' for delayed or 'com' for compact
# [n] number of cores involved in dask computation                  e.g.  4
# [ppisn] if the debug includes only ppisn, also or excludes it     e.g. 'with','without','only'
#
# the following is an example:
# version = '3.0.0-Spindevel_RLO'                                      # to select SEVN2 version
# Nsim,Z,SN,kick = '1mln','015', 'com', 'unified265'                   # to select a set of simulations
# n = 4                                                            # cores for dask computation
# ppisn = 'only'

def Read(version,Nsim,Z,SN,kick,n, ppisn):
    ### input path with dataframes to be debugged ###
    path_results = f'./v_{version}/{Nsim}_Z{Z}_{SN}_{kick}/'         # path to new folder with all useful results
    path_df= f'{path_results}dataframes/'                     # where to write processed dataframes

    ### paths where to save debugged files ###
    path_ppisn = f'{path_results}ppisn_{ppisn}/'
    path_ppisn_prog = f'{path_ppisn}progenitors/'                   # where to write progenitors
    path_ppisn_rem = f'{path_ppisn}remnants/'                       # where to write remnants
    path_ppisn_in_WRBH = f'{path_ppisn}initial_WRBH/'               # where to write initial timestamp for WRBH phase
    path_ppisn_fin_WRBH = f'{path_ppisn}final_WRBH/'                # where to write final timestamp for WRBH phase

    ### check directories exist or create them ###
    path_list = [path_ppisn, path_ppisn_prog,path_ppisn_rem,path_ppisn_in_WRBH,path_ppisn_fin_WRBH]
    for path in path_list:
        os.makedirs(path, exist_ok=True)

    ### copy files ###
    _=shutil.copy2(f'{path_df}evolved.csv', path_ppisn)          # evolved.sh     


    ##########################################
    #### re-define condition for being a BH and a NS ###
    MminBH = 3.  # minimum mass to be a BH i.e. who underwent a PPISN and has M<MminBH is a NS now

    # if ppisn == 'with', the original analysis based on the PhaseBSE only was already considering
    # compact objects with M<3 Msun that underwent a PPISN episode as BHs, so the original dataframes were already ok
    if ppisn == 'with':
        _=shutil.copy2(f'{path_df}WRBH.csv', path_ppisn)          # WRBH.csv   
        _=shutil.copy2(f'{path_df}BHBH.csv', path_ppisn)          # BHBH.csv   
        _=shutil.copy2(f'{path_df}BHNS.csv', path_ppisn)          # BHNS.csv 

    # otherwise select the compact objects formed after ppsin and decide whether to classify them as BH or NS
    # and extract them from the WRBH, BHNS, BHBH files
    else:
        ### extract WRBH, BHBH and BHNS data previously identified only by condition on PhaseBSE via results.py ###
        WRBH=dd.read_csv(f'{path_df}WRBH.csv', blocksize= 128* 1024 * 1024)                   # WRBH
        BHBH=dd.read_csv(f'{path_df}BHBH.csv', blocksize= 128* 1024 * 1024)                   # BHBH
        BHNS=dd.read_csv(f'{path_df}BHNS.csv', blocksize= 128* 1024 * 1024)                   # BHNS

        for df,df_name in zip([WRBH,BHBH,BHNS],['WRBH','BHBH','BHNS']):
            print(f'Elaborating {df_name}.csv')
            # select only rows with at least one ppisn-derived compact object
            co0_ppisn = (df['PhaseBSE_0'] == 14) & (df['Mass_0'] < MminBH)
            co1_ppisn = (df['PhaseBSE_1'] == 14) & (df['Mass_1'] < MminBH)
            df_ppisn = df.loc[co0_ppisn | co1_ppisn]
            df_ppisn = df_ppisn.compute(num_workers=n)

            # either write only the binaries with at least one ppisn-derived compact object
            if ppisn == 'only':
                df_ppisn.to_csv(f'{path_ppisn}{df_name}.csv')
                df_ppisn = None

            # or remove the ppisn-derived compact objects from the original dataframes
            elif ppisn == 'without':
                print('Merging dataframes')
                df_select = dd.merge(df,df_ppisn, how='left', indicator=True)
                df_clean = df_select.loc[df_select['_merge'] == 'left_only']
                df_ppisn = None
                df_clean.to_csv(f'{path_ppisn}{df_name}.csv',single_file=True)
                df_clean = None
            
            # clean memory and print messages
            df = None
            print(f'Ended elaborating {df_name}.csv')


# In[ ]:


# version = '3.0.0-Spindevel_RLO'                                      # to select SEVN2 version
# Nsim,Z,SN,kick = '1mln','015', 'com', 'unified265'                   # to select a set of simulations
# n = 4                                                            # cores for dask computation
# ppisn = 'without'

# Read2(version,Nsim,Z,SN,kick,n, ppisn)

