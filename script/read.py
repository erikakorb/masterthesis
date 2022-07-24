#!/usr/bin/env python
# coding: utf-8

# # Preliminary operations

# In[3]:


### import libraries ###
import os
import shutil
import re
import time
import dask.dataframe as dd
import pandas as pd
import numpy as np
import WRBH_select    # WRBH_select.py, in the same folder, to select subset of binaries from WRBH df


# In[5]:


# physical costants
G4pi2 =  9953108.1      # G/(4 pi^2) in units of R_sun^3/(M_sun yr^2)


# # Function to read and write dataframes

# In[6]:


### define main function to process the SEVN2 output dataframes ###

# as input use the parameters from the results.py file, located in the same working directory
#
# [version] is the SEVN2 version adopted                            e.g. '3.0.0-Spindevel'
# [Nsim] is the number of simulations                               e.g. '1mln'
# [Z] is the metallicity of the stars                               e.g. '02' for Z=0.02 solar metallicity
# [SN] is the SN explosion prescription adopted                     e.g. 'del' for delayed or 'com' for compact
# [n] number of cores involved in dask computation                  e.g.  4
# [Hsup] is the threshold for spectroscopic WR                      e.g. '0.3' for WR with H < 0.3 on surface
#
# the following is an example:
# version = '3.0.0-Spindevel'                                      # to select SEVN2 version
# Nsim,Z,SN= '1mln','02', 'com'                                    # to select a set of simulations
# n = 4                                                            # cores for dask computation
# Hsup= 0.3                                                        # for functions inside WRBH_select.py


def Read(version,Nsim,Z,SN,kick,Hsup,n):

    start=time.time()
    ### paths ###

    ### input paths ###
    path_to_sevn2 = f'./SEVN2-{version}_Z{Z}_{SN}_{kick}/'                              # original folder with SEVN2 output
    path_to_sevn2_output = f'{path_to_sevn2}run_scripts/sevn_output/'  # original sevn_output folder with SEVN2 output
    path_evolved = f'{path_to_sevn2_output}evolved_0.dat'          # evolved binaries
    path_log = f'{path_to_sevn2_output}logfile_0.dat'              # logfile
    path_out = f'{path_to_sevn2_output}output_0.csv'               # output file

    ### prepare new folder with results ###
    path_results = f'./v_{version}/{Nsim}_Z{Z}_{SN}_{kick}/'         # path to new folder with all useful results
    path_to_copied = f'{path_results}copied/'                 # where to copy some of the simulated files

    path_df= f'{path_results}dataframes/'                     # where to write processed dataframes
    path_df_prog = f'{path_df}progenitors/'                   # where to write progenitors
    path_df_rem = f'{path_df}remnants/'                       # where to write remnants
    path_df_in_WRBH = f'{path_df}initial_WRBH/'               # where to write initial timestamp for WRBH phase
    path_df_fin_WRBH = f'{path_df}final_WRBH/'                # where to write final timestamp for WRBH phase

    ### check directories exist or create them ###
    path_list = [path_results,path_to_copied,path_df,path_df_prog,path_df_rem,path_df_in_WRBH,path_df_fin_WRBH]
    for path in path_list:
        os.makedirs(path, exist_ok=True)


    ### copy some original SEVN2 output files ###
    _=shutil.copy2(f'{path_to_sevn2}run_scripts/run.sh', path_to_copied)          # run.sh        
    _=shutil.copy2(f'{path_to_sevn2}run_scripts/listBin.dat', path_to_copied)     # listBin.dat
    _=shutil.copy2(f'{path_to_sevn2_output}evolved_0.dat', path_to_copied)        # evolved_0.dat (contains seeds)
    _=shutil.copy2(f'{path_to_sevn2_output}logfile_0.dat', path_to_copied)        # logfile_0.dat (contains key points)
    _=shutil.copy2(f'{path_to_sevn2_output}launch_line.txt', path_to_copied)      # launch_line.txt (for bug fix)
    _=shutil.copy2(f'{path_to_sevn2_output}used_params.svpar', path_to_copied)    # (for bug fix)
    _=shutil.copy2(f'{path_to_sevn2_output}failed_0.dat', path_to_copied)         # (for bug fix)
    
    ##########################################

    ### identify data of interest in SEVN output, evolved and log files ###

    ### output file read with dask.dataframe ###
    startout=time.time()
    out=dd.read_csv(path_out, blocksize= 128* 1024 * 1024)                   # output file

    ##########################################

    ### identify data of interest in SEVN output, evolved and log files ###

    ### output file read with dask.dataframe ###
    startout=time.time()
    out=dd.read_csv(path_out, blocksize= 128* 1024 * 1024)                   # output file

    # binaries of compact objects
    BH0,BH1 = (out['PhaseBSE_0'] == 14), (out['PhaseBSE_1'] == 14)
    NS0,NS1 = (out['PhaseBSE_0'] == 13), (out['PhaseBSE_1'] == 13)

    BHBH = out.loc[BH0 & BH1]                            # all rows with two BH
    BHNS = out.loc[(BH0 & NS1) | (NS0 & BH1)]            # all rows with a BH and a NS

    # BHs with a companion WR, either He-naked or spectroscopic (check is not a remnant in that case)
    WR0_He = ((out['PhaseBSE_0'] == 7) | (out['PhaseBSE_0'] == 8) | (out['PhaseBSE_0'] == 9))
    WR0_sp = ((out['Hsup_0'] < Hsup) & (out['RemnantType_0']== 0) & (out['PhaseBSE_0']!= 12))
    WR0 = (WR0_He | WR0_sp)
    WR1_He = ((out['PhaseBSE_1'] == 7) | (out['PhaseBSE_1'] == 8) | (out['PhaseBSE_1'] == 9))
    WR1_sp = ((out['Hsup_1'] < Hsup) & (out['RemnantType_1']== 0) & (out['PhaseBSE_1']!= 12))
    WR1 = (WR1_He | WR1_sp)

    WRBH = out.loc[ (BH0 & WR1) | (WR0 & BH1) ]             # all rows with a WR and a BH

    # compute dataframes
    WRBH = WRBH.compute(num_workers=n) 
    BHBH = BHBH.compute(num_workers=n) 
    BHNS = BHNS.compute(num_workers=n) 


    endout=time.time()
    print('Time to elaborate the output file [s]: ', endout-startout)

    ### add columns for Roche Lobe filling ratio for WR + BH systems ###
    WRBH['RL0_fill'] = WRBH['Radius_0'] / WRBH['RL0']  
    WRBH['RL1_fill'] = WRBH['Radius_1'] / WRBH['RL1']

    ####################################################################
    ### extract CE counts from the log file with regular expressions ###

    # in the logfile, CE lines have the following structure
    # B;name;ID;CE;time;ID1:M1:MHe1:MCO1:phase1:rem_type1:ID2:M2:MHe2:MCO2:phase2:rem_type2:a:afin:fate
    # where the labels '1' and '2' indicate respectively the values of the star
    # that started (primary) or suffered(secondary) the CE
    # all values refer at the beginning of the CE, exctept for 'afin' that is 
    # the semimajor axis after the CE

    #regex_str=r'B;\d+;\d+;CE;(\d+.\d+);(\d+):(\d+.\d+):\d+.\d+:\d+.\d+:\d+:\d+:(\d+):(\d+.\d+):\d+.\d+:\d+.\d+:\d+:\d+:(\d+\d+):(\d+\d+):\d+'
    #fl = '\d+.\d+'
    #exp = '\d+.\d+[e][+\-]?\d+'
    #regex_str=fr'B;\d+;(\d+);CE;({fl});(\d+):({exp}):{exp}:{exp}:\d+:\d+:(\d+):({exp}):{exp}:{exp}:\d+:\d+:({exp}):({exp}):\d+'
    regex_str=r'B;\d+;(\d+);CE;'
    with open(path_log,"r") as f:
        CE_mask = re.findall(regex_str,f.read())  # find all the CE from the logfile and save the binary ID

    CE = pd.DataFrame({'ID':np.asarray(CE_mask,dtype=int)})
    CE = CE.groupby(['ID']).size().reset_index(name='NCE')    # count how many CE occurred in a given binary

    ### read evolved file and add period column with pandas ###
    evolved = pd.read_csv(path_evolved, sep='\s+').rename(columns = {'#ID':'ID'})
    evolved['Period'] = np.sqrt(evolved['a']**3 / (G4pi2*(evolved['Mass_0']+evolved['Mass_1']))) # Kepler 3rd law

    ### add CE info on dataframes of interest ###
    evolved = pd.merge(evolved,CE, on='ID',how='left')
    BHBH = pd.merge(BHBH,CE, on='ID',how='left')
    BHNS = pd.merge(BHNS,CE, on='ID',how='left')

    ### write reduced dataframes ###
    startmain=time.time()
    print('Begin writing the main dataframes')

    evolved.to_csv(f'{path_df}evolved.csv')
    WRBH.to_csv(f'{path_df}WRBH.csv')
    BHBH.to_csv(f'{path_df}BHBH.csv')
    BHNS.to_csv(f'{path_df}BHNS.csv')

    endmain = time.time()
    print(f'Ended writing the main dataframes in [s]', endmain-startmain)

