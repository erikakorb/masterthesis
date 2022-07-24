import numpy as np
import pandas as pd
import os
import shutil
import subprocess
import seed_select   # from seed_select.py in same folder
#import plot_single   # from plot_single.py in same folder

####################################
######### Set parameters ###########
####################################

# parameters to identify simulation
Nsim, Z, SN, kick = '1mln','015','rap', 'hobbs265'
ppisn = 'without'
version = '3.0.0-Spindevel_RLO'

# extract listBin.dat with seeds
path_version = f'./v_{version}/'         # for SEVN2 version adopted
path = f'./{path_version}{Nsim}_Z{Z}_{SN}_{kick}/ppisn_{ppisn}/'
bintype = 'BHBH_GW_WRBH'
if ppisn == 'only':
    df_name = bintype
else:
    obs = 'cyg_x-3'
    mrange = 'Ko17'
    df_name = f'{bintype}_{obs}--{mrange}'



#############################################
####### Re-run the sample of interest #######
#############################################


#generate initial conditions with seed
runsevn = seed_select.GenerateBin(version,Nsim,Z,SN,kick,ppisn,df_name)
print(runsevn)

if runsevn == 'yes':
    print('Run the SEVN2 simulation')
    # run the SEVN2 simulation
    ### !!! WARNING !!!! ####
    ### the run.sh file still needs to be checked manually!!! ###
    path_to_run_scripts = f'/home/erika/Scrivania/sevn/SEVN2-{version}/run_scripts'
    if os.path.exists(f'{path_to_run_scripts}sevn_output'):
        shutil.rmtree(f'{path_to_run_scripts}sevn_output',ignore_errors=True)

    p = subprocess.Popen('./run.sh', cwd= path_to_run_scripts, stdout=subprocess.PIPE)
    for line in p.stdout:
            print(line.decode().strip()) # print output of sevn

# copy the run_scripts directory with the results
shutil.copytree(path_to_run_scripts, f'{path}{df_name}/run_scripts', dirs_exist_ok=True)
print('Sevn output copied')


