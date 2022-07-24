import numpy as np
import pandas as pd
import os

# function to generate initial conditions with same seed from a larger sample
# [version] is the SEVN2 version adopted                            e.g. '3.0.0-Spindevel'
# [Nsim] is the number of simulations                               e.g. '1mln'
# [Z] is the metallicity of the stars                               e.g. '02' for Z=0.02 solar metallicity
# [SN] is the SN explosion prescription adopted                     e.g. 'del' for delayed or 'com' for compact
# [df_name] name of the type of dataframe       e.g. 'BHBH_GW_WRBH_cyg_x-3'


def GenerateBin(version,Nsim,Z,SN,kick,ppisn,df_name):
        print('Generating the listBin.dat file with seeds')
        
        # Dataframe with binaries of interest
        path = f'./v_{version}/{Nsim}_Z{Z}_{SN}_{kick}/ppisn_{ppisn}/'         		# path to folder with all useful results
        path_to_df = f'{path}progenitors/p_{df_name}.csv'
        df = pd.read_csv(path_to_df)

        # create new folder to store results
        os.makedirs(path+df_name, exist_ok=True)

        # Select only columns of interest
        cols = df.columns.to_list()
        filtered = df[cols[4:-2]]
       
 
        # Filter and write the listBin.dat file with the SEN2 input syntax
        path_to_listBin = f'~/Scrivania/sevn/SEVN2-{version}/run_scripts/listBin.dat'
        if os.path.exists(path_to_listBin):
                os.remove(path_to_listBin)
        filtered.to_csv(path_to_listBin,sep='\t',index=False,header=False)
        
        if len(filtered.index) == 0:
        	print('##############################')
        	print('No binaries to re-simulate')
        	print('##############################')
        	runsevn = 'no'
       	else:
       		runsevn = 'yes'
       	return runsevn
        
        
def ExtractIC(path):
        df = pd.read_csv(path,sep='\t')
        print(df)

        # Select only columns of interest
        cols = df.columns.to_list()
        filtered = df[cols[4:-2]]
        filtered.to_csv('filteredIC.dat',sep='\t',index=False, header=False)




####################################
#ExtractIC('./v_3.0.0-Spindevel_RLO/1mln_Z015_com_unified265/dataframes/strange/strangeprog.dat')


        
# example ready to be run

# parameters to identify simulation
#Nsim, Z, SN = '1mln','02','com'
#version = '3.0.0-Spindevel'
#df_name = 'BHBH_GW_WRBH_cyg_x-3'

#seed_select.Generate(version,Nsim,Z,SN,df_name)



####################################################################
####################################################################
# MANUALLY MODIFY AN INPUT BINARY FILE ####
####################################################
path_listBin = 'listBin.dat'
df = pd.read_csv(path_listBin, sep='\s+', header=None)
df[1] = '0.02'
df[6] = '0.02'
df[3] = 'compact'
df[8] = 'compact'
df.to_csv('listBin2.dat',sep='\t',index=False,header=False)




####################################################################
####################################################################
# MANUALLY MODIFY AN INPUT SSE FILE ####
####################################################
##path_listStar = 'listStar.dat'
##df = pd.read_csv(path_listStar, sep='\s+', header=None)
##df[0] = [str(el) + 'HE' for el in df[0]]
##df[1] = '0.015'
##df[3] = 'rapid'
##df[4] = 'cheb'
##df.to_csv('listStar2.dat',sep='\t',index=False,header=False)



