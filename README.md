# latex
The `latex` folder contain the `thesis.tex` file and all the auxiliary files needed for the compilation of the **thesis**. Inside the `fonti` folder there are also all the pdfs used as bibliography

# script
The `script` folder contains all the **python** and **jupyter-notebook scripts** used for the analysis of the **SEVN2** outputs





## Pipeline for output anaylisis
The pipeline must be executed in this order or some scripts won't work.


### 1. Simulate and store the SEVN2 simulations

Use the **SEVN2** software to explore the parameter space. For each combination of parameters, one million of binaries are simulated with SEVN2 in a folder named like: `SEVN2-version_Nsim_ZZ_SN_kick`. For instance the fiducial simulation is ran in the folder `SEVN2-3.0.0-Spindevel_RLO_Z015_com_unified265`, where the parameters are:
  - `version` of **SEVN2** taken from the `3.0.0-Spindevel` **branch** where the `_RLO` means it already had implemented the option to choose between optimistic and pessimistic scenario for a Hertzsprung-Gap star to survive a Common Envelope. All simulations used in this thesis use this version of SEVN2
  - `Nsim` **number of binaries simulated**, in this thesis always `1mln` i.e. one million
  - `Z` **metallicity** of the stars, chosen as `Z=0.015` as fiducial. Other simulations used `Z=0.02`
  - `SN` model of **core-collapse supernova** chosen as `com` for *compact* in the fiducial case. Other simulations used `rap` for *rapid* and `del` for *delayed*
  - `kick` model that combines the **kick model** `unified` with **random kick amplitude extracted from a Maxwellian with standard deviation** of `265` km/s. Other kick models include `hobbspure` and `hobbs` , respectively the *hobbs_pure* and *hobbs* option in SEVN2 without and with fallback. Options for the standard deviation of the Maxwellian included also `70` km/s but only for the `hobbspure` option


### 2. Reduce the full output of SEVN2

The first part of the pipeline is usually carried out in the server where the simulations ran and the output folder is then copied in the personal pc for further analysis and plots.
  - The output analysis is performed by manually inserting the correct parameters in the `results.py` script, that among the `read.py` and `WRBH_select.py` needs to be placed in the same folder that contains the `SEVN2-version_Nsim_ZZ_SN_kick` folder. The pipeline will create sub-dataframes with only the rows of interest and will store them in an ad-hoc created folder whose name identifies the parameter space of the simulation e.g. `1mln_Z015_com_unified265`.
  - If the `read` option in `results.py` is set to true, the script calls the `read.py` (that eventually calls the `WRBH_select.py` functions) to read the original output file `output_0.csv` and **select only the rows containing Wolf-Rayet - Black hole, Black-Hole - Black-Hole and Black-Hole - Neutron  Star systems**. The extracted rows are stored in three dataframes `WRBH.csv`,`BHBH.csv` and `BHNS.csv` in the `dataframes` folder, with a path similar to `1mln_Z015_com_unified265/dataframes/WRBH.csv`. Moreover, `read.py` copies the `run_scripts` files (expect for the huge `output0.csv` file) into the newly created `copied` directory to have a **backup of the initial conditions of the simulations**.
  - Once the `WRBH.csv`,`BHBH.csv` and `BHNS.csv` are created, the `results.py` calls the functions inside `WRBH_select.py` to identify the Cyg X-3 candidates and other properties of interest and **extracts sub-dataframes, classifying and storing them according to the binary evolution stage** (progenitors, remnants, initial or final stage of either the WR-BH phase or of the Cyg X-3 phase) **and their final fate** (Black-Hole-Black-Hole or Black-Hole-Neutron Star; bound or broken or merging within a Hubble time). **The results of such classification is reported in a verbose form in the** `result.txt` **file** in the `dataframes` folder.


### 3. Further output reduction to remove Black-Holes with mass < 3 Msun

**Make sure** that the `1mln_Z015_com_unified265` - like folder, **output of the first part of the pipeline, has been manually copied into a folder that has the same path of the remaining python scripts and is identified with the version of the SEVN2 adopted**. For instance the `results.txt` file of the above example should now have this path from the working directory: `./v_3.0.0.0-Spindevel_RLO/1mln_Z015_com_unified/dataframes/results.txt`.
  - Similar to the first reduction phase, put the correct parameter space as input to the `results2.py` script and run it. The script will call the `read2.py` and `WRBH_select2.py` scripts to read the `WRBH.csv`,`BHBH.csv` and `BHNS.csv` dataframes and re-analyze them; removing or selecting only the Black-Holes that have a mass < 3 Msun if the new parameter `ppisn` is respectively set to `only` or to `without`. At the end, a new folder `ppisn_pissnparametervalue` will be created along the `dartaframes` one, e.g. `1mln_Z015_unified265/ppisn_without/results.txt` could be the path to new new, cleaned, `results.txt` file
  - Note that these spurious black holes originate only in the `compact` CCSN model, but the creation of a `ppisn_without` folder is necessary to make the following scripts to work, therefore for the simulations with `rap` and `del` it is mandatory to manually rename the `dataframes` folders into `ppisn_without` folders (avoid running the `results2.py` script that is expensive and would produce the same dataframes)
    

### 4. Re-simulate selected binaries

The so-far reduced outputs only contain the Wolf-Rayet-Black hole timesteps, with the addition of their progenitor and remnant configuration. To, instead, have also **all the evolutionary timesteps** e.g. of the Cyg X-3 candidates it is necessary to **re-simulate those binaries with SEVN2** and plot all the timesteps. To do so use the `resimulate.py` script that:
  - calls the `seed_select.py` that reads the `evolved_0.dat` file and cuts it to create a `listBin.dat` file with initial conditions only of the binary of interest
  - copies the `listBin.dat` file into the SEVN2 repository, runs it and copies the output of the simulation in a new folder in the already existing folder that identifies the simulation. For instance in the path `./v_3.0.0-Spindevel_RLO/1mln_Z015_com_unified265/ppisn_without/BHBH_GW_WRBH_cyg_x-3--Ko17/run_scripts/sevn_output/output_0.csv` it will be possible to find the result of the re-simulation of Cyg X-3 candidates as with the Koljonen+2017 criteria the merge within a Hubble time with the fiducial model.
**Warning: The** `run.sh` **file of SEVN2 needs to be *manually checked* to be sure that the binaries are re-simulated with the same parameters desired.**
 

### 5. Plots

Once a detailed output like `./v_3.0.0-Spindevel_RLO/1mln_Z015_com_unified265/ppisn_without/BHBH_GW_WRBH_cyg_x-3--Ko17/run_scripts/sevn_output/output_0.csv` is created, it is possible to run:
  - `probability.ipynb` for plotting the evolution of **all Cyg X-3 candidates and histogram2d the merging binary black holes**
  - `singlebinary.ipynb` for plotting in the detail the evolution of a **single binary per time**

### 6. Other scripts

Short description of the additional anaylsis tools:
  - `SNhistogram.ipynb` allows plotting the **remnant mass distributions with different CCSN models**, eventually correlating them with properties like the initial mass, CO core masse or pre-SN mass. **It requires to have sets of single stellar evolution tracks** that were manually simulated and stored in paths like `./singlestar/Z015_com/sevn_output/output_0.csv`
  -`parsec.ipynb` allows to select a **single star in the parsec tables** and plot it. Each line in those tables correspond to a single star evolved with parsec and their properties are the datapoints used by SEVN2 to interpolate intermediate stellar tracks.
  -`convert.py` allows simple **conversions for binary properties and units of measure**, e.g. to rapidly obtain semimajor in solar radii given the masses in solar masses and the orbital period in hours.
  -`maxwell.py` uses the `scipy` library to **obtain the standard deviation of a Maxwellian given its median value**

