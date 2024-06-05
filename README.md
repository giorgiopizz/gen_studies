# GEN studies

## Getting started

Clone the repo:

`git clone https://github.com/giorgiopizz/gen_studies`

Setup an environment with python3.10 (optional, see micromamba below).

To install gen_studies it's sufficient to run:

`pip install -e .` in the parent folder: `gen_studies`


## Analysis configuration
To run an analysis write a configuration python file like `configs/HH_lhe/config.py`
(always under the `configs` folder) where you define different configuration variables.

### General section
* `lumi` 
* `samples` dictionary where each key is a sample and has the structure:
    * `xs`: cross-section of the sample in $\textrm{pb}^{-1}$
    * `files_pattern`: pattern to be used with glob to get all the files
    * `limit_files`: the max number of files to process
    * `nevents_per_file`: the number of events per root file
    * `nevents_per_job`: sort of chunksize, will concatenate enough files to get the requested number of events in each call to process
    * `eft`: can be an empty dict for samples with no eft
        * `reweight_card`: path to the reweight card to parse
        * `ops`: list of active operators (subset of the ones specified in the reweight_card)

### Analysis section

* `branches`: the subset of branches that will be read from all the root files 
* `object_defintions`: a function that takes the events (as awkward array) and creates all the collections and columns needed for your analysis
* `get_regions`: a function that returns a dictionary with all the regions and corresponding function to select a subset of events based on some cuts
* `get_variables`: a function that returns a dictionary with all the variables. Each key (variable name) should have the following structure:
    ```python
    def get_variables():
        return {
            # 1D variable
            "detajj": {
                "func": lambda events: abs(events.Jet[:, 0].deltaeta(events.Jet[:, 1])),
                "axis": hist.axis.Regular(15, 2.5, 8, name="detajj"),
                "formatted": "\Delta\eta_{jj}", # optional, default is the variable name, a.k.a the key of the dict
                "fold": 2, # optional, default is 3
            },
            # 2D variable
            "mjj:ptj1": {
                "func1": lambda events: (events.Jet[:, 0] + events.Jet[:, 1]).mass,
                "axis1": hist.axis.Regular(10, 200, 3000, name="mjj"),
                "func2": lambda events: events.Jet[:, 0].pt,
                "axis2": hist.axis.Regular(6, 30, 150, name="ptj1"),
                "formatted": "m_{jj} \; [GeV] \; : \, p^T_{j1} \; [GeV]",
            }
        }
    ```
* `get_variations`: a function that returns a dictionary with all the variations. Each variation has :
    * `switches`: mapping of branches to be replaced 
    * `func`: function that takes the events and creates a new column(s) to be used in the variation. See the configs for example.
* `systematics`: a dictionary of the systematics of the analysis. They can take many variations as input and manipulate them

### Plot section
* `scales`: the scale to use for the plots (`lin` and `log`)
* `plot_ylim_ratio`: will set limits to the bottom pad, i.e. ratio, use `(None, None)` to let matplotlib figure out the limits
* `plots`: a dictionary of `plot` where each plot configures which samples to use and other options


### Fit section
* `combine_path`: path to the CMSSW with combine and AnalyticAnomaloutCoupling (e.g. `/gwpool/users/gpizzati/combine_clean/CMSSW_11_3_4/src`)
* `npoints_fit_1d`: number of points for the grid scan in combine for 1D
* `npoints_fit_2d`: number of points for the grid scan in combine for 2D (2 ops)
* `structures` a dictionary of `structure` where each structure configures which samples to use and which are the signals or data
* `structures_ops` a dictionary where for each structure above one defines the dictionary of operators with their ranges

## Scripts

### Check config
Run in the `configs/analysis_name/` folder to check the config `gs-check-config` 

### Analysis
Run in the `configs/analysis_name/` folder the analysis with `gs-analyis-run` 

### Plot
Run in the `configs/analysis_name/` folder the plots with `gs-plot-run` 
 
Run in the `configs/analysis_name/` folder the variation plots with `gs-plot-variations` 

### Fit
Run in the `configs/analysis_name/` folder the plots with `gs-fit-makecards`


Run in the `configs/analysis_name/` folder the plots with `gs-fit-run` 


Run in the `configs/analysis_name/` folder the plots with `gs-fit-plot` 



## Python environment: micromamba with python3.10
Use micromamba, it's fast. If you don't have it yet:
run:

`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

and configure it (the default options are ok) and activate micromamba (`source ~/.bashrc`).

Once micromamba is set up:

`micromamba create -n gen python=3.10`

will create the environment `(gen)` used for this repo.

Activate it with:

`micromamba activate gen`

This source will be needed everytime the environment is changed or you logout.

You can source from whatever directory.