# LHE studies

## Getting started

### Micromamba with python3.10
Use micromamba, it's fast. If you don't have it yet:
run:

`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

and configure it (the default options are ok) and activate micromamba (`source ~/.bashrc`).

Once micromamba is set up:

`micromamba create -f env.yaml`

will create the environment `(lhe)` used for this repo.

Activate it with:

<!-- `micromamba activate lhe`. -->
`source setup.sh`

This source will be needed everytime the environment is changed or you logout.

You can source from whatever directory e.g.:

`cd analysis; source ../setup.sh` 

### Install gen_studies
It's sufficient to run 
`pip install -e .` in gen_studies


## Analysis configuration
To run an analysis write a configuration python file like `configs/osww/config.py` (always under the `configs` folder) where you define:
* `lumi` 
* `samples` dictionary where each key is a sample and has the structure:
    * `xs` 
    * `files_pattern`: pattern to be used with glob to get all the files
    * `limit_files`: the max number of files to process
    * `nevents_per_file`: the number of events per root file
    * `nevents_per_job`: sort of chunksize, will concatenate enough files to get the requested number of events in each call to process
    * `eft`: can be an empty dict for samples with no eft
        * `reweight_card`: path to the reweight card to parse
        * `ops`: list of active operators (subset of the ones specified in the reweight_card)
* `branches`: the subset of branches that will be read from all the root files 
* `object_selections`: a function that takes the events (as awkward array) and creates all the collections and columns needed for your analysis
* `selections`: a function that takes the events and returns a subset of them based on some cuts
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

* `scales`
* `get_plot(op)`



## Analysis
Run in the `configs/analysis_name/` folder the analysis with `run-analyis` 

## Plot
Run in the `configs/analysis_name/` folder the plots with `run-plot` 
