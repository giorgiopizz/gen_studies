# LHE studies

## Getting started
Use micromamba, it's fast. If you don't have it yet:
run:

`"${SHELL}" <(curl -L micro.mamba.pm/install.sh)`

and configure it (the default options are ok) and activate micromamba (`source ~/.bashrc`).

Once micromamba is set up:

`micromamba create -f env.yaml`

will create the environment `(lhe)` used for this repo.
Activate it with:

`micromamba activate lhe`.


## Analysis configuration
To run an analysis write a configuration python file like `configs/osww.py` (always under the `configs` folder) where you define:
* `xs` 
* `lumi` 
* `reweight_card`: path to the reweight card to parse
* `files_pattern`: pattern to be used with glob to get all the files
* `limit_files`: the max number of files to process
* `nevents_per_file`: the number of events per root file
* `nevents_per_job`: sort of chunksize, will concatenate enough files to get the requested number of events in each call to process
* `ops`: list of active operators (subset of the ones specified in the reweight_card)
* `process`: the process function that will take a chunk and produce histograms


## Analysis
Run `analysis/analysis.py` with `python analysis.py analysis_name` 

## Plot
Run `plot/plot.py` with `python plot.py analysis_name` 



