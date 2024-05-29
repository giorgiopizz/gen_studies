# LHE studies

## Analysis configuration
To run an analysis write a configuration python file like `analysis/osww.py` (always under the analysis folder) where you define:
* `xs` 
* `lumi` 
* `reweight_card`: path to the reweight card to parse
* `files_pattern`: pattern to be used with glob to get all the files
* `limit_files`: the max number of files to process
* `nevents_per_file`: the number of events per root file
* `nevents_per_job`: sort of chunksize, will concatenate enough files to get the requested number of events in each call to process
* `ops`: list of active operators (subset of the ones specified in the reweight_card)


## Analysis
Run `analysis/analysis.py` with `python analysis.py analysis_name` 

## Plot
Run `plot/plot.py` with `python plot.py analysis_name` 



