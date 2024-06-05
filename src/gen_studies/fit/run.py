import os
import subprocess
import sys


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821

    get_regions = analysis_dict["get_regions"]
    get_variables = analysis_dict["get_variables"]
    structures = analysis_dict["structures"]
    structures_ops = analysis_dict["structures_ops"]
    combine_path = analysis_dict["combine_path"]
    npoints_fit_1d = analysis_dict["npoints_fit_1d"]
    npoints_fit_2d = analysis_dict["npoints_fit_2d"]

    variables = get_variables()
    regions = get_regions()

    for variable in list(variables.keys())[:1]:
        for region_name in regions:
            for structure_name in structures:
                _variable = variable.replace(":", "_")
                ops_ranges = structures_ops[structure_name]
                ops = list(ops_ranges.keys())
                if len(ops) > 1:
                    npoints = npoints_fit_1d
                else:
                    npoints = npoints_fit_2d

                path = f"datacards/{structure_name}/{region_name}/{_variable}"
                print("Running in", path)
                command = f"cd {combine_path}; cmsenv; cd -;"
                command += f" cd {path}; "
                command += (
                    "text2workspace.py datacard.txt "
                    "-P HiggsAnalysis.AnalyticAnomalousCoupling."
                    "AnomalousCouplingEFTNegative:analiticAnomalousCouplingEFTNegative"
                    " -o ws.root --X-allow-no-signal --PO eftOperators="
                )
                command += ",".join(ops)
                command += ";"
                command += "combine -M MultiDimFit ws.root  --algo=grid "
                command += f"--points {npoints} -m 125 -t -1 "
                command += "--robustFit=1 --X-rtd FITTER_NEW_CROSSING_ALGO "
                command += "--X-rtd FITTER_NEVER_GIVE_UP --X-rtd FITTER_BOUND "
                command += "--redefineSignalPOIs "
                command += ",".join([f"k_{op}" for op in ops])
                command += (
                    " --freezeParameters r --setParameters r=1  --setParameterRanges "
                )
                command += ":".join(
                    [f"k_{op}={ops_ranges[op][0]},{ops_ranges[op][1]}" for op in ops]
                )
                command += " --verbose -1"

                print(command)
                proc = subprocess.Popen(command, shell=True)
                proc.wait()


if __name__ == "__main__":
    main()
