import concurrent.futures
import os
import sys

import uproot

from gen_studies.plot.utils import final_bkg_plot


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821
    samples = analysis_dict["samples"]

    get_regions = analysis_dict["get_regions"]
    get_variables = analysis_dict["get_variables"]
    systematics = analysis_dict["systematics"]

    plots = analysis_dict["plots"]
    scales = analysis_dict["scales"]
    lumi = analysis_dict["lumi"]
    plot_ylim_ratio = analysis_dict["plot_ylim_ratio"]

    variables = get_variables()
    regions = get_regions()

    input_file = uproot.open("histos.root")

    os.makedirs("plots", exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        tasks = []
        # for variable in variables:
        for variable in list(variables.keys())[:1]:
            for region_name in regions:
                for plot_name in plots:
                    if "formatted" in variables[variable]:
                        formatted = "$" + variables[variable]["formatted"] + "$"
                    else:
                        formatted = variable + ""

                    # Do not overwrite
                    _variable = variable.replace(":", "_")

                    for scale in scales:
                        tasks.append(
                            pool.submit(
                                final_bkg_plot,
                                input_file,
                                samples,
                                plot_name,
                                plots,
                                region_name,
                                _variable,
                                systematics,
                                scale,
                                formatted,
                                lumi,
                                plot_ylim_ratio,
                            )
                        )
        concurrent.futures.wait(tasks)
        for task in tasks:
            print(task.result())


if __name__ == "__main__":
    main()
