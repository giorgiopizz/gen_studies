import os
import sys

import uproot


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821
    samples = analysis_dict["samples"]
    get_variables = analysis_dict["get_variables"]
    get_plot_dict = analysis_dict["get_plot"]
    scales = analysis_dict["scales"]
    lumi = analysis_dict["lumi"]
    plot_ylim_ratio = analysis_dict.get("plot_ylim_ratio")

    variables = get_variables()

    ops = []
    for sample_name in samples:
        if samples[sample_name]["eft"] != {}:
            ops = samples[sample_name]["eft"]["ops"]

    input_file = uproot.open("histos.root")

    os.makedirs("datacards", exist_ok=True)


if __name__ == "__main__":
    main()
