# ruff: noqa: F841
import os
import sys


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821

    lumi = analysis_dict["lumi"]
    samples = analysis_dict["samples"]
    get_regions = analysis_dict["get_regions"]
    get_variables = analysis_dict["get_variables"]
    systematics = analysis_dict["systematics"]

    plots = analysis_dict["plots"]
    scales = analysis_dict["scales"]
    scales = analysis_dict["scales"]
    plot_ylim_ratio = analysis_dict["plot_ylim_ratio"]

    structures = analysis_dict["structures"]
    structures_ops = analysis_dict["structures_ops"]
    combine_path = analysis_dict["combine_path"]
    npoints_fit_1d = analysis_dict["npoints_fit_1d"]
    npoints_fit_2d = analysis_dict["npoints_fit_2d"]

    print("Checks passed")


if __name__ == "__main__":
    main()
