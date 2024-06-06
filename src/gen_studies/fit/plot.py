import os
import sys

import matplotlib as mpl
import uproot
from gen_studies.fit.utils import plot1d, plot2d

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

d = hep.style.CMS
plt.style.use([d, hep.style.firamath])


def main():
    os.makedirs("scans", exist_ok=True)
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821

    lumi = analysis_dict["lumi"]
    plot_label = analysis_dict["plot_label"]
    get_regions = analysis_dict["get_regions"]
    get_variables = analysis_dict["get_variables"]
    structures = analysis_dict["structures"]
    structures_ops = analysis_dict["structures_ops"]

    variables = get_variables()
    regions = get_regions()

    for variable in list(variables.keys())[:]:
        for region_name in regions:
            for structure_name in structures:
                _variable = variable.replace(":", "_")
                ops_ranges = structures_ops[structure_name]
                ops = list(ops_ranges.keys())
                path = f"datacards/{structure_name}/{region_name}/{_variable}"
                output_path = f"scans/{structure_name}/{region_name}/"
                os.makedirs(output_path, exist_ok=True)
                output_path += f"{_variable}.png"
                filename = "higgsCombineTest.MultiDimFit.mH125.root"
                file = uproot.open(f"{path}/{filename}")
                if len(ops) == 1:
                    plot1d(file, ops, lumi, output_path, plot_label)
                else:
                    plot2d(file, ops, lumi, output_path, plot_label)
                file.close()
