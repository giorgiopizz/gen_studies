import concurrent.futures  # noqa: F401
import os
import sys

import uproot
from gen_studies.fit.utils import make_datacard


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821

    get_regions = analysis_dict["get_regions"]
    get_variables = analysis_dict["get_variables"]
    systematics = analysis_dict["systematics"]
    structures = analysis_dict["structures"]

    variables = get_variables()
    regions = get_regions()

    input_file = uproot.open("histos.root")

    os.makedirs("datacards", exist_ok=True)

    # for variable in variables:
    for variable in list(variables.keys())[:]:
        for region_name in regions:
            for structure_name in structures:
                # Do not overwrite
                _variable = variable.replace(":", "_")
                make_datacard(
                    input_file,
                    region_name,
                    _variable,
                    systematics,
                    structure_name,
                    structures,
                )


if __name__ == "__main__":
    main()
