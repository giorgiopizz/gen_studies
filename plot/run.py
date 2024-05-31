import sys
import uproot
import os
import concurrent.futures
from plot.utils import final_plot

if len(sys.argv) != 2:
    print("Should pass the name of a valid analysis, osww, ...", file=sys.stderr)
    sys.exit()

analysis_name = sys.argv[1]

exec(f"import configs.{analysis_name} as analysis_cfg")


get_variables = analysis_cfg.get_variables  # type: ignore # noqa: F821
ops = analysis_cfg.ops  # type: ignore # noqa: F821
variables = get_variables()

input_file = uproot.open("../analysis/histos.root")

# ops = ops[:1]
scales = ["lin", "log"]

os.makedirs("plots", exist_ok=True)

with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
    tasks = []
    for variable in variables:
        for op in ops:
            if "formatted" in variables[variable]:
                formatted = "$" + variables[variable]["formatted"] + "$"
            else:
                formatted = variable + ""

            # Do not overwrite
            _variable = variable.replace(":", "_")

            for scale in scales:
                tasks.append(
                    pool.submit(
                        final_plot,
                        input_file,
                        _variable,
                        op,
                        scale,
                        formatted,
                    )
                )
    concurrent.futures.wait(tasks)
