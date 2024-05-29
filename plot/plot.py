import sys
import uproot
import numpy as np
import os
import concurrent.futures


if len(sys.argv) != 2:
    print("Should pass the name of a valid analysis, osww, ...", file=sys.stderr)
    sys.exit()

analysis_name = sys.argv[1]
fw_path = os.path.abspath("../")
sys.path.insert(0, fw_path)
from plot.utils import final_plot  # noqa: E402


exec(f"import analysis.{analysis_name} as analysis_cfg")

get_variables = analysis_cfg.get_variables  # type: ignore # noqa: F821
ops = analysis_cfg.ops  # type: ignore # noqa: F821
variables = get_variables()


input_file = uproot.open("../analysis/histos.root")

variables = ["mjj", "mll", "detajj", "dphijj", "ptj1"][:]
scales = ["lin", "log"][:]


os.makedirs("plots", exist_ok=True)


with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
    tasks = []
    for variable in variables:
        for op in ops:
            for scale in scales:
                tasks.append(
                    pool.submit(
                        final_plot,
                        input_file,
                        variable,
                        op,
                        scale,
                    )
                )
    concurrent.futures.wait(tasks)
