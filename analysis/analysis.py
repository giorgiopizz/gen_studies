import glob
from math import ceil
import uproot
import hist
import vector

import concurrent.futures
import sys
import os

analysis_name = sys.argv[1]
fw_path = os.path.abspath("../")
sys.path.insert(0, fw_path)

from analysis.utils import (  # noqa: E402
    add_dict,
    add_dict_iterable,
    hist_fold,
    hist_unroll,
    read_ops,
)

if len(sys.argv) != 2:
    print("Should pass the name of a valid analysis, osww, ...", file=sys.stderr)
    sys.exit()


exec(f"import configs.{analysis_name} as analysis_cfg")

xs = analysis_cfg.xs  # type: ignore # noqa: F821
reweight_card = analysis_cfg.reweight_card  # type: ignore # noqa: F821
files_pattern = analysis_cfg.files_pattern  # type: ignore # noqa: F821
limit_files = analysis_cfg.limit_files  # type: ignore # noqa: F821
nevents_per_file = analysis_cfg.nevents_per_file  # type: ignore # noqa: F821
nevents_per_job = analysis_cfg.nevents_per_job  # type: ignore # noqa: F821
get_variables = analysis_cfg.get_variables  # type: ignore # noqa: F821
selections = analysis_cfg.selections  # type: ignore # noqa: F821
lumi = analysis_cfg.lumi  # type: ignore # noqa: F821
ops = analysis_cfg.ops  # type: ignore # noqa: F821
process = analysis_cfg.process  # type: ignore # noqa: F821


vector.register_awkward()


_, rwgts = read_ops(reweight_card)


files = glob.glob(files_pattern)
files = files[:limit_files]


nfiles_per_job = ceil(nevents_per_job / nevents_per_file)
njobs = ceil(len(files) / nfiles_per_job)


local = True
if local:
    results = {}
    for ijob in range(njobs):
        start = ijob * nfiles_per_job
        stop = min((ijob + 1) * nfiles_per_job, len(files))
        chunk = dict(
            files={k: "Events" for k in files[start:stop]},
            num_workers=2,
        )
        results = add_dict(results, process(chunk, rwgts))
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        tasks = []
        for ijob in range(njobs):
            start = ijob * nfiles_per_job
            stop = min((ijob + 1) * nfiles_per_job, len(files))
            chunk = dict(
                files={k: "Events" for k in files[start:stop]},
                num_workers=1,
            )
            tasks.append(pool.submit(process, chunk, rwgts))
        print("waiting for tasks")
        concurrent.futures.wait(tasks)
        print("tasks completed")
        results = add_dict_iterable([task.result() for task in tasks])
    # print(results)


print("\n\n", "Done", results["nevents"], "events")
scale = (
    xs * 1000.0 * lumi / results["sumw"]
)  # scale histos to xs in fb, multiply by lumi and get number of events


first_hist = list(results["histos"].values())[0]
# Look for components
components = [
    first_hist.axes[1].value(i) for i in range(len(first_hist.axes[1].centers))
]
variables = get_variables()

out = uproot.recreate("histos.root")
for variable_name in variables:
    for component in components:
        # print(component)
        if ":" in variable_name:
            h = results["histos"][variable_name][:, :, hist.loc(component)].copy()
        else:
            h = results["histos"][variable_name][:, hist.loc(component)].copy()
        # _h is now the histogram we will be saving, will have to scale to xs and fold

        # scale in place
        a = h.view(True)
        a.value = a.value * scale
        a.variance = a.variance * scale * scale

        # fold in place
        hist_fold(h, variables[variable_name].get("fold", 3))

        if ":" in variable_name:
            # will not unroll in place -> overwrite variable
            h = hist_unroll(h)

        out[f"{variable_name.replace(':', '_')}/histo_{component}"] = h

out.close()
print(ops)
