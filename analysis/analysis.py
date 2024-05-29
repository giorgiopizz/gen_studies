import gc
import glob
from math import ceil
import uproot
import hist
import vector
import awkward as ak

import concurrent.futures
import sys
import os

if len(sys.argv) != 2:
    print("Should pass the name of a valid analysis, osww, ...", file=sys.stderr)
    sys.exit()

analysis_name = sys.argv[1]
fw_path = os.path.abspath("../")
sys.path.insert(0, fw_path)

from analysis.utils import (  # noqa: E402
    add_dict,
    add_dict_iterable,
    create_components,
    hist_fold,
    hist_unroll,
    read_ops,
)

exec(f"import {analysis_name} as analysis_cfg")

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


vector.register_awkward()


_, rwgts = read_ops(reweight_card)


files = glob.glob(files_pattern)
files = files[:limit_files]


nfiles_per_job = ceil(nevents_per_job / nevents_per_file)
njobs = ceil(len(files) / nfiles_per_job)

particle_branches = ["pt", "eta", "phi", "mass", "pdgId", "status"]
branches = [f"LHEPart_{k}" for k in particle_branches] + [
    "genWeight",
    "LHEReweightingWeight",
]


def process(chunk):
    events = uproot.concatenate(**chunk)

    nReweights = ak.num(events.LHEReweightingWeight)
    if not ak.all(nReweights == len(rwgts)):
        print("Error for chunk", chunk)
        print(
            "Wrong number of rwgts, expected",
            len(rwgts),
            "got",
            nReweights[nReweights != len(rwgts)],
        )
        return {}

    # create histograms for this iteration

    nevents = len(events)

    sumw = ak.sum(events.genWeight)
    events = create_components(events, ops, rwgts)

    Particle = ak.zip(
        {k: events[f"LHEPart_{k}"] for k in particle_branches}, with_name="Momentum4D"
    )
    Particle = Particle[Particle.status == 1]
    events["Particle"] = Particle

    # Define MET
    neutrinos_gen = events.Particle[
        (
            (abs(events.Particle.pdgId) == 12)
            | (abs(events.Particle.pdgId) == 14)
            | (abs(events.Particle.pdgId) == 16)
        )
    ]
    # events["MET"] = neutrinos_gen[:, 0] + neutrinos_gen[:, 1]
    events["MET"] = ak.sum(neutrinos_gen, axis=1)

    # Remove neutrinos from particle
    events["Particle"] = events.Particle[
        (
            (abs(events.Particle.pdgId) != 12)
            & (abs(events.Particle.pdgId) != 14)
            & (abs(events.Particle.pdgId) != 16)
        )
    ]

    # Define leptons no tau
    events["Lepton"] = events.Particle[
        (abs(events.Particle.pdgId) == 11) | (abs(events.Particle.pdgId) == 13)
    ]
    # Remove events where nleptons != 2
    events = events[ak.num(events.Lepton) == 2]

    # Define Jets
    # Everything that is not Lepton or Neutrino
    events["Jet"] = events.Particle[
        (
            (abs(events.Particle.pdgId) != 11)
            & (abs(events.Particle.pdgId) != 12)
            & (abs(events.Particle.pdgId) != 13)
            & (abs(events.Particle.pdgId) != 14)
            & (abs(events.Particle.pdgId) != 15)
            & (abs(events.Particle.pdgId) != 16)
        )
    ]

    variables = get_variables()
    # variable definitions
    for variable_name in variables:
        if ":" in variable_name:
            variable_name1, variable_name2 = variable_name.split(":")
            events[variable_name1] = variables[variable_name]["func1"](events)
            events[variable_name2] = variables[variable_name]["func2"](events)
        else:
            events[variable_name] = variables[variable_name]["func"](events)

    histos = {}
    for variable_name in variables:
        if ":" in variable_name:
            variable_name1, variable_name2 = variable_name.split(":")
            histos[variable_name] = hist.Hist(
                variables[variable_name]["axis1"],
                variables[variable_name]["axis2"],
                hist.axis.StrCategory([], name="component", growth=True),
                hist.storage.Weight(),
            )
        else:
            histos[variable_name] = hist.Hist(
                variables[variable_name]["axis"],
                hist.axis.StrCategory([], name="component", growth=True),
                hist.storage.Weight(),
            )

    events = selections(events)

    for variable_name in variables:
        for component_name in ak.fields(events.components):
            weight = events["genWeight"] * events["components"][component_name]

            if ":" in variable_name:
                variable_name1, variable_name2 = variable_name.split(":")
                kwargs = {
                    variable_name1: events[variable_name1],  # single variable
                    variable_name2: events[variable_name2],  # single variable
                    "component": component_name,
                    "weight": weight,
                }
            else:
                kwargs = {
                    variable_name: events[variable_name],  # single variable
                    "component": component_name,
                    "weight": weight,
                }
            histos[variable_name].fill(**kwargs)

    result = {
        "nevents": nevents,
        "sumw": sumw,
        "histos": histos,
    }
    del events
    gc.collect()
    return result


local = True
if local:
    results = {}
    for ijob in range(njobs):
        start = ijob * nfiles_per_job
        stop = min((ijob + 1) * nfiles_per_job, len(files))
        chunk = dict(
            files={k: "Events" for k in files[start:stop]},
            filter_name=branches,
            num_workers=4,
        )
        # tasks.append(pool.submit(process, chunk))
        results = add_dict(results, process(chunk))
else:
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as pool:
        tasks = []
        for ijob in range(njobs):
            start = ijob * nfiles_per_job
            stop = min((ijob + 1) * nfiles_per_job, len(files))
            chunk = dict(
                files={k: "Events" for k in files[start:stop]},
                filter_name=branches,
                num_workers=1,
            )
            tasks.append(pool.submit(process, chunk))
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
