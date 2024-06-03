import concurrent.futures
import glob
import os
import sys
from math import ceil

import hist
import uproot
import vector

from gen_studies.analysis.process import process
from gen_studies.analysis.utils import (
    add_dict,
    add_dict_iterable,
    hist_fold,
    hist_unroll,
    read_ops,
)

vector.register_awkward()


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821
    get_variables = analysis_dict["get_variables"]
    selections = analysis_dict["selections"]
    lumi = analysis_dict["lumi"]
    branches = analysis_dict["branches"]
    object_definitions = analysis_dict["object_definitions"]
    selections = analysis_dict["selections"]
    runner = analysis_dict["runner"]
    samples = analysis_dict["samples"]

    results = {}
    print("Running analysis")
    for sample_name in samples:
        print(sample_name)

        xs = samples[sample_name]["xs"]
        files_pattern = samples[sample_name]["files_pattern"]
        limit_files = samples[sample_name]["limit_files"]
        nevents_per_file = samples[sample_name]["nevents_per_file"]
        nevents_per_job = samples[sample_name]["nevents_per_job"]

        doEft = False

        if samples[sample_name]["eft"] != {}:
            doEft = True
            reweight_card = samples[sample_name]["eft"]["reweight_card"]
            ops = samples[sample_name]["eft"]["ops"]
            _, rwgts = read_ops(reweight_card)

        files = glob.glob(files_pattern)
        files = files[:limit_files]

        if len(files) == 0:
            print(
                "Could not find any files with the pattern provided",
                files_pattern,
                file=sys.stderr,
            )
            sys.exit(1)

        print("Should process", len(files), "files")
        print("for a total of", len(files) * nevents_per_file, "events")

        nfiles_per_job = ceil(nevents_per_job / nevents_per_file)
        njobs = ceil(len(files) / nfiles_per_job)

        process_args = (
            sample_name,
            branches,
            object_definitions,
            get_variables,
            selections,
        )

        if doEft:
            eft = {
                "rwgts": rwgts,
                "ops": ops,
            }
        else:
            eft = {}

        if runner["local"]:
            for ijob in range(njobs):
                start = ijob * nfiles_per_job
                stop = min((ijob + 1) * nfiles_per_job, len(files))
                chunk = dict(
                    files={k: "Events" for k in files[start:stop]},
                    num_workers=2,
                )

                results = add_dict(results, process(chunk, *process_args, eft))
                print(f"Done {ijob+1}/{njobs}")
        else:
            with concurrent.futures.ProcessPoolExecutor(
                runner.get("max_workers", 2)
            ) as pool:
                tasks = []
                for ijob in range(njobs):
                    start = ijob * nfiles_per_job
                    stop = min((ijob + 1) * nfiles_per_job, len(files))
                    chunk = dict(
                        files={k: "Events" for k in files[start:stop]},
                        num_workers=1,
                    )
                    tasks.append(
                        pool.submit(process, chunk, *process_args, eft)
                    )
                print("waiting for tasks")
                concurrent.futures.wait(tasks)
                print("tasks completed")
                results = add_dict(
                    results,
                    add_dict_iterable([task.result() for task in tasks]),
                )

        print(
            "\n\nDone",
            results[sample_name]["nevents"],
            "events for",
            sample_name,
        )

    variables = get_variables()
    out = uproot.recreate("histos.root")
    print("Postprocessing and saving histos")
    for sample_name in samples:
        print(sample_name)
        xs = samples[sample_name]["xs"]
        result = results[sample_name]
        scale = (
            xs * 1000.0 * lumi / result["sumw"]
        )  # scale histos to xs in fb, multiply by lumi and get number of events

        first_hist = list(result["histos"].values())[0]
        # Look for components
        components = [
            first_hist.axes[1].value(i)
            for i in range(len(first_hist.axes[1].centers))
        ]

        for variable_name in variables:
            for component in components:
                # print(component)
                if ":" in variable_name:
                    h = result["histos"][variable_name][
                        :, :, hist.loc(component)
                    ].copy()
                else:
                    h = result["histos"][variable_name][
                        :, hist.loc(component)
                    ].copy()
                # h is now the histogram we will be saving
                # will have to scale to xs and fold

                # scale in place
                a = h.view(True)
                a.value = a.value * scale
                a.variance = a.variance * scale * scale

                # fold in place
                hist_fold(h, variables[variable_name].get("fold", 3))

                if ":" in variable_name:
                    # will not unroll in place -> overwrite variable
                    h = hist_unroll(h)
                good_variable = variable_name.replace(":", "_")
                out[f"{good_variable}/histo_{sample_name}_{component}"] = h
        print("Saved components", ", ".join(components))
    out.close()


if __name__ == "__main__":
    main()
