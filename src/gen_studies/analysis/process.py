import gc

import awkward as ak
import hist
import uproot

from gen_studies.analysis.utils import create_components


def read_events(chunk, branches):
    """
    Read events given a chunk and the list of branches.

    If key "files" is in chunk concatenate the files.

    If key "file" is in chunk: open single file from start to stop
    should provide a file with start and stop
    chunk structure is:
    ```python
    {
        "file": "/gwteras/cms/store/user/nanoAOD_100.root",  # required
        "tree": "Events",  # optional, default Events
        "start": 0,
        "stop": 100,
        "open_options": {  # optional dict
            "num_workers": 2
        },
        "arrays_options": {  # optional dict
            "decompression_executor": (uproot.source.
                                      futures.TrivialExecutor()),
            "interpretation_executor": (uproot.source.
                                      futures.TrivialExecutor()),
        },
    }
    ```


    Parameters
    ----------
    chunk : dict
        chunk dictionary
    branches : list
        list of branches name to read

    Returns
    _______
    events: ak.Array
        the events read from file(s)
    """
    if "files" in chunk:
        return uproot.concatenate(**chunk, filter_name=branches)
    elif "file" in chunk:
        filename = chunk.pop("file")
        treename = chunk.get("tree", "Events")
        start = chunk.pop("start")
        stop = chunk.pop("stop")
        file = uproot.open(
            filename,
            **chunk.get("open_options", {}),
        )
        events = file[treename].arrays(
            filter_name=branches,
            entry_start=start,
            entry_stop=stop,
            **chunk.get("arrays_options", {}),
        )
        file.close()
        return events
    else:
        raise Exception(
            'Could not parse chunk, "files" nor "file" found', chunk
        )


def process(
    chunk,
    sample_name,
    branches,
    object_definitions,
    get_variables,
    selections,
    eft={},
):
    events = read_events(chunk, branches)

    if eft:
        rwgts = eft["rwgts"]
        ops = eft["ops"]

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

    nevents = len(events)
    sumw = ak.sum(events.genWeight)

    if eft:
        events = create_components(events, ops, rwgts)
    else:
        events["components"] = ak.zip({"sm": ak.ones_like(events.genWeight)})

    # Define the Physical objects you want to work with
    events = object_definitions(events)

    # variable definitions
    variables = get_variables()
    for variable_name in variables:
        if ":" in variable_name:
            variable_name1, variable_name2 = variable_name.split(":")
            events[variable_name1] = variables[variable_name]["func1"](events)
            events[variable_name2] = variables[variable_name]["func2"](events)
        else:
            events[variable_name] = variables[variable_name]["func"](events)

    # Create histograms
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

    # Select events
    events = selections(events)

    # Fill each histogram
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
        sample_name: {
            "nevents": nevents,
            "sumw": sumw,
            "histos": histos,
        }
    }
    del events
    gc.collect()
    return result
