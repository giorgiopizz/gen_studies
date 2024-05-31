import uproot
import awkward as ak
from analysis.utils import create_components
import hist
import gc


def process(chunk, branches, rwgts, ops, object_definitions, get_variables, selections):
    events = uproot.concatenate(**chunk, filter_name=branches)

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

    # Define the Physical objects you want to work with
    events = object_definitions(events)

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
