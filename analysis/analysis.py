import glob
import itertools
import uproot
import hist
import vector
import awkward as ak
import numpy as np

vector.register_awkward()


def add_dict(d1, d2):
    if isinstance(d1, dict):
        d = {}
        common_keys = set(list(d1.keys())).intersection(list(d2.keys()))
        for key in common_keys:
            d[key] = add_dict(d1[key], d2[key])
        for key in d1:
            if key in common_keys:
                continue
            d[key] = d1[key]
        for key in d2:
            if key in common_keys:
                continue
            d[key] = d2[key]

        return d
    elif isinstance(d1, np.ndarray):
        return np.concatenate([d1, d2])
    elif isinstance(d1, ak.highlevel.Array):
        return ak.concatenate([d1, d2])
    else:
        return d1 + d2


def add_dict_iterable(iterable):
    tmp = -99999
    for it in iterable:
        if tmp == -99999:
            tmp = it
        else:
            tmp = add_dict(tmp, it)
    return tmp


def read_ops(filename):
    with open(filename) as file:
        lines = file.read().split("\n")
    lines = list(
        filter(lambda k: k.startswith("#"), lines)
    )  # only take comments and not mixed terms (',')
    rwgts = {}
    ops = []

    for irwgt, line in enumerate(lines):
        if "sm" in line.lower():
            _, op, _ = line.split(" ")
            rwgts["sm"] = irwgt
        elif "," not in line:
            _, op_val, _ = line.split(" ")
            op, _ = op_val.split("=")
            ops.append(op)
            # res.append([op, val, rwgt])
            rwgts[op_val] = irwgt
        else:
            splitted = line.split(" ")
            ops_val = " ".join(splitted[1:-1])
            # res.append([op, val, rwgt])
            rwgts[ops_val] = irwgt
    return list(set(ops)), rwgts


def create_components(events, active_ops, rwgts):
    weights = events["LHEReweightingWeight"]
    new_weights = {}
    new_weights["sm"] = ak.copy(weights[:, rwgts["sm"]])
    for op in active_ops:
        # make sm_lin_quad
        new_weights[f"sm_lin_quad_{op}"] = ak.copy(weights[:, rwgts[f"{op}=1"]])
        # make linear
        new_weights[f"lin_{op}"] = 0.5 * (
            weights[:, rwgts[f"{op}=1"]] - weights[:, rwgts[f"{op}=-1"]]
        )
        # make quad
        new_weights[f"quad_{op}"] = 0.5 * (
            weights[:, rwgts[f"{op}=1"]]
            + weights[:, rwgts[f"{op}=-1"]]
            - 2 * new_weights["sm"]
        )
    for op1, op2 in list(itertools.combinations(active_ops, 2)):
        _op1, _op2 = op1, op2
        rwgt_key = f"{op1}=1, {op2}=1"
        if rwgt_key not in rwgts:
            rwgt_key = f"{op2}=1, {op1}=1"
            _op1, _op2 = op2, op1
        new_weights[f"mixed_{_op1}_{_op2}"] = (
            weights[:, rwgts[rwgt_key]]
            - new_weights["sm"]
            - new_weights[f"lin_{op1}"]
            - new_weights[f"quad_{op1}"]
            - new_weights[f"lin_{op2}"]
            - new_weights[f"quad_{op2}"]
        )
    events["components"] = ak.zip(new_weights)
    return events


right_xs = 0.20933649499999987


ops, rwgts = read_ops(
    "/gwpool/users/santonellini/eft/genproductions/bin/MadGraph5_aMCatNLO/folder_osww_dim6_cpodd/osww_dim6_cpodd_reweight_card.dat"
)
# print(rwgts)

files = glob.glob("/gwteras/cms/store/user/gpizzati/sara/lhe_root/final*.root")[:30]
# files = [
#     f"/gwteras/cms/store/user/gpizzati/sara/lhe_root/out_{i}.root" for i in range(4)
# ]
# files = glob.glob("../root/*.root")[:1]

particle_branches = ["pt", "eta", "phi", "mass", "pdgId", "status"]
branches = [f"Particle_{k}" for k in particle_branches] + [
    "genWeight",
    "LHEReweightingWeight",
]

variables = {
    "mjj": {
        "func": lambda events: (events.Jet[:, 0] + events.Jet[:, 1]).mass,
        "axis": hist.axis.Regular(30, 200, 3000, name="mjj"),
    },
    "detajj": {
        "func": lambda events: abs(events.Jet[:, 0].deltaeta(events.Jet[:, 1])),
        "axis": hist.axis.Regular(30, 2.5, 8, name="mjj"),
    },
    "dphijj": {
        "func": lambda events: abs(events.Jet[:, 0].deltaphi(events.Jet[:, 1])),
        "axis": hist.axis.Regular(30, 0, np.pi, name="mjj"),
    },
    "ptj1": {
        "func": lambda events: events.Jet[:, 0].pt,
        "axis": hist.axis.Regular(30, 30, 150, name="mjj"),
    },
    "events": {
        "func": lambda events: ak.ones_like(events.genWeight),
        "axis": hist.axis.Regular(1, 0, 2, name="events"),
    },
}


results = {}
for events in uproot.iterate({k: "Events" for k in files}, filter_name=branches):
    # create histograms for this iteration
    histos = {}
    for variable_name in variables:
        histos[variable_name] = hist.Hist(
            variables[variable_name]["axis"],
            hist.axis.StrCategory([], name="component", growth=True),
            hist.storage.Weight(),
        )

    # # FIXME
    # events["genWeight"] = 0.20881501800000024 / 1000
    nevents = len(events)

    sumw = ak.sum(events.genWeight)
    # events = create_components(events, ["cWtil", "cHWtil"], rwgts)
    events = create_components(events, ops, rwgts)

    # FIXME
    events[("components", "tot")] = ak.ones_like(events.genWeight)

    print(events["components"]["sm"] * events["genWeight"])

    particle = ak.zip(
        {k: events[f"Particle_{k}"] for k in particle_branches}, with_name="Momentum4D"
    )
    particle = particle[particle.status == 1]
    neutrinos_gen = particle[
        (
            (abs(particle.pdgId) == 12)
            | (abs(particle.pdgId) == 14)
            | (abs(particle.pdgId) == 16)
        )
    ]

    particle = particle[
        (
            (abs(particle.pdgId) != 12)
            & (abs(particle.pdgId) != 14)
            & (abs(particle.pdgId) != 16)
        )
    ]

    # neutrinos_d = {k: events[f'Particle_{k}'] for k in particle_branches if k != 'mass' and k != 'eta'}
    # neutrinos_d['eta'] = ak.zeros_like(neutrinos_d['pt'])
    # neutrinos_d['mass'] = ak.zeros_like(neutrinos_d['pt'])
    # neutrinos = ak.zip(
    #     neutrinos_d,
    #     with_name='Momentum4D'
    # )

    events["Lepton"] = particle[:, [0, 1]]
    events["Jet"] = particle[:, -2:]
    sumv = (
        events["Jet"][:, 0]
        + events["Jet"][:, 1]
        + events["Lepton"][:, 0]
        + events["Lepton"][:, 1]
    )
    neutrinos_d = {
        "px": -sumv.px,
        "py": -sumv.py,
        "pz": ak.zeros_like(sumv.pz),
        "mass": ak.zeros_like(sumv.pz),
    }
    events["MET"] = ak.zip(neutrinos_d, with_name="Momentum4D")
    events["GenMET"] = neutrinos_gen[:, 0] + neutrinos_gen[:, 1]


    # # print(repr(events['Lepton']))
    # print(repr(events["MET"][:].pt))
    # print(repr(events["MET"][:].phi))
    # print(repr(events["MET"][:].mass))
    # print("\n")
    # print(repr(events["GenMET"][:].pt))
    # print(repr(events["GenMET"][:].phi))
    # print(repr(events["GenMET"][:].mass))
    # # print(repr(events['Jet']))

    # print(ak.num(events["Lepton"]))
    # print(ak.num(events["Jet"]))
    

    # variable definitions
    for variable_name in variables:
        events[variable_name] = variables[variable_name]["func"](events)

    # selections
    events = events[(
        ((events.Jet[:, 0].pt > 30.0) & (events.Jet[:, 1].pt > 30.0))
        & (events.detajj >= 2.5)
        & (events.mjj >= 200)
        )]

    for variable_name in variables:
        for component_name in ak.fields(events.components):
            weight = events["genWeight"] * events["components"][component_name]
            histos[variable_name].fill(
                events[variable_name],
                component=component_name,
                weight=weight,
            )
    result = {
        "nevents": nevents,
        "sumw": sumw,
        "histos": histos,
    }
    results = add_dict(results, result)

lumi = 100 # fb^-1
scale = (
    right_xs * 1000 * lumi / results["sumw"]
)  # scale histos to xs in fb, might add lumi

h = list(results["histos"].values())[0]
components = [h.axes[1].value(i) for i in range(len(h.axes[1].centers))]
out = uproot.recreate("histos.root")
for variable_name in variables:
    for component in components:
        print(component)
        _h = results["histos"][variable_name][:, hist.loc(component)].copy()
        a = _h.view(True)
        a.value = a.value * scale
        a.variance = a.variance * scale * scale
        out[f"{variable_name}/histo_{component}"] = _h

print(ops)
# print(components)
