# ruff: noqa: E501
import itertools
from typing import OrderedDict

import awkward as ak
import hist
import numpy as np
from gen_studies.analysis.utils import flatten_samples, read_ops
from gen_studies.plot.utils import cmap

# General config
lumi = 100.0  # fb^-1

runner = dict(
    local=True,
    max_workers=6,
)

samples = {}
samples["OSWW"] = dict(
    xs=1.090e-02,  # need to change
    files_pattern="/gwteras/cms/store/user/gpizzati/PrivateMC/triennali/osww/OSWW_dim6_cpodd/RunIISummer20UL18NanoAODv9_106X_upgrade2018_realistic_v11_nanoGEN_NANOAODSIM/240508_084113/*/*root",
    limit_files=10,
    nevents_per_file=10000,
    nevents_per_job=100000,  # need to change ?
    eft=dict(
        reweight_card="/gwpool/users/santonellini/eft/genproductions/bin/MadGraph5_aMCatNLO/folder_osww_dim6_cpodd/osww_dim6_cpodd_reweight_card.dat",
        ops=["cWtil", "cHWtil", "cHWBtil", "cHBtil", "ctBIm", "ctWIm"],
    ),
)

flat_samples = flatten_samples(samples)

particle_branches = ["pt", "eta", "phi", "mass", "pdgId", "status"]
branches = [f"LHEPart_{k}" for k in particle_branches] + [
    "genWeight",
    "LHEReweightingWeight",
    "LHEScaleWeight",
    "LHEPdfWeight",
]


def object_definitions(events):
    Particle = ak.zip(
        {k: events[f"LHEPart_{k}"] for k in particle_branches},
        with_name="Momentum4D",
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
    return events


def get_variables():
    return {
        "mjj": {
            "func": lambda events: (events.Jet[:, 0] + events.Jet[:, 1]).mass,
            "axis": hist.axis.Regular(15, 500, 3000, name="mjj"),
            "formatted": r"m_{jj} \; [GeV]",
        },
        "mll": {
            "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).mass,
            "axis": hist.axis.Regular(30, 20, 3000, name="mll"),
            "formatted": r"m_{ll} \; [GeV]",
        },
        "mjj:ptj1": {
            "func1": lambda events: (events.Jet[:, 0] + events.Jet[:, 1]).mass,
            "axis1": hist.axis.Regular(10, 200, 3000, name="mjj"),
            "func2": lambda events: events.Jet[:, 0].pt,
            "axis2": hist.axis.Regular(6, 30, 150, name="ptj1"),
            "formatted": r"m_{jj}\,:\,p^T_{j1}",
        },
        "detajj": {
            "func": lambda events: abs(events.Jet[:, 0].deltaeta(events.Jet[:, 1])),
            "axis": hist.axis.Regular(15, 2.5, 8, name="detajj"),
            "formatted": r"\Delta\eta_{jj}",
        },
        "dphijj": {
            "func": lambda events: abs(events.Jet[:, 0].deltaphi(events.Jet[:, 1])),
            "axis": hist.axis.Regular(30, 0, np.pi, name="dphijj"),
            "formatted": r"\Delta\phi_{jj}",
        },
        "ptj1": {
            "func": lambda events: events.Jet[:, 0].pt,
            "axis": hist.axis.Regular(30, 30, 150, name="ptj1"),
            "formatted": r"p^T_{j1} \; [GeV]",
        },
        "ptj2": {
            "func": lambda events: events.Jet[:, 1].pt,
            "axis": hist.axis.Regular(30, 30, 150, name="ptj2"),
            "formatted": r"p^T_{j2} \; [GeV]",
        },
        "ptl1": {
            "func": lambda events: events.Lepton[:, 0].pt,
            "axis": hist.axis.Regular(30, 25, 150, name="ptl1"),
            "formatted": r"p^T_{l1} \; [GeV]",
        },
        "ptl2": {
            "func": lambda events: events.Lepton[:, 1].pt,
            "axis": hist.axis.Regular(30, 20, 150, name="ptl2"),
            "formatted": r"p^T_{l2} \; [GeV]",
        },
        "ptll": {
            "func": lambda events: (events.Lepton[:, 0] + events.Lepton[:, 1]).pt,
            "axis": hist.axis.Regular(30, 20, 2000, name="ptll"),
            "formatted": r"p^T_{ll} \; [GeV]",
        },
        "etaj1": {
            "func": lambda events: events.Jet[:, 0].eta,
            "axis": hist.axis.Regular(30, 0, 5, name="etaj1"),
            "formatted": r"\eta_{j1}",
        },
        "etaj2": {
            "func": lambda events: events.Jet[:, 1].eta,
            "axis": hist.axis.Regular(30, 0, 5, name="etaj2"),
            "formatted": r"\eta_{j2}",
        },
        "etal1": {
            "func": lambda events: events.Lepton[:, 0].eta,
            "axis": hist.axis.Regular(30, 0, 2.5, name="etal1"),
            "formatted": r"\eta_{l1}",
        },
        "etal2": {
            "func": lambda events: events.Lepton[:, 1].eta,
            "axis": hist.axis.Regular(30, 0, 2.5, name="etal2"),
            "formatted": r"\eta_{l2}",
        },
        "phij1": {
            "func": lambda events: events.Jet[:, 0].phi,
            "axis": hist.axis.Regular(30, 0, np.pi, name="phij1"),
            "formatted": r"\phi_{j1}",
        },
        "phij2": {
            "func": lambda events: events.Jet[:, 1].phi,
            "axis": hist.axis.Regular(30, 0, np.pi, name="phij2"),
            "formatted": r"\phi_{j2}",
        },
        "events": {
            "func": lambda events: ak.ones_like(events.genWeight),
            "axis": hist.axis.Regular(1, 0, 2, name="events"),
        },
    }


def get_regions():
    def sr(events):
        return (
            ((events.Jet[:, 0].pt > 30.0) & (events.Jet[:, 1].pt > 30.0))
            & (abs(events.detajj) >= 2.5)
            & (events.mjj >= 500)
            & (events.mll >= 20)
            & (events.ptl1 >= 25)
            & (events.ptl2 >= 20)
            & (events.ptj1 >= 30)
            & (events.ptj2 >= 30)
            & (abs(events.Jet[:, 0].eta) < 5)
            & (abs(events.Jet[:, 1].eta) < 5)
            & (abs(events.Lepton[:, 0].eta) < 2.5)
            & (abs(events.Lepton[:, 1].eta) < 2.5)
        )

    return {
        "sr": sr,
    }


# samples_for_nuis = flat_samples
samples_for_nuis = [sample for sample in flat_samples if sample.endswith("sm")]


def get_variations():
    variations = {
        "nominal": {
            "switches": [],
            "func": lambda events: events,
            "samples": flat_samples,
        },
    }

    # QCDScales
    def wrapper(variation_idx, weight_idx):
        def func(events):
            events[f"weight_QCDScale_{variation_idx}"] = (
                events.genWeight[:] * events.LHEScaleWeight[:, weight_idx]
            )
            return events

        return func

    variation_idx = 0
    for weight_idx in [0, 1, 3, 4, 6, 7]:
        variations[f"QCDScale_{variation_idx}"] = {
            "switches": [
                ("genWeight", f"weight_QCDScale_{variation_idx}"),
            ],
            "func": wrapper(variation_idx, weight_idx),
            "samples": samples_for_nuis,
        }
        variation_idx += 1

    # PDF
    def wrapper(variation_idx, weight_idx):
        def func(events):
            events[f"weight_PDF_{variation_idx}"] = (
                events.genWeight[:] * events.LHEPdfWeight[:, weight_idx]
            )
            return events

        return func

    variation_idx = 0
    for weight_idx in range(1, 101):
        variations[f"PDF_{variation_idx}"] = {
            "switches": [
                ("genWeight", f"weight_PDF_{variation_idx}"),
            ],
            "func": wrapper(variation_idx, weight_idx),
            "samples": samples_for_nuis,
        }
        variation_idx += 1

    return variations


systematics = {
    "QCDScale": {
        "name": "QCDScale",
        "kind": "weight_envelope",
        "type": "shape",
        "samples": {
            skey: [f"QCDScale_{i}" for i in range(6)] for skey in samples_for_nuis
        },
    },
    "PDF": {
        "name": "PDF",
        "kind": "weight_square",
        "type": "shape",
        "samples": {
            skey: [f"PDF_{i}" for i in range(100)] for skey in samples_for_nuis
        },
    },
    "lumi": {
        "name": "lumi",
        "type": "lnN",
        "samples": dict((skey, "1.02") for skey in flat_samples),
    },
}


# Plot config
plot_label = "OSWW"
scales = ["lin", "log"][:1]
plot_ylim_ratio = (-0.5, 0.5)


plots = {}
for op in samples["OSWW"]["eft"]["ops"]:
    plot_name = f"1d_{op}"
    plots[plot_name] = OrderedDict()
    plot = plots[plot_name]
    colors = iter(cmap)

    plot["OSWW_sm"] = {
        "color": next(colors),
        "name": "OSWW",
    }

    plot[f"OSWW_lin_{op}"] = {
        "color": next(colors),
        "name": f"Lin {op}",
        "isSignal": True,
        "superimposed": True,
        "stacked": False,
    }

    plot[f"OSWW_quad_{op}"] = {
        "color": next(colors),
        "name": f"Quad {op}",
        "isSignal": True,
        "superimposed": True,
        "stacked": False,
    }

# Fit config
combine_path = "/gwpool/users/gpizzati/combine_clean/CMSSW_11_3_4/src"
npoints_fit_1d = 1000
npoints_fit_2d = 1000

structures = {}
structures_ops = {}
for op in samples["OSWW"]["eft"]["ops"]:
    structure_name = f"1d_{op}"
    structures[structure_name] = {}
    structures_ops[structure_name] = {op: [-10, 10]}

    structure = structures[structure_name]

    structure["OSWW_sm"] = {
        "name": "sm",
        "isSignal": False,
        "isData": False,
    }

    structure[f"OSWW_sm_lin_quad_{op}"] = {
        "name": f"sm_lin_quad_{op}",
        "isSignal": True,
        "isData": False,
        "noStat": True,
    }

    structure[f"OSWW_quad_{op}"] = {
        "name": f"quad_{op}",
        "isSignal": True,
        "isData": False,
        "noStat": True,
    }

_, rwgts = read_ops(samples["OSWW"]["eft"]["reweight_card"])
for ops in list(itertools.combinations(samples["OSWW"]["eft"]["ops"], 2)):
    _op1, _op2 = ops
    rwgt_key = f"{_op1}=1, {_op2}=1"
    if rwgt_key not in rwgts:
        _op1, _op2 = _op2, _op1
    structure_name = f"2d_{_op1}_{_op2}"
    structures[structure_name] = {}
    structures_ops[structure_name] = {
        _op1: [-6, 2],
        _op2: [-6, 3],
    }
    structure = structures[structure_name]

    structure["OSWW_sm"] = {
        "name": "sm",
        "isSignal": False,
        "isData": False,
    }

    for op in ops:
        structure[f"OSWW_sm_lin_quad_{op}"] = {
            "name": f"sm_lin_quad_{op}",
            "isSignal": True,
            "isData": False,
            "noStat": True,
        }

        structure[f"OSWW_quad_{op}"] = {
            "name": f"quad_{op}",
            "isSignal": True,
            "isData": False,
            "noStat": True,
        }

    structure[f"OSWW_sm_lin_quad_mixed_{_op1}_{_op2}"] = {
        "name": f"sm_lin_quad_mixed_{_op1}_{_op2}",
        "isSignal": True,
        "isData": False,
        "noStat": True,
    }
