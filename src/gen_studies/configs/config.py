# ruff: noqa: E501
import itertools
from typing import OrderedDict

import awkward as ak
import hist
from gen_studies.analysis.utils import flatten_samples, read_ops
from gen_studies.plot.utils import cmap

# General config
lumi = 300.0  # fb^-1

runner = dict(
    local=True,
    max_workers=6,
)

# dictionary where each key is a sample and has the structure
samples = {}
reweight_card = "/gwpool/users/tecedor/prova/genproductions/bin/MadGraph5_aMCatNLO/cards/folder_HHjj_smhloop0_dim6_12ops_CPV_mixed/HHjj_smhloop0_dim6_12ops_CPV_mixed_reweight_card.dat"

samples["Sample"] = dict(
    xs=1.12e-02,  # cross-section of the sample in $\textrm{pb}^{-1}$
    files_pattern="*root",  # pattern to be used with glob to get all the files
    limit_files=10,  # the max number of files to process
    nevents_per_file=5000,  # the number of events per root file
    nevents_per_job=100000,  # chunksize, will concatenate enough files
    eft=dict(  # can be an empty dict for samples with no eft
        # path to the reweight card to parse
        reweight_card=reweight_card,
        # list of active operators (subset of the ones specified in the reweight_card)
        ops=[
            "cH",
            "cHbox",
        ],
    ),
)

flat_samples = flatten_samples(samples)

# Analysis config
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

    # Define Higgs
    events["Higgs"] = events.Particle[events.Particle.pdgId == 25]

    # Define Jets
    # Everything that is not Lepton or Neutrino
    # Or Higgs
    events["Jet"] = events.Particle[
        (
            (abs(events.Particle.pdgId) != 11)
            & (abs(events.Particle.pdgId) != 12)
            & (abs(events.Particle.pdgId) != 13)
            & (abs(events.Particle.pdgId) != 14)
            & (abs(events.Particle.pdgId) != 15)
            & (abs(events.Particle.pdgId) != 16)
            & (abs(events.Particle.pdgId) != 25)
        )
    ]

    return events


def get_variables():
    return {
        "mjj": {
            "func": lambda events: (events.Jet[:, 0] + events.Jet[:, 1]).mass,
            "axis": hist.axis.Regular(50, 250, 3500, name="mjj"),
            "formatted": r"m_{jj} \; [GeV]",
        },
        "mhh:pth1": {
            "func1": lambda events: (events.Higgs[:, 0] + events.Higgs[:, 1]).mass,
            "axis1": hist.axis.Regular(10, 200, 3000, name="mhh"),
            "func2": lambda events: events.Higgs[:, 0].pt,
            "axis2": hist.axis.Regular(6, 30, 150, name="pth1"),
            "formatted": "m_{hh}:p^T_{h1}",
        },
        "detajj": {
            "func": lambda events: abs(events.Jet[:, 0].deltaeta(events.Jet[:, 1])),
            "axis": hist.axis.Regular(25, 1, 8, name="detajj"),
            "formatted": r"\Delta\eta_{jj}",
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
    }


def get_regions():
    def sr(events):
        return (
            ((events.Jet[:, 0].pt > 30.0) & (events.Jet[:, 1].pt > 30.0))
            & (abs(events.detajj) >= 2.5)
            & (events.mjj >= 150)
            & (events.ptj1 >= 30)
            & (events.ptj2 >= 30)
            & (abs(events.Jet[:, 0].eta) < 5)
            & (abs(events.Jet[:, 1].eta) < 5)
        )

    return {
        "sr": sr,
    }


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
        "type": "shape",
        "kind": "weight_envelope",
        "samples": {
            skey: [f"QCDScale_{i}" for i in range(6)] for skey in samples_for_nuis
        },
    },
    "PDF": {
        "name": "PDF",
        "type": "shape",
        "kind": "weight_square",
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
scales = ["lin", "log"][:1]
plot_ylim_ratio = (None, None)


plots = {}
for op in samples["Sample"]["eft"]["ops"]:
    plot_name = f"1d_{op}"
    plots[plot_name] = OrderedDict()
    plot = plots[plot_name]
    colors = iter(cmap)

    plot["Sample_sm"] = {
        "color": next(colors),
        "name": "Sample",
    }

    plot[f"Sample_lin_{op}"] = {
        "color": next(colors),
        "name": f"Lin {op}",
        "isSignal": True,
        "superimposed": True,
    }

    plot[f"Sample_quad_{op}"] = {
        "color": next(colors),
        "name": f"Quad {op}",
        "isSignal": True,
        "superimposed": True,
    }


# Fit config
combine_path = "/gwpool/users/gpizzati/combine_clean/CMSSW_11_3_4/src"
npoints_fit_1d = 1000
npoints_fit_2d = 5000

structures = {}
structures_ops = {}
for op in samples["Sample"]["eft"]["ops"]:
    structure_name = f"1d_{op}"
    structures[structure_name] = {}
    structures_ops[structure_name] = {op: [-10, 10]}

    structure = structures[structure_name]

    structure["Sample_sm"] = {
        "name": "sm",
        "isSignal": False,
        "isData": False,
    }

    structure[f"Sample_sm_lin_quad_{op}"] = {
        "name": f"sm_lin_quad_{op}",
        "isSignal": True,
        "isData": False,
        "noStat": True,
    }

    structure[f"Sample_quad_{op}"] = {
        "name": f"quad_{op}",
        "isSignal": True,
        "isData": False,
        "noStat": True,
    }

_, rwgts = read_ops(samples["Sample"]["eft"]["reweight_card"])
for ops in list(itertools.combinations(samples["Sample"]["eft"]["ops"], 2)):
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

    structure["Sample_sm"] = {
        "name": "sm",
        "isSignal": False,
        "isData": False,
    }

    for op in ops:
        structure[f"Sample_sm_lin_quad_{op}"] = {
            "name": f"sm_lin_quad_{op}",
            "isSignal": True,
            "isData": False,
            "noStat": True,
        }

        structure[f"Sample_quad_{op}"] = {
            "name": f"quad_{op}",
            "isSignal": True,
            "isData": False,
            "noStat": True,
        }

    structure[f"Sample_sm_lin_quad_mixed_{_op1}_{_op2}"] = {
        "name": f"sm_lin_quad_mixed_{_op1}_{_op2}",
        "isSignal": True,
        "isData": False,
        "noStat": True,
    }
