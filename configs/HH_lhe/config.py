from typing import OrderedDict

import awkward as ak
import hist
import numpy as np

# General config
lumi = 300.0  # fb^-1

runner = dict(
    local=True,
    max_workers=6,
)

samples = {}

samples["HHjj"] = dict(
    xs=1.090e-02,  # need to change
    files_pattern="/gwteras/cms/store/user/gpizzati/PrivateMC/triennali/HHjj_smhloop0_dim6_12ops_CPV_mixed_new/HHJJ_SMHLOOP0_DIM6_12OPS_CPV_MIXED_NEW_dim6_cpodd/RunIISummer20UL18NanoAODv9_106X_upgrade2018_realistic_v11_nanoGEN_NANOAODSIM/240604_140232/0000/nanoAOD_*",
    limit_files=10,
    nevents_per_file=1000,
    nevents_per_job=1000,  # need to change ?
    eft=dict(
        reweight_card="/gwpool/users/tecedor/prova/genproductions/bin/MadGraph5_aMCatNLO/cards/folder_HHjj_smhloop0_dim6_12ops_CPV_mixed/HHjj_smhloop0_dim6_12ops_CPV_mixed_reweight_card.dat",
        # ops=[
        #     "cH",
        #     "cHW",
        #     "cHWB",
        #     "cHDD",
        #     "cHbox",
        #     "cHQ3",
        #     "cHWtil",
        #     "cHWBtil",
        #     "cHBtil",
        #     "cbBIm",
        #     "cbHIm",
        #     "cbWIm",
        # ],
        ops=["cHbox", "cHWtil", "cHWBtil", "cHBtil"],
    ),
)

samples["TTbar"] = dict(
    xs=1.090e-02,  # need to change
    files_pattern=(
        "/gwteras/cms/store/user/gpizzati/PrivateMC"
        "/triennali/HHjj_smhloop0_dim6_12ops_CPV_mixed/"
        "HHJJ_SMHLOOP0_DIM6_12OPS_CPV_MIXED_dim6_cpodd/"
        "RunIISummer20UL18NanoAODv9_106X_upgrade2018_realistic_v11_nanoGEN_NANOAODSIM"
        "/240529_154449/0000/*root"
    ),
    limit_files=2,
    nevents_per_file=5000,
    nevents_per_job=50000,  # need to change ?
    eft=dict(),
)

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
    # events["Lepton"] = events.Particle[
    #  (abs(events.Particle.pdgId) == 11) | (abs(events.Particle.pdgId) == 13)
    # ]
    # Remove events where nleptons != 2
    # events = events[ak.num(events.Lepton) == 2]

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
        "mhh": {
            "func": lambda events: (events.Higgs[:, 0] + events.Higgs[:, 1]).mass,
            "axis": hist.axis.Regular(50, 250, 2000, name="mhh"),
            "formatted": r"m_{hh} \; [GeV]",
        },
        "mjj:ptj1": {
            "func1": lambda events: (events.Jet[:, 0] + events.Jet[:, 1]).mass,
            "axis1": hist.axis.Regular(10, 200, 3000, name="mjj"),
            "func2": lambda events: events.Jet[:, 0].pt,
            "axis2": hist.axis.Regular(6, 30, 150, name="ptj1"),
            "formatted": "m_{jj}:p^T_{j1}",
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
        "dphijj": {
            "func": lambda events: abs(events.Jet[:, 0].deltaphi(events.Jet[:, 1])),
            "axis": hist.axis.Regular(50, 0, np.pi, name="dphijj"),
            "formatted": r"\Delta\phi_{jj}",
        },
        "detahh": {
            "func": lambda events: abs(events.Higgs[:, 0].deltaeta(events.Higgs[:, 1])),
            "axis": hist.axis.Regular(25, 1, 8, name="detahh"),
            "formatted": r"\Delta\eta_{hh}",
        },
        "dphihh": {
            "func": lambda events: abs(events.Higgs[:, 0].deltaphi(events.Higgs[:, 1])),
            "axis": hist.axis.Regular(50, 0, np.pi, name="dphihh"),
            "formatted": r"\Delta\phi_{hh}",
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
        # "ptl1": {
        #    "func": lambda events: events.Lepton[:, 0].pt,
        #    "axis": hist.axis.Regular(30, 25, 150, name="ptl1"),
        #    "formatted": "p^T_{l1} \; [GeV]",
        # },
        # "ptl2": {
        #    "func": lambda events: events.Lepton[:, 1].pt,
        #    "axis": hist.axis.Regular(30, 20, 150, name="ptl2"),
        #    "formatted": "p^T_{l2} \; [GeV]",
        # },
        "pth1": {
            "func": lambda events: events.Higgs[:, 0].pt,
            "axis": hist.axis.Regular(30, 25, 150, name="pth1"),
            "formatted": r"p^T_{h1} \; [GeV]",
        },
        "pth2": {
            "func": lambda events: events.Higgs[:, 1].pt,
            "axis": hist.axis.Regular(30, 20, 150, name="pth2"),
            "formatted": r"p^T_{h2} \; [GeV]",
        },
        "pthh": {
            "func": lambda events: (events.Higgs[:, 0] + events.Higgs[:, 1]).pt,
            "axis": hist.axis.Regular(30, 20, 2000, name="pthh"),
            "formatted": r"p^T_{hh} \; [GeV]",
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
        # "etal1": {
        #  "func": lambda events: events.Lepton[:, 0].eta,
        #  "axis": hist.axis.Regular(30, 0, 2.5, name="etal1"),
        #    "formatted": "\eta_{l1}",
        # },
        # "etal2": {
        #  "func": lambda events: events.Lepton[:, 1].eta,
        #  "axis": hist.axis.Regular(30, 0, 2.5, name="etal2"),
        #   "formatted": "\eta_{l2}",
        "etah1": {
            "func": lambda events: events.Higgs[:, 0].eta,
            "axis": hist.axis.Regular(30, 0, 2.5, name="etah1"),
            "formatted": r"\eta_{h1}",
        },
        "etah2": {
            "func": lambda events: events.Higgs[:, 1].eta,
            "axis": hist.axis.Regular(30, 0, 2.5, name="etah2"),
            "formatted": r"\eta_{h2}",
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
        "phih1": {
            "func": lambda events: events.Higgs[:, 0].phi,
            "axis": hist.axis.Regular(30, 0, np.pi, name="phih1"),
            "formatted": r"\phi_{h1}",
        },
        "phih2": {
            "func": lambda events: events.Higgs[:, 1].phi,
            "axis": hist.axis.Regular(30, 0, np.pi, name="phih2"),
            "formatted": r"\phi_{h2}",
        },
        "events": {
            "func": lambda events: ak.ones_like(events.genWeight),
            "axis": hist.axis.Regular(1, 0, 2, name="events"),
        },
    }


def get_variations():
    variations = {
        "nominal": {
            "switches": [],
            "func": lambda events: events,
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
        }
        variation_idx += 1

    return variations


systematics = {
    "QCDScale": {
        # "name": "PDF",
        "kind": "weight_envelope",
        # "type": "shape",
        # "AsLnN": "0",
        "samples": {sample: [f"QCDScale_{i}" for i in range(6)] for sample in samples},
    },
    "PDF": {
        # "name": "PDF",
        "kind": "weight_square",
        # "type": "shape",
        # "AsLnN": "0",
        "samples": {sample: [f"PDF_{i}" for i in range(100)] for sample in samples},
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


# Plot config
scales = ["lin", "log"][:1]


def get_plot(op):
    cmap = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
    colors = iter(cmap)
    plot = OrderedDict()
    plot["TTbar_sm"] = {
        "color": next(colors),
        "name": "TTbar",
        # 'isSignal': False
    }

    plot["HHjj_sm"] = {
        "color": next(colors),
        "name": "HHjj",
        # 'isSignal': True
    }

    plot[f"HHjj_lin_{op}"] = {
        "color": next(colors),
        "name": f"Lin {op}",
        "isSignal": True,
        "superimposed": True,
    }

    plot[f"HHjj_quad_{op}"] = {
        "color": next(colors),
        "name": f"Quad {op}",
        "isSignal": True,
        "superimposed": True,
    }
    return plot
