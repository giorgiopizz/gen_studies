import os
import sys

import matplotlib as mpl
import numpy as np
import uproot
from gen_studies.analysis.utils import flatten_samples

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from gen_studies.plot.utils import cmap


def plot_simple(h, ax, label, color, **kwargs):
    edges = h.axes[0].edges
    content = h.values()
    ax.stairs(
        content,
        edges,
        label=label,
        color=color,
        **kwargs.get("stairs", {}),
    )


def plot_ratio_simple(h1, h2, ax, label, color, **kwargs):
    edges = h1.axes[0].edges
    c1 = h1.values()
    c2 = h2.values()
    ax.stairs(
        c1 / c2,
        edges,
        label=label,
        color=color,
        **kwargs.get("stairs", {}),
    )


def main():
    path = os.path.abspath(".")
    print("Working in analysis path:", path)
    sys.path.insert(0, path)

    exec("import config as analysis_cfg", globals(), globals())

    analysis_dict = analysis_cfg.__dict__  # type: ignore # noqa: F821
    samples = analysis_dict["samples"]

    get_regions = analysis_dict["get_regions"]
    get_variables = analysis_dict["get_variables"]
    systematics = analysis_dict["systematics"]

    lumi = analysis_dict["lumi"]

    variables = get_variables()
    regions = get_regions()

    os.makedirs("plots_variations", exist_ok=True)

    d = hep.style.CMS
    plt.style.use([d, hep.style.firamath])

    file = uproot.open("histos.root")

    flat_samples = flatten_samples(samples)
    for region in regions:
        for variable in variables:
            if "formatted" in variables[variable]:
                formatted = "$" + variables[variable]["formatted"] + "$"
            else:
                formatted = variable + ""
            _variable = variable.replace(":", "_")
            for sample_name in flat_samples:
                final_name = f"{region}/{_variable}/histo_{sample_name}"
                h_nominal = file[final_name].to_hist()
                colors = iter(cmap)
                variations = {}
                for syst in list(systematics.keys()) + ["stat"]:
                    if syst == "stat":
                        vvar_up = h_nominal.values() + np.sqrt(h_nominal.variances())
                        vvar_do = h_nominal.values() - np.sqrt(h_nominal.variances())
                    else:
                        if sample_name not in systematics[syst]["samples"]:
                            vvar_up = h_nominal.values().copy()
                            vvar_do = h_nominal.values().copy()
                        else:
                            syst_type = systematics[syst]["type"]
                            if syst_type == "shape":
                                _final_name = final_name + f"_{syst}Up"
                                vvar_up = file[_final_name].values()

                                _final_name = final_name + f"_{syst}Down"
                                vvar_do = file[_final_name].values()
                            elif syst_type == "lnN":
                                scaling = float(
                                    systematics[syst]["samples"][sample_name]
                                )
                                vvar_up = scaling * h_nominal.values()
                                vvar_do = 1.0 / scaling * h_nominal.values()

                    h_up = h_nominal.copy()
                    histo_view = h_up.view()
                    histo_view.value = vvar_up.copy()
                    histo_view.variance = np.zeros_like(histo_view.variance)

                    h_down = h_nominal.copy()
                    histo_view = h_down.view()
                    histo_view.value = vvar_do.copy()
                    histo_view.variance = np.zeros_like(histo_view.variance)

                    variations[syst] = {
                        "up": h_up,
                        "down": h_down,
                        "color": next(colors),
                    }

                fig, ax = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                    figsize=(10, 10),
                    dpi=100,
                )
                fig.tight_layout(pad=-0.5)
                hep.cms.label(
                    "Work in progress", data=False, ax=ax[0], exp="", lumi=str(lumi)
                )

                plot_simple(
                    h_nominal, ax[0], "Nominal", "black", **{"stairs": {"zorder": +10}}
                )

                for variation in variations:
                    plot_simple(
                        variations[variation]["up"],
                        ax[0],
                        f"{variation}",
                        variations[variation]["color"],
                        **dict(stairs=dict(linewidth=2)),
                    )

                    plot_simple(
                        variations[variation]["down"],
                        ax[0],
                        None,
                        variations[variation]["color"],
                        **dict(stairs=dict(linestyle="dashed", linewidth=2)),
                    )

                # ax[0].set_yscale("log")
                ax[0].set_ylabel("Events")
                ax[0].legend(
                    title=f"{region} {sample_name}",
                    fancybox=True,
                )


                edges = h_nominal.axes[0].edges
                ax[1].plot(edges, np.ones_like(edges), color="black")

                for variation in variations:
                    plot_ratio_simple(
                        variations[variation]["up"],
                        h_nominal,
                        ax[1],
                        variation,
                        variations[variation]["color"],
                        **dict(
                            stairs=dict(baseline=1.0, linewidth=2), plot_errors=True
                        ),
                    )

                    plot_ratio_simple(
                        variations[variation]["down"],
                        h_nominal,
                        ax[1],
                        None,
                        variations[variation]["color"],
                        **dict(
                            stairs=dict(baseline=1.0, linestyle="dashed", linewidth=2),
                        ),
                    )

                ax[1].legend()
                ax[1].set_ylabel("Variation / Nominal")

                ax[1].set_xlabel(formatted)
                # ax[1].set_ylim(0, 2)

                plt.savefig(
                    f"plots_variations/{region}_{_variable}_{sample_name}.png",
                    facecolor="white",
                    pad_inches=0.1,
                    bbox_inches="tight",
                )
                plt.close()


if __name__ == "__main__":
    main()
