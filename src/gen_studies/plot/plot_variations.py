import os
import sys

import matplotlib as mpl
import numpy as np
import uproot

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

    # region = "sr"
    # variable = "etah1"
    # sample = "HHjj_sm"

    component = "sm"
    for region in regions:
        for variable in variables:
            if "formatted" in variables[variable]:
                formatted = "$" + variables[variable]["formatted"] + "$"
            else:
                formatted = variable + ""
            _variable = variable.replace(":", "_")
            for sample in samples:
                final_name = f"{region}/{_variable}/histo_{sample}_{component}"
                h_nominal = file[final_name].to_hist()
                colors = iter(cmap)
                variations = {}
                for variation in systematics:
                    # variation = "QCDScale"
                    h_up = file[final_name + f"_{variation}Up"].to_hist()
                    h_down = file[final_name + f"_{variation}Down"].to_hist()
                    variations[variation] = {
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

                # ax[0].set_title()

                plot_simple(
                    h_nominal,
                    ax[0],
                    "Nominal",
                    "black",
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

                edges = h_nominal.axes[0].edges
                content = h_nominal.values()
                err = np.sqrt(h_nominal.variances())
                ax[0].stairs(
                    content + err,
                    edges,
                    baseline=content - err,
                    fill=True,
                    zorder=-10,
                    hatch="///",
                    color="darkgrey",
                    alpha=1,
                    facecolor="none",
                    linewidth=0.0,
                )
                ax[0].set_yscale("log")
                ax[0].set_ylabel("Events")
                ax[0].legend(
                    title=f"{region} {sample}",
                    fancybox=True,
                )

                # for tag in ["Up", "Down"]:
                #     _final_name = final_name + f"_{variation}{tag}"

                ax[1].plot(edges, np.ones_like(edges), color="black")
                ax[1].stairs(
                    (content + err) / content,
                    edges,
                    baseline=(content - err) / content,
                    fill=True,
                    color="lightgrey",
                )

                for variation in variations:
                    plot_ratio_simple(
                        variations[variation]["up"],
                        h_nominal,
                        ax[1],
                        f"{variation}/Nominal",
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
                ax[1].set_ylabel("Component / SM")

                ax[1].set_xlabel(formatted)
                ax[1].set_ylim(0, 2)

                plt.savefig(
                    f"plots_variations/{region}_{_variable}_{sample}.png",
                    facecolor="white",
                    pad_inches=0.1,
                    bbox_inches="tight",
                )
                plt.close()


if __name__ == "__main__":
    main()
