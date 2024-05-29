import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

d = hep.style.CMS
plt.style.use([d, hep.style.firamath])

cmap = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
sm_color = cmap[0]
lin_color = cmap[1]
quad_color = cmap[2]


def plot_good(h, ax, label, color, **kwargs):
    centers = h.axes[0].centers()
    edges = h.axes[0].edges()
    content = h.values()
    sumw2 = np.square(h.errors())

    ax.errorbar(
        centers,
        content,
        yerr=np.sqrt(sumw2),
        fmt="o",
        label=label,
        color=color,
        **kwargs.get("errorbar", {}),
    )
    ax.stairs(
        content,
        edges,
        # label=label,
        color=color,
        **kwargs.get("stairs", {}),
    )


def plot_ratio(h1, h2, ax, label, color, **kwargs):
    centers = h1.axes[0].centers()
    edges = h1.axes[0].edges()
    cont1 = h1.values()
    cont2 = h2.values()
    ratio = cont1 / cont2
    err1 = h1.errors()
    err2 = h2.errors()

    err = np.square(ratio) * (np.power(err1 / cont1, 2) + np.power(err2 / cont2, 2))

    if kwargs.get("plot_errors", True):
        ax.errorbar(
            centers,
            ratio,
            yerr=err,
            fmt="o",
            label=label,
            color=color,
            **kwargs.get("errorbar", {}),
        )

    ax.stairs(
        ratio,
        edges,
        # label=label,
        color=color,
        **kwargs.get("stairs", {}),
    )


def format_variable_name(variable):
    variable = variable + ""
    formatted = "$" + variable + "$"
    formatted = formatted.replace("eta", "\eta")
    formatted = formatted.replace("phi", "\phi")
    formatted = formatted.replace("pt", "p^T")

    if variable.startswith("d"):
        formatted = formatted.replace("d", "\Delta", 1)
    formatted = formatted.replace(formatted[-3:-1], "_{" + formatted[-3:-1] + "}")

    return formatted


def final_plot(input_file, variable, op, scale, formatted):
    # Begin single plot
    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 10),
        dpi=100,
    )  # dpi=100
    fig.tight_layout(pad=-0.5)

    hep.cms.label("Work in progress", data=False, ax=ax[0], exp="", lumi="100")

    h_sm = input_file[variable]["histo_sm"]
    h_lin = input_file[variable][f"histo_lin_{op}"]
    h_quad = input_file[variable][f"histo_quad_{op}"]

    ax[0].plot(h_sm.axes[0].edges(), np.zeros_like(h_sm.axes[0].edges()), color="black")
    plot_good(h_sm, ax[0], "SM", sm_color)
    plot_good(h_lin, ax[0], f"Lin {op}", lin_color)
    plot_good(h_quad, ax[0], f"Quad {op}", quad_color)
    ax[0].legend()
    ax[0].set_ylabel("Events")
    if scale == "log":
        ax[0].set_yscale(scale)

    # Lower plot
    edges = h_sm.axes[0].edges()
    ax[1].plot(edges, np.zeros_like(edges), color="black")

    plot_ratio(
        input_file[variable][f"histo_lin_{op}"],
        input_file[variable]["histo_sm"],
        ax[1],
        "Lin/SM",
        lin_color,
        **dict(stairs=dict(baseline=0.0), plot_errors=True),
    )

    plot_ratio(
        input_file[variable][f"histo_quad_{op}"],
        input_file[variable]["histo_sm"],
        ax[1],
        "Quad/SM",
        quad_color,
        **dict(stairs=dict(baseline=0.0)),
    )
    ax[1].legend()

    min_val_ratio = -0.1
    max_val_ratio = 0.5
    ax[1].set_xlim(edges[0], edges[-1])
    ax[1].set_ylim(min_val_ratio, max_val_ratio)

    ax[1].set_ylabel("Component / SM")
    ax[1].set_xlabel(formatted)

    fig.savefig(
        f"plots/{scale}_{variable}_{op}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
