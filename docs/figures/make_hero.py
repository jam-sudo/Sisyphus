"""Render the README hero figure: docs/figures/hero-{light,dark}.svg.

The concentration panel is a real engine run: caffeine 100 mg oral on the
reference body graph, 200 Monte Carlo draws over the Distribution-valued
physiology and drug parameters. Requires matplotlib, which is not a runtime
dependency; text is converted to paths, so the SVGs render identically
everywhere.

    python docs/figures/make_hero.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import sisyphus.engine.flux  # noqa: E402,F401  (registers flux functions)
from sisyphus.compounds import load_compound  # noqa: E402
from sisyphus.engine.compiler import ODECompiler  # noqa: E402
from sisyphus.engine.uncertainty import UncertaintyEngine  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
N_DRAWS = 200
T_END_H = 12.0

# Anatomical-atlas convention: arterial red, venous blue, portal violet, gut ochre.
THEMES = {
    "light": dict(ink="#1f2328", muted="#59636e", fill="#ffffff", arterial="#b8323c",
                  venous="#2c5ba0", portal="#6b4fa0", gut="#a07418"),
    "dark": dict(ink="#e6edf3", muted="#9198a1", fill="#0d1117", arterial="#f0727a",
                 venous="#7fa6e6", portal="#a991db", gut="#d9ae55"),
}

# Panel A layout in data units (x 0-10, y 0-7). Organs whose vascular and tissue
# spaces are separate (permeability-limited) are drawn as split nodes.
VEN, ART, LUNG = (1.2, 3.9), (8.8, 3.9), (5.0, 6.75)
BLOOD_W, ORGAN_W = 2.05, 1.6
SYSTEMIC = [("brain", 6.0, False), ("heart", 5.45, False), ("kidney", 4.9, False),
            ("muscle", 4.35, True), ("adipose", 3.8, True), ("skin", 3.25, True),
            ("bone", 2.7, True)]
LIVER, GUT, SPLEEN, PORTAL = (5.0, 1.95), (7.0, 1.35), (7.0, 2.05), (6.0, 1.65)
LUMEN = ["stomach", "duodenum", "jejunum", "ileum", "colon"]
LUMEN_Y = 0.35


def simulate() -> tuple[np.ndarray, np.ndarray]:
    graph = build_from_yaml(ROOT / "data/physiology/reference_man.yaml")
    drug = load_compound(ROOT / "data/compounds/caffeine.yaml")
    compiled = ODECompiler().compile(graph)
    mc = UncertaintyEngine().propagate(compiled, graph, drug, n_samples=N_DRAWS, seed=42,
                                       t_span=(0.0, T_END_H))
    t = np.linspace(0.0, T_END_H, 241)
    conc = np.array([np.interp(t, r.time_h, r.concentrations["venous_blood"])
                     for r in mc.sim_results])
    return t, np.percentile(conc, [5, 50, 95], axis=0)


def node(ax, xy, label, c, *, split=False, w=ORGAN_W, color=None):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.2), w, 0.4,
                                boxstyle="round,pad=0,rounding_size=0.2", lw=0.9,
                                ec=color or c["ink"], fc=c["fill"], zorder=3))
    if split:
        ax.plot([x + w / 2 - 0.3] * 2, [y - 0.2, y + 0.2], lw=0.7, color=c["muted"], zorder=4)
    ax.text(x - (0.15 if split else 0), y, label, ha="center", va="center", fontsize=9.5,
            color=c["ink"], zorder=4)


def edge(ax, a, b, color, *, ls="-", lw=1.0, rad=0.0, arrow=True):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", lw=lw, ls=ls,
                                 color=color, arrowstyle="-|>" if arrow else "-",
                                 mutation_scale=7, shrinkA=0, shrinkB=0, zorder=2))


def sink(ax, xy, c):
    ax.add_patch(plt.Circle(xy, 0.09, fc="none", ec=c["muted"], lw=0.8, zorder=3))


def draw_graph(ax, c):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.75, 7.1)
    ax.axis("off")
    node(ax, VEN, "venous blood", c, w=BLOOD_W, color=c["venous"])
    node(ax, ART, "arterial blood", c, w=BLOOD_W, color=c["arterial"])
    node(ax, LUNG, "lung", c)
    edge(ax, (VEN[0], VEN[1] + 0.2), (LUNG[0] - ORGAN_W / 2, LUNG[1]), c["venous"], rad=-0.25)
    edge(ax, (LUNG[0] + ORGAN_W / 2, LUNG[1]), (ART[0], ART[1] + 0.2), c["arterial"], rad=-0.25)
    for name, y, split in SYSTEMIC:
        node(ax, (5.0, y), name, c, split=split)
        edge(ax, (ART[0] - BLOOD_W / 2, ART[1]), (5.0 + ORGAN_W / 2, y), c["arterial"], lw=0.8)
        edge(ax, (5.0 - ORGAN_W / 2, y), (VEN[0] + BLOOD_W / 2, VEN[1]), c["venous"], lw=0.8)
    kidney_y = SYSTEMIC[2][1]
    edge(ax, (5.0 + ORGAN_W / 2, kidney_y + 0.08), (6.35, kidney_y + 0.32), c["muted"],
         ls=":", lw=0.9)
    sink(ax, (6.44, kidney_y + 0.35), c)
    # Splanchnic circulation: gut wall and spleen drain through the portal vein.
    node(ax, GUT, "gut wall", c)
    node(ax, SPLEEN, "spleen", c)
    node(ax, LIVER, "liver", c)
    ax.add_patch(plt.Circle(PORTAL, 0.11, fc=c["fill"], ec=c["portal"], lw=1.1, zorder=3))
    for src in (GUT, SPLEEN):
        edge(ax, (ART[0] - BLOOD_W / 2, ART[1] - 0.1), (src[0] + ORGAN_W / 2, src[1]),
             c["arterial"], lw=0.8)
        edge(ax, (src[0] - ORGAN_W / 2, src[1]), (PORTAL[0] + 0.11, PORTAL[1]), c["portal"], lw=0.9)
    edge(ax, (PORTAL[0] - 0.11, PORTAL[1]), (LIVER[0] + ORGAN_W / 2, LIVER[1] - 0.06),
         c["portal"], lw=0.9)
    edge(ax, (ART[0] - BLOOD_W / 2, ART[1] - 0.1), (LIVER[0] + 0.5, LIVER[1] + 0.2), c["arterial"],
         lw=0.8, rad=0.12)
    edge(ax, (LIVER[0] - ORGAN_W / 2, LIVER[1]), (VEN[0] + BLOOD_W / 2, VEN[1] - 0.05),
         c["venous"], lw=0.8)
    edge(ax, (LIVER[0] - 0.2, LIVER[1] - 0.2), (LIVER[0] - 0.55, LIVER[1] - 0.62), c["muted"],
         ls=":", lw=0.9)
    sink(ax, (LIVER[0] - 0.62, LIVER[1] - 0.7), c)
    edge(ax, (GUT[0] + 0.35, GUT[1] - 0.2), (GUT[0] + 0.85, GUT[1] - 0.58), c["muted"],
         ls=":", lw=0.9)
    sink(ax, (GUT[0] + 0.93, GUT[1] - 0.64), c)
    for i, (label, color, ls) in enumerate((("blood flow", c["arterial"], "-"),
                                             ("portal flow", c["portal"], "-"),
                                             ("absorption", c["gut"], (0, (3, 2))),
                                             ("clearance", c["muted"], ":"))):
        y = 2.55 - 0.42 * i
        edge(ax, (0.05, y), (0.6, y), color, ls=ls, lw=0.9)
        ax.text(0.75, y, label, va="center", fontsize=8.5, color=c["muted"])
    # GI lumen transit chain with absorption into the gut wall.
    xs = np.linspace(2.4, 8.6, len(LUMEN))
    for i, (x, name) in enumerate(zip(xs, LUMEN)):
        ax.add_patch(plt.Circle((x, LUMEN_Y), 0.1, fc=c["gut"], ec="none", zorder=3))
        ax.text(x, LUMEN_Y - 0.36, name, ha="center", va="center", fontsize=8.5, color=c["muted"])
        if i:
            edge(ax, (xs[i - 1] + 0.12, LUMEN_Y), (x - 0.12, LUMEN_Y), c["gut"], lw=0.9)
        if 1 <= i <= 3:
            edge(ax, (x, LUMEN_Y + 0.12), (GUT[0] - 0.25 + 0.25 * (i - 2), GUT[1] - 0.2),
                 c["gut"], ls=(0, (3, 2)), lw=0.8)


def draw_equation(fig, c, left):
    fig.text(left, 0.62, r"$\dfrac{d\mathbf{A}}{dt} \;=\; \sum_{e \,\in\, E}\; \phi_{\tau(e)}"
             r"\left(\mathbf{A};\, \theta_e\right)$", fontsize=19, color=c["ink"], va="center")
    fig.text(left, 0.43, "one flux term per typed edge", fontsize=10, style="italic",
             color=c["muted"], va="center")
    fig.text(left, 0.29, r"$\theta_e \sim \mathcal{D}(\mu,\ \mathrm{CV})$", fontsize=15,
             color=c["ink"], va="center")
    fig.text(left, 0.17, "every parameter is a distribution", fontsize=10, style="italic",
             color=c["muted"], va="center")


def arrow(fig, x0, x1, label, c):
    fig.patches.append(FancyArrowPatch((x0, 0.5), (x1, 0.5), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=10, lw=0.9,
                                       color=c["muted"]))
    fig.text((x0 + x1) / 2, 0.555, label, ha="center", fontsize=9.5, style="italic",
             color=c["muted"])


def draw_curve(ax, c, t, bands):
    lo, mid, hi = bands
    ax.fill_between(t, lo, hi, color=c["venous"], alpha=0.2, lw=0)
    ax.plot(t, mid, color=c["venous"], lw=1.8)
    i = int(mid.argmax())
    ax.plot(t[i], mid[i], "o", ms=4.5, color=c["ink"], zorder=4)
    ax.annotate(r"$C_{\max}$", (t[i], mid[i]), xytext=(12, 2), textcoords="offset points",
                fontsize=13, color=c["ink"], va="center")
    ax.set_xlim(0, T_END_H)
    ax.set_ylim(0, hi.max() * 1.18)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("time after dose (h)", fontsize=9.5, color=c["muted"])
    ax.set_ylabel("plasma concentration (mg/L)", fontsize=9.5, color=c["muted"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["muted"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=c["muted"], labelsize=9, width=0.8, length=3)
    ax.patch.set_alpha(0)
    ax.text(0.98, 0.95, f"caffeine 100 mg oral\n{N_DRAWS} Monte Carlo draws, 5–95%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, style="italic",
            color=c["muted"], linespacing=1.4)


def render(theme: str, t, bands) -> None:
    c = THEMES[theme]
    plt.rcParams.update({"font.family": ["Charter", "STIXGeneral"], "mathtext.fontset": "stix",
                         "svg.fonttype": "path", "svg.hashsalt": "sisyphus-hero"})
    fig = plt.figure(figsize=(10.5, 3.4))
    draw_graph(fig.add_axes((0.0, 0.0, 0.41, 1.0)), c)
    arrow(fig, 0.415, 0.46, "compile", c)
    draw_equation(fig, c, 0.47)
    arrow(fig, 0.69, 0.735, f"solve ×{N_DRAWS}", c)
    draw_curve(fig.add_axes((0.79, 0.17, 0.205, 0.74)), c, t, bands)
    fig.savefig(OUT / f"hero-{theme}.svg", transparent=True, metadata={"Date": None},
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    t, bands = simulate()
    for theme in THEMES:
        render(theme, t, bands)
