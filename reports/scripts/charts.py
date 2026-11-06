"""Result charts for the report (vector PDF, column width), from the runs' decode_lab.json and per_class.json.
Palette and marks follow the dataviz skill's reference instance: categorical slots 1-3 (validated all-pairs), solid
hairline grid, a legend for every multi-series chart, text in ink colours, never in series colours.
  uv run python reports/scripts/charts.py"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RUNS, FIGS = Path('workspace/mtdet/runs'), Path('reports/report-overleaf/figs')
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.5, 'pdf.fonttype': 42})


def style(ax):
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(width=0.4, length=2)


def per_class():
    d = json.loads(Path('reports/scripts/per_class.json').read_text())
    names = list(d['nd'][0]['per_class_AP'])
    mean = {s: {c: 100 * st.mean(r['per_class_AP'][c] for r in d[s]) for c in names} for s in ('fit', 'nd', 'lf')}
    order = sorted(names, key=lambda c: mean['nd'][c])
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    style(ax)
    for i, c in enumerate(order):
        lo, hi = min(mean[s][c] for s in mean), max(mean[s][c] for s in mean)
        ax.plot([lo, hi], [i, i], color=GRID, lw=1.6, zorder=1, solid_capstyle='round')
    for s, color, label, marker in (('fit', BLUE, 'FIT (training images)', 'o'),
                                    ('nd', ORANGE, 'ND (val-like holdout)', 'D'),
                                    ('lf', AQUA, 'LF (leak-free)', 's')):
        ax.scatter([mean[s][c] for c in order], range(len(order)), s=16, color=color, marker=marker, label=label,
                   zorder=3, edgecolors='white', linewidths=0.6)
    for i, c in enumerate(order):  # direct label: the ND value only
        ax.annotate(f"{mean['nd'][c]:.0f}", (mean['nd'][c], i), xytext=(0, 4.5), textcoords='offset points',
                    ha='center', fontsize=5.5, color=INK2)
    ax.set_yticks(range(len(order)), [c.replace('_', ' ') for c in order])
    ax.set_xlim(-2, 102)
    ax.set_xlabel('AP (%), IoU 0.50:0.95, mean of 3 seeds')
    ax.legend(loc='lower center', bbox_to_anchor=(0.42, 1.0), ncol=3, frameon=False, handletextpad=0.1,
              columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / 'per_class.pdf')


def lab(run, key):
    return json.loads((RUNS / run / 'decode_lab.json').read_text())[key]['delta_m']


def epochs(key='base'):
    series = {  # (epochs, runs): plain = flips only; SAR 0.5 / 0.8 = SARAugment probability
        'plain': [(100, ['Y_s0']), (150, ['Y150_s0']), (250, ['Y250_s0', 'Y250_s1', 'Y250_s2'])],
        'SAR, p = 0.5': [(150, ['YSA150_s0']), (250, ['YSA250_s0', 'YSA250_s1', 'YSA250_s2']),
                         (350, ['YSA350_s0', 'YSA350_s1', 'YSA350_s2']), (450, ['YSA450_s0', 'YSA450_s1', 'YSA450_s2'])],
        'SAR, p = 0.8': [(250, ['YSP250_s0', 'YSP250_s1', 'YSP250_s2']), (350, ['YSP350_s0', 'YSP350_s1', 'YSP350_s2'])],
    }
    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    style(ax)
    out = {}
    for (name, points), color, marker in zip(series.items(), (BLUE, ORANGE, AQUA), ('o', 'D', 's')):
        xs, ms, sds = [], [], []
        for e, runs in points:
            v = [lab(r, key) for r in runs]
            xs.append(e), ms.append(st.mean(v)), sds.append(st.stdev(v) if len(v) > 1 else 0)
            out.setdefault(name, []).append((e, round(st.mean(v), 2), round(sds[-1], 2), len(v)))
        ax.plot(xs, ms, color=color, lw=1.0, zorder=2)
        ax.errorbar(xs, ms, yerr=sds, fmt='none', ecolor=color, elinewidth=0.7, capsize=1.6, zorder=2)
        for x, m, (e, runs) in zip(xs, ms, points):
            filled = len(runs) > 1
            ax.scatter([x], [m], s=16, marker=marker, zorder=3, linewidths=0.8,
                       color=color if filled else 'white', edgecolors=color if not filled else 'white')
        ax.plot([], [], color=color, marker=marker, ms=3.5, lw=1.0, label=name)
    ax.set_xticks([100, 150, 250, 350, 450])
    ax.set_xlabel('training epochs (holdout runs, local train only)')
    ax.set_ylabel('ND $\\Delta_m$ (%)')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor=GRID, handletextpad=0.3)
    ax.text(0.02, 0.97, 'filled: mean $\\pm$ std of 3 seeds; hollow: 1 seed', transform=ax.transAxes, fontsize=5.5,
            va='top', color=INK2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / f'epochs_{key}.pdf')
    return out


if __name__ == '__main__':
    per_class()
    print(json.dumps(epochs('fuse_b1_jd0.5')))  # the report's Fig. 2: ND under the final decoding
