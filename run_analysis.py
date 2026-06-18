"""
run_analysis.py  --  ME 417 Project 2 demonstration driver
================================================================================
Generates, for a 10 kW / 12 V fuel-cell system over 15 C - 250 C:

    Graph 1 : efficiency        vs temperature
    Graph 2 : cell voltage      vs temperature
    Graph 3 : fuel consumption  vs temperature

for HYDROGEN and HEXANOL (the team's assigned fuel) -> six graphs total.
It also (a) writes the raw sweep data to CSV files for graphing, and
(b) prints fully worked single-temperature examples for hand-checking.
================================================================================
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

from fuel_cell import FuelCellSystem, FUELS, HYDROGEN, HEXANOL

POWER_W = 10_000.0        # 10 kW   (per Project 2 Additional Details)
VOLTAGE_V = 12.0          # 12 V
T_MIN, T_MAX = 15.0, 250.0
STEP = 1.0


def sweep(system):
    T = np.arange(T_MIN, T_MAX + STEP, STEP)
    eff, vcell, fuel_g_s = [], [], []
    for t in T:
        r = system.analyze(t, POWER_W, VOLTAGE_V)
        eff.append(r.efficiency * 100.0)
        vcell.append(r.V_cell)
        fuel_g_s.append(r.g_per_s)
    return T, np.array(eff), np.array(vcell), np.array(fuel_g_s)


def write_csv(fuel, T, eff, vcell, fuel_g_s):
    fn = f"results_{fuel.formula}.csv"
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_C", "efficiency_%", "cell_voltage_V", "fuel_g_per_s"])
        for row in zip(T, eff, vcell, fuel_g_s):
            w.writerow([f"{row[0]:.1f}",
                        f"{row[1]:.4f}" if row[1] > 0 else "nan",
                        f"{row[2]:.5f}" if row[2] > 0 else "nan",
                        f"{row[3]:.6f}" if row[3] > 0 else "nan"])
    return fn


def mask(y):
    y = np.array(y, dtype=float)
    y[y <= 0] = np.nan
    return y


def plot_breaks(ax, T, y, boundaries, **kw):
    y = mask(y).copy()
    for b in boundaries:
        if T[0] < b < T[-1]:
            idx = int(np.searchsorted(T, b))
            if 0 < idx < len(y):
                y[idx] = np.nan
    ax.plot(T, y, **kw)


COLORS = {"eff": "tab:blue", "volt": "tab:green", "fuel": "tab:red"}


def make_plots():
    fuels = [HYDROGEN, HEXANOL]
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"ME 417 Project 2 -- Fuel Cell Performance\n"
                 f"10 kW system, 12 V bus, {T_MIN:.0f}-{T_MAX:.0f} degC",
                 fontsize=14, fontweight="bold")

    for col, fuel in enumerate(fuels):
        T, eff, vcell, fuel_g_s = sweep(FuelCellSystem(fuel))
        write_csv(fuel, T, eff, vcell, fuel_g_s)
        breaks = [fuel.boiling_point_C]   # only the FUEL phase change matters

        plot_breaks(axes[0][col], T, eff, breaks, color=COLORS["eff"], lw=2)
        axes[0][col].set(title=f"{fuel.name}: Efficiency vs Temperature",
                         xlabel="Temperature [degC]", ylabel="Efficiency [%]")

        plot_breaks(axes[1][col], T, vcell, breaks, color=COLORS["volt"], lw=2)
        axes[1][col].set(title=f"{fuel.name}: Cell Voltage vs Temperature",
                         xlabel="Temperature [degC]", ylabel="Operating cell voltage [V]")

        plot_breaks(axes[2][col], T, fuel_g_s, breaks, color=COLORS["fuel"], lw=2)
        axes[2][col].set(title=f"{fuel.name}: Fuel Consumption vs Temperature",
                         xlabel="Temperature [degC]", ylabel="Fuel consumption [g/s]")

        for ax in axes[:, col]:
            ax.grid(True, alpha=0.3)
            bp = fuel.boiling_point_C
            if T_MIN < bp < T_MAX:
                ax.axvline(bp, color="gray", ls="--", lw=1, alpha=0.7)
                ax.text(bp, ax.get_ylim()[1], f" {fuel.formula} b.p.", fontsize=7,
                        color="gray", va="top", rotation=90)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "fuel_cell_graphs.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved six graphs to:  {out}")


def worked_example(fuel, T_C):
    r = FuelCellSystem(fuel).analyze(T_C, POWER_W, VOLTAGE_V)
    w = fuel.w_elec(T_C)
    dH = fuel.delta_H(T_C)
    print("\n" + "=" * 58)
    print(f" WORKED EXAMPLE:  {fuel.name}  at  {T_C:.0f} degC  ({T_C+273.15:.2f} K)")
    print("=" * 58)
    n2 = 3.76 * fuel.o2
    print(f"  Air reaction: {fuel.formula} + {fuel.o2:g} O2 + {n2:g} N2 -> "
          f"{fuel.co2:g} CO2 + {fuel.h2o:g} H2O + {n2:g} N2   (n={fuel.n_electrons} e-)")
    print(f"  w_elec = -dG = {w:12.1f} J/mol")
    print(f"  dH_rxn       = {dH:12.1f} J/mol")
    print(f"  E_rev = w_elec/(nF) = {r.E_rev:.4f} V")
    print("  " + "-" * 52)
    print(r.report())
    if r.eta_thermo > 1.0:
        print("  NOTE: thermodynamic efficiency dG/dH exceeds 100% because this"
              "\n        reaction increases entropy (liquid fuel -> many gas moles),"
              "\n        so the ideal cell can absorb heat from its surroundings."
              "\n        Overall efficiency (x voltage x utilisation) stays < 100%."
              "\n        Efficiency is on a lower-heating-value (vapour water) basis.")


def main():
    worked_example(HYDROGEN, 80.0)     # typical PEMFC point
    worked_example(HEXANOL, 25.0)      # liquid hexanol, room temperature
    make_plots()
    print("\nCSV data written:  results_H2.csv,  results_C6H14O.csv")


if __name__ == "__main__":
    main()
