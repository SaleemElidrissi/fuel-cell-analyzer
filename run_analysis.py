"""
run_analysis.py
================================================================================
Demonstration driver for the BS Engineering Fuel Cell Analysis Software.

Produces, for a 50 kW / 12 V system over 10 degC - 250 degC:

    Graph 1 : Efficiency        vs temperature
    Graph 2 : Cell voltage      vs temperature
    Graph 3 : Fuel consumption  vs temperature

for BOTH hydrogen and a second fuel (default: methanol) -> six graphs total.
It also prints a fully worked single-temperature example for each fuel so the
numbers can be checked by hand.

Run:
    python run_analysis.py                 # hydrogen + methanol (default)
    python run_analysis.py Ethanol         # hydrogen + ethanol
    python run_analysis.py "Natural Gas (CH4)"
================================================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from fuel_cell import FuelCellSystem, FUELS, HYDROGEN

# --------------------------------------------------------------------------- #
# Demonstration specification (from the assignment)
# --------------------------------------------------------------------------- #
POWER_W = 50_000.0      # 50 kW
VOLTAGE_V = 12.0        # 12 V
T_MIN, T_MAX = 10.0, 250.0
N_POINTS = 241          # 1 degC resolution


def sweep(system: FuelCellSystem):
    """Return temperature array and per-temperature output arrays."""
    T = np.linspace(T_MIN, T_MAX, N_POINTS)
    eff, vcell, fuel_g_s = [], [], []
    for t in T:
        r = system.analyze(t, POWER_W, VOLTAGE_V)
        eff.append(r.efficiency * 100.0)        # %
        vcell.append(r.V_cell)                  # V/cell
        fuel_g_s.append(r.g_per_s)              # g/s
    return T, np.array(eff), np.array(vcell), np.array(fuel_g_s)


def mask_invalid(y):
    """Blank out non-physical points (efficiency <= 0) so curves read cleanly."""
    y = np.array(y, dtype=float)
    y[y <= 0] = np.nan
    return y


def plot_with_phase_breaks(ax, T, y, boundaries, **kw):
    """Plot y(T) but break the line at phase boundaries.

    A boiling point produces a genuine discontinuity in the thermodynamic
    properties (latent heat -> HHV/LHV step), so the curve should show a gap
    there rather than a vertical connecting segment.
    """
    y = mask_invalid(y).copy()
    edges = [b for b in boundaries if T[0] < b < T[-1]]
    for b in edges:
        # blank the single sample just past each boundary to split the line
        idx = int(np.searchsorted(T, b))
        if 0 < idx < len(y):
            y[idx] = np.nan
    ax.plot(T, y, **kw)


def make_plots(second_fuel_name: str):
    second = FUELS[second_fuel_name]
    fuels = [HYDROGEN, second]

    # one figure, 3 rows (the three required quantities) x 2 cols (the two fuels)
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(
        f"BS Engineering -- Fuel Cell Performance\n"
        f"50 kW system, 12 V bus, {T_MIN:.0f}-{T_MAX:.0f} degC",
        fontsize=14, fontweight="bold",
    )

    for col, fuel in enumerate(fuels):
        sysm = FuelCellSystem(fuel)
        T, eff, vcell, fuel_g_s = sweep(sysm)

        bp = fuel.boiling_point_C
        breaks = [bp, 100.0]   # fuel and water boiling points

        # --- Graph 1: efficiency ---
        ax = axes[0][col]
        plot_with_phase_breaks(ax, T, eff, breaks, color="tab:blue", lw=2)
        ax.set_title(f"{fuel.name}: Efficiency vs Temperature")
        ax.set_xlabel("Temperature [degC]")
        ax.set_ylabel("Efficiency [%]")
        ax.grid(True, alpha=0.3)

        # --- Graph 2: cell voltage ---
        ax = axes[1][col]
        plot_with_phase_breaks(ax, T, vcell, breaks, color="tab:green", lw=2)
        ax.set_title(f"{fuel.name}: Cell Voltage vs Temperature")
        ax.set_xlabel("Temperature [degC]")
        ax.set_ylabel("Operating cell voltage [V]")
        ax.grid(True, alpha=0.3)

        # --- Graph 3: fuel consumption ---
        ax = axes[2][col]
        plot_with_phase_breaks(ax, T, fuel_g_s, breaks, color="tab:red", lw=2)
        ax.set_title(f"{fuel.name}: Fuel Consumption vs Temperature")
        ax.set_xlabel("Temperature [degC]")
        ax.set_ylabel("Fuel consumption [g/s]")
        ax.grid(True, alpha=0.3)

        # mark boiling point / phase boundaries where they fall in range
        for ax in axes[:, col]:
            for boundary, label in [(bp, f"{fuel.formula} b.p."), (100.0, "H2O b.p.")]:
                if T_MIN < boundary < T_MAX:
                    ax.axvline(boundary, color="gray", ls="--", lw=1, alpha=0.7)
                    ax.text(boundary, ax.get_ylim()[1], f" {label}",
                            fontsize=7, color="gray", va="top", rotation=90)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"fuel_cell_graphs_H2_vs_{second.formula}.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved six graphs to:  {out}")
    return out


def worked_example(fuel, T_C: float):
    """Print a fully worked single-point example for hand-checking."""
    sysm = FuelCellSystem(fuel)
    r = sysm.analyze(T_C, POWER_W, VOLTAGE_V)
    dH, dS = fuel.delta_H_S(T_C)
    dG = fuel.delta_G(T_C)
    print("\n" + "=" * 56)
    print(f" WORKED EXAMPLE:  {fuel.name}  at  {T_C:.0f} degC")
    print("=" * 56)
    print(f"  Reaction: {fuel.formula} + {fuel.o2} O2 -> "
          f"{fuel.co2} CO2 + {fuel.h2o} H2O   (n = {fuel.n_electrons} e-)")
    print(f"  dH = {dH:12.1f} J/mol   ({r.phase}-phase fuel,"
          f" water {'liquid' if T_C < 100 else 'vapour'})")
    print(f"  dS = {dS:12.3f} J/mol-K")
    print(f"  dG = dH - T*dS = {dG:12.1f} J/mol   (T = {T_C+273.15:.2f} K)")
    print(f"  E_rev = -dG/(nF) = {r.E_rev:.4f} V")
    print("  " + "-" * 50)
    print(r.report())


def main():
    second_name = sys.argv[1] if len(sys.argv) > 1 else "Methanol"
    if second_name not in FUELS:
        print(f"Unknown fuel '{second_name}'. Choose from: {list(FUELS)}")
        sys.exit(1)

    # Worked examples: hydrogen at 80 degC (typical PEMFC), second fuel below b.p.
    worked_example(HYDROGEN, 80.0)
    second = FUELS[second_name]
    demo_T = 25.0 if (second.fuel_liquid_key is not None) else 200.0
    worked_example(second, demo_T)

    make_plots(second_name)


if __name__ == "__main__":
    main()
