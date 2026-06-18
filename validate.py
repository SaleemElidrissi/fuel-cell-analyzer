"""
validate.py  --  cross-check the model against the professor's worked numbers
================================================================================
"Project 2 Additional Details" works an H2/air fuel cell at 350 K and lists the
sensible enthalpies (h_T2 - h_298) read from the ideal-gas tables:

    H2 :  9971 - 8468 = 1503 kJ/kmol
    O2 : 10213 - 8682 = 1531 kJ/kmol
    H2O: 11652 - 9904 = 1748 kJ/kmol   (vapour)

This script recomputes those same quantities from the Cp polynomials used in the
code and reports the difference, then prints the full H2 result at 350 K.
================================================================================
"""

from fuel_cell import SPECIES, h_species, HYDROGEN, HEXANOL, FuelCellSystem, F

TABLE = {            # professor's slide-5 sensible enthalpies at 350 K [J/mol]
    "H2_g": 1503.0,
    "O2_g": 1531.0,
    "H2O_g": 1748.0,
}
T_K = 350.0

print("Sensible enthalpy  h(350 K) - h(298.15 K)   [J/mol]")
print(f"{'species':8} {'polynomial':>12} {'prof. table':>12} {'diff %':>8}")
for key, ref in TABLE.items():
    _hf, _s, cp = SPECIES[key]
    poly = cp.h_int(T_K)
    print(f"{key:8} {poly:12.1f} {ref:12.1f} {100*(poly-ref)/ref:8.2f}")

print("\nReaction enthalpy of H2/air at 350 K (vapour water):")
dH = HYDROGEN.delta_H(350.0 - 273.15)
print(f"  dH_rxn = {dH:,.1f} J/mol   (textbook LHV approx -241,800 J/mol)")

print("\nFull H2 result at 350 K (77 C), 25 kW / 12 V (the example's power):")
r = FuelCellSystem(HYDROGEN).analyze(350.0 - 273.15, 25_000.0, 12.0)
print(f"  E_rev   = {r.E_rev:.4f} V       (ideal H2/air ~1.18-1.20 V)")
print(f"  I_load  = {r.I_load:.1f} A")
print(f"  fuel    = {r.g_per_s:.4f} g/s H2")

print("\nReversible voltage spot-checks at 298.15 K (25 C):")
for fuel in (HYDROGEN, HEXANOL):
    print(f"  {fuel.name:9} E_rev = {fuel.reversible_voltage(25.0):.4f} V,  "
          f"eta_thermo = {fuel.thermodynamic_efficiency(25.0)*100:.1f} %")
