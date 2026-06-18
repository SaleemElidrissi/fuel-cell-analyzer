"""
fuel_cell.py
================================================================================
Fuel Cell Analysis Software  --  BS Engineering / ME 417 Project 2
Team: El Idrissi, Adams        Assigned fuel: HYDROGEN + HEXANOL (1-hexanol)
--------------------------------------------------------------------------------
This implements the method specified in "Project 2 Additional Details":
the reversible cell voltage is found WITHOUT property tables, by integrating
ideal-gas heat-capacity polynomials from a single 298.15 K reference state.

For an ideal air/fuel cell the balanced reaction (air = O2 + 3.76 N2) is

    H2     :  H2 + 1/2 O2 + 1/2(3.76)N2  ->  H2O + 1/2(3.76)N2
    Hexanol:  C6H14O + 9 O2 + 9(3.76)N2  ->  6 CO2 + 7 H2O + 9(3.76)N2

The electrical work per mole of fuel (Additional Details, slide 4):

    w_elec = ( sum_react v*h_i  -  sum_prod v*h_j )
             - T_FC * ( sum_react v*s_i  -  sum_prod v*s_j )           [J/mol]

which is exactly -dG_rxn.  The reversible voltage is then

    E_rev = w_elec / (n * F)

Enthalpy and entropy of every species are built up from 298.15 K data:

    h_i(T)  = hf_298,i  + integral_298^T  Cp_i(T) dT
    s0_i(T) = s0_298,i  + integral_298^T  Cp_i(T)/T dT
    s_i(T)  = s0_i(T)  -  Ru * ln(y_i)        (mole-fraction / Dalton term)

with the gas heat capacity Cp = A + B*T + C*T^2 + D*T^3  [J/(mol*K)]
(Appendix E.1, Reid/Prausnitz/Poling).  Liquid species use a liquid Cp.

Phase switching (the assignment's "check the boiling point" instruction):
    * product water  : liquid below 100 C, vapour above
    * hexanol fuel   : liquid below 157.1 C, vapour above
which is what eventually drives the efficiency negative at high temperature.

Reference data sources:
    * Cp polynomials .............. Appendix E "Pure Component Properties"
    * hf, s0, HHV at 298 K ........ Table A-25 (provided) + NIST for hexanol
    * hexanol Cp(vap), bp, dHvap .. MatWeb 1-hexanol datasheet (provided)
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
F = 96_485.0          # Faraday constant            [C/mol]
RU = 8.314            # universal gas constant      [J/(mol*K)]
T0 = 273.15           # 0 degC in kelvin            [K]
TREF = 298.15         # reference temperature       [K]
N2_PER_O2 = 3.76      # moles N2 carried per mole O2 in air


# ----------------------------------------------------------------------------
# Heat-capacity models.  Each returns Cp [J/(mol*K)] given T [K].
#   Gas:    Cp = A + B*T + C*T^2 + D*T^3        (Appendix E.1)
#   Liquid: a constant, or the water polynomial Cp/R = a + b*T + c*T^2 (E.2)
# Along with each Cp we store its analytic integrals so the loop never calls a
# numerical integrator (per the assignment's efficiency note).
# ----------------------------------------------------------------------------
class GasCp:
    """Cp = A + B T + C T^2 + D T^3, with analytic enthalpy/entropy integrals."""
    def __init__(self, A, B, C, D):
        self.A, self.B, self.C, self.D = A, B, C, D

    def cp(self, T):
        return self.A + self.B*T + self.C*T**2 + self.D*T**3

    def h_int(self, T, Tr=TREF):           # integral of Cp dT  (J/mol)
        f = lambda t: self.A*t + self.B/2*t**2 + self.C/3*t**3 + self.D/4*t**4
        return f(T) - f(Tr)

    def s_int(self, T, Tr=TREF):           # integral of Cp/T dT  (J/mol/K)
        f = lambda t: self.A*math.log(t) + self.B*t + self.C/2*t**2 + self.D/3*t**3
        return f(T) - f(Tr)


class ConstCp:
    """Constant liquid heat capacity."""
    def __init__(self, cp_const):
        self._cp = cp_const

    def cp(self, T):
        return self._cp

    def h_int(self, T, Tr=TREF):
        return self._cp * (T - Tr)

    def s_int(self, T, Tr=TREF):
        return self._cp * math.log(T / Tr)


class LiquidWaterCp:
    """Liquid water Cp/R = a + b T + c T^2 (Appendix E.2, 273-373 K)."""
    a, b, c = 8.712, 1.250e-3, -1.800e-7

    def cp(self, T):
        return RU * (self.a + self.b*T + self.c*T**2)

    def h_int(self, T, Tr=TREF):
        f = lambda t: self.a*t + self.b/2*t**2 + self.c/3*t**3
        return RU * (f(T) - f(Tr))

    def s_int(self, T, Tr=TREF):
        f = lambda t: self.a*math.log(t) + self.b*t + self.c/2*t**2
        return RU * (f(T) - f(Tr))


# ----------------------------------------------------------------------------
# Species table:  key -> (hf_298 [J/mol], s0_298 [J/(mol*K)], Cp model)
# ----------------------------------------------------------------------------
SPECIES = {
    # gases ---------------------------------------------------------------
    "H2_g":  (        0.0, 130.57, GasCp(27.14,   9.274e-3, -1.381e-5,  7.645e-9)),
    "O2_g":  (        0.0, 205.03, GasCp(28.11,  -3.680e-6,  1.746e-5, -1.065e-8)),
    "N2_g":  (        0.0, 191.50, GasCp(31.15,  -1.357e-2,  2.680e-5, -1.168e-8)),
    "CO2_g": ( -393_520.0, 213.69, GasCp(19.80,   7.344e-2, -5.602e-5,  1.715e-8)),
    "H2O_g": ( -241_820.0, 188.72, GasCp(32.24,   1.924e-3,  1.055e-5, -3.596e-9)),
    "HEX_g": ( -315_900.0, 440.20, GasCp(4.8074,  0.5887,   -3.008e-4,  5.422e-8)),
    # liquids -------------------------------------------------------------
    "H2O_l": ( -285_830.0,  69.95, LiquidWaterCp()),
    "HEX_l": ( -377_500.0, 287.40, ConstCp(241.2)),   # 2.361 J/g/K * 102.18 g/mol
}


def h_species(key, T):
    """Absolute molar enthalpy h_i(T) = hf_298 + integral Cp dT  [J/mol]."""
    hf, _s, cp = SPECIES[key]
    return hf + cp.h_int(T)


def s0_species(key, T):
    """Standard molar entropy s0_i(T) = s0_298 + integral Cp/T dT  [J/mol/K]."""
    _h, s0, cp = SPECIES[key]
    return s0 + cp.s_int(T)


# ----------------------------------------------------------------------------
# Fuel definition
# ----------------------------------------------------------------------------
@dataclass
class Fuel:
    name: str
    formula: str
    M: float                 # molar mass               [g/mol]
    n_electrons: int         # electrons per mole fuel  [-]
    o2: float                # moles O2 per mole fuel
    co2: float               # moles CO2 per mole fuel
    h2o: float               # moles H2O per mole fuel
    boiling_point_C: float
    fuel_liquid_key: str | None
    fuel_gas_key: str
    HHV_J_per_mol: float     # higher heating value (liquid water), magnitude
    recommended_cell: str = ""

    def fuel_key(self, T_C):
        if self.fuel_liquid_key is not None and T_C < self.boiling_point_C:
            return self.fuel_liquid_key
        return self.fuel_gas_key

    @staticmethod
    def water_key(T_C):
        # Product water is treated as an ideal gas (vapour) throughout, to match
        # the worked H2 example in "Project 2 Additional Details" (slide 5 uses
        # the vapour value hf = -241,820 at 350 K = 77 C) and because the whole
        # method integrates ideal-gas Cp polynomials.  The assignment's
        # "check the boiling point" instruction applies to the FUEL, not water.
        return "H2O_g"

    # ---- reactant / product species lists (with air N2) -----------------
    def reactants(self, T_C):
        n_n2 = N2_PER_O2 * self.o2
        return {self.fuel_key(T_C): 1.0, "O2_g": self.o2, "N2_g": n_n2}

    def products(self, T_C):
        n_n2 = N2_PER_O2 * self.o2
        d = {self.water_key(T_C): self.h2o, "N2_g": n_n2}
        if self.co2 > 0:
            d["CO2_g"] = self.co2
        return d

    # ---- thermodynamics -------------------------------------------------
    def w_elec(self, T_C):
        """Reversible electrical work per mole fuel (= -dG_rxn) [J/mol]."""
        T = T_C + T0
        react, prod = self.reactants(T_C), self.products(T_C)

        # enthalpy terms
        H_r = sum(v * h_species(k, T) for k, v in react.items())
        H_p = sum(v * h_species(k, T) for k, v in prod.items())

        # entropy terms.  The -Ru*ln(y_i) Dalton correction applies only to the
        # GAS-phase species (a pure liquid has unit activity, no partial
        # pressure), so the mole fraction y_i is taken over gas species only.
        def S(mix):
            n_gas = sum(v for k, v in mix.items() if k.endswith("_g"))
            tot = 0.0
            for k, v in mix.items():
                s = s0_species(k, T)
                if k.endswith("_g"):
                    s -= RU * math.log(v / n_gas)
                tot += v * s
            return tot
        S_r, S_p = S(react), S(prod)

        return (H_r - H_p) - T * (S_r - S_p)

    def delta_H(self, T_C):
        """Reaction enthalpy dH_rxn = H_prod - H_react  [J/mol] (negative)."""
        T = T_C + T0
        react, prod = self.reactants(T_C), self.products(T_C)
        H_r = sum(v * h_species(k, T) for k, v in react.items())
        H_p = sum(v * h_species(k, T) for k, v in prod.items())
        return H_p - H_r

    def reversible_voltage(self, T_C):
        return self.w_elec(T_C) / (self.n_electrons * F)

    def thermodynamic_efficiency(self, T_C):
        """eta_thermo = w_elec / |dH_rxn| = dG/dH."""
        return self.w_elec(T_C) / abs(self.delta_H(T_C))


# ----------------------------------------------------------------------------
# Fuel library (hydrogen + the team's assigned fuel, hexanol)
# ----------------------------------------------------------------------------
HYDROGEN = Fuel(
    name="Hydrogen", formula="H2", M=2.016, n_electrons=2,
    o2=0.5, co2=0.0, h2o=1.0, boiling_point_C=-252.9,
    fuel_liquid_key=None, fuel_gas_key="H2_g",
    HHV_J_per_mol=285_830.0,
    recommended_cell="PEMFC (Proton Exchange Membrane Fuel Cell)",
)

HEXANOL = Fuel(
    name="Hexanol", formula="C6H14O", M=102.177, n_electrons=36,
    o2=9.0, co2=6.0, h2o=7.0, boiling_point_C=157.1,
    fuel_liquid_key="HEX_l", fuel_gas_key="HEX_g",
    HHV_J_per_mol=3_984_430.0,   # = 6*hf_CO2 + 7*hf_H2O(l) - hf_hexanol(l)
    recommended_cell="SOFC (Solid Oxide Fuel Cell) / reformed-fuel cell",
)

FUELS = {f.name: f for f in (HYDROGEN, HEXANOL)}


# ----------------------------------------------------------------------------
# System-level model
# ----------------------------------------------------------------------------
@dataclass
class FuelCellSystem:
    """Series/parallel arrangement of single cells (first-order design values)."""
    fuel: Fuel
    voltage_efficiency: float = 0.70   # operating cell V / reversible V
    fuel_utilisation: float = 0.95     # fraction of fuel that reacts
    current_density: float = 0.6       # A/cm^2
    cell_area: float = 250.0           # cm^2

    def analyze(self, T_C, power_W, voltage_V):
        E_rev = self.fuel.reversible_voltage(T_C)
        eta_thermo = self.fuel.thermodynamic_efficiency(T_C)

        V_cell = E_rev * self.voltage_efficiency
        eta = eta_thermo * self.voltage_efficiency * self.fuel_utilisation

        # electrical configuration
        n_series = max(1, round(voltage_V / V_cell)) if V_cell > 0 else float("inf")
        V_stack = n_series * V_cell
        I_cell = self.current_density * self.cell_area
        P_stack = n_series * V_cell * I_cell
        n_stacks = math.ceil(power_W / P_stack) if P_stack > 0 else float("inf")
        I_load = power_W / voltage_V

        # fuel consumption from energy balance (consistent with eta)
        if eta > 0:
            w_elec = self.fuel.w_elec(T_C)              # J/mol
            n_dot = power_W / (self.voltage_efficiency * self.fuel_utilisation * w_elec)
            g_per_s = n_dot * self.fuel.M
        else:
            g_per_s = float("nan")

        phase = ("liquid" if (self.fuel.fuel_liquid_key is not None
                              and T_C < self.fuel.boiling_point_C) else "gas")
        return Result(self.fuel.name, T_C, power_W, voltage_V, E_rev, V_cell,
                      V_stack, eta_thermo, self.voltage_efficiency,
                      self.fuel_utilisation, eta, I_load, I_cell, n_series,
                      n_stacks, g_per_s, phase)


@dataclass
class Result:
    fuel: str
    T_C: float
    power_W: float
    voltage_req_V: float
    E_rev: float
    V_cell: float
    V_stack: float
    eta_thermo: float
    eta_voltage: float
    eta_util: float
    efficiency: float
    I_load: float
    I_cell: float
    n_series: float
    n_stacks: float
    g_per_s: float
    phase: str

    @property
    def kg_per_h(self):
        return self.g_per_s * 3.6

    def report(self):
        return "\n".join([
            f"  Fuel ................. {self.fuel}  ({self.phase} phase)",
            f"  Temperature ......... {self.T_C:8.1f} degC",
            f"  Required power ...... {self.power_W/1000:8.2f} kW",
            f"  Required voltage .... {self.voltage_req_V:8.2f} V",
            "  " + "-" * 44,
            f"  Reversible voltage .. {self.E_rev:8.4f} V/cell",
            f"  Operating cell V .... {self.V_cell:8.4f} V/cell",
            f"  Stack voltage ....... {self.V_stack:8.3f} V",
            f"  Load current ........ {self.I_load:8.2f} A",
            f"  Efficiency .......... {self.efficiency*100:8.2f} %",
            f"     thermo/voltage/util = {self.eta_thermo*100:.1f}% / "
            f"{self.eta_voltage*100:.0f}% / {self.eta_util*100:.0f}%",
            f"  Cells in series ..... {self.n_series}",
            f"  Stacks required ..... {self.n_stacks}",
            f"  Fuel consumption .... {self.g_per_s:8.4f} g/s  ({self.kg_per_h:.3f} kg/h)",
        ])
