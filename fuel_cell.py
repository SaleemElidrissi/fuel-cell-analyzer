"""
fuel_cell.py
================================================================================
Fuel Cell Analysis Software  --  BS Engineering, Project 2
--------------------------------------------------------------------------------
A thermodynamic performance model for hydrogen- and hydrocarbon/alcohol-fuelled
fuel cells.

Given:
    * Fuel cell operating temperature      [degC]
    * Required electrical power            [W]
    * Required system (bus) voltage        [V]

the model returns:
    * Fuel cell (load) current             [A]
    * Fuel cell stack voltage              [V]
    * Overall fuel cell efficiency         [-]
    * Number of stacks required            [-]
    * Fuel consumption rate                [g/s] and [kg/h]

--------------------------------------------------------------------------------
MODEL SUMMARY
--------------------------------------------------------------------------------
1.  Reversible (open-circuit) cell voltage is obtained from the Gibbs free
    energy of the cell reaction:

        E_rev(T) = -dG(T) / (n * F)
        dG(T)    = dH - T*dS                (van 't Hoff / Gibbs-Helmholtz form)

    dH and dS are evaluated from standard enthalpies/entropies of formation and
    are switched between the LIQUID and GAS values of the fuel and of the
    product water according to the temperature relative to their boiling points.
    This phase switching is what produces the falling -- and eventually negative
    -- efficiency at high temperature that the assignment warns about.

2.  The overall efficiency is the product of three sub-efficiencies:

        eta = eta_thermo * eta_voltage * eta_util

        eta_thermo  = dG / dH               (maximum thermodynamic efficiency)
        eta_voltage = V_cell / E_rev        (voltage / polarisation efficiency)
        eta_util    = fuel utilisation      (fraction of fuel actually reacted)

3.  Electrical configuration:

        V_cell   = E_rev * eta_voltage      (operating voltage of ONE cell)
        N_series = round(V_required / V_cell)   cells in series  -> 1 stack
        V_stack  = N_series * V_cell  (~ V_required)
        P_stack  = N_series * (V_cell * I_cell)
        N_stacks = ceil(P_required / P_stack)   stacks in parallel
        I_load   = P_required / V_required      current delivered to the bus

4.  Fuel consumption follows directly from an energy balance (guaranteed
    consistent with the reported efficiency):

        n_dot_fuel = P_required / (eta * |dH(T)|)     [mol/s]
        m_dot_fuel = n_dot_fuel * M_fuel              [g/s]

--------------------------------------------------------------------------------
All thermodynamic data are standard-state (298.15 K, 1 atm) values taken from
NIST / standard thermodynamic tables.  Heat-capacity (Cp) corrections are
neglected; over 10-250 degC this is a small effect compared with the phase
change and keeps the model transparent, as is appropriate for a first-order
design tool.
================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
F = 96_485.0        # Faraday constant            [C/mol]
R = 8.314           # Universal gas constant      [J/(mol*K)]
T0 = 273.15         # 0 degC in kelvin            [K]


# ----------------------------------------------------------------------------
# Species thermodynamic data
#   Hf : standard enthalpy of formation   [J/mol]
#   S  : standard absolute entropy        [J/(mol*K)]
# ----------------------------------------------------------------------------
SPECIES = {
    #              Hf            S
    "H2_g":   (        0.0, 130.68),
    "O2_g":   (        0.0, 205.14),
    "H2O_l":  ( -285_830.0,  69.95),
    "H2O_g":  ( -241_820.0, 188.84),
    "CO2_g":  ( -393_520.0, 213.79),

    "CH3OH_l": (-238_400.0, 126.80),   # methanol (liquid)
    "CH3OH_g": (-201_000.0, 239.90),   # methanol (vapour)

    "C2H5OH_l": (-277_690.0, 160.70),  # ethanol (liquid)
    "C2H5OH_g": (-234_950.0, 281.60),  # ethanol (vapour)

    "CH4_g":   (-74_870.0, 186.25),    # methane (natural gas) -- gas at all T here
}


@dataclass
class Fuel:
    """Description of a fuel and its electrochemical oxidation reaction.

    The reaction is written per mole of fuel:

        fuel + (a) O2  ->  (b) CO2 + (c) H2O

    `n_electrons` is the number of electrons transferred per mole of fuel.
    """
    name: str
    formula: str
    M: float                      # molar mass                    [g/mol]
    n_electrons: int              # electrons per mole of fuel    [-]
    boiling_point_C: float        # normal boiling point          [degC]
    # stoichiometric coefficients
    o2: float                     # moles O2 per mole fuel
    co2: float                    # moles CO2 per mole fuel
    h2o: float                    # moles H2O per mole fuel
    # species keys for the two phases of the fuel
    fuel_liquid_key: str | None   # None if the fuel has no liquid phase in range
    fuel_gas_key: str
    recommended_cell: str = ""    # recommended fuel-cell technology (for the memo)

    # ------------------------------------------------------------------ #
    def _fuel_species_key(self, T_C: float) -> str:
        """Return the species key for the fuel at temperature `T_C`."""
        if self.fuel_liquid_key is not None and T_C < self.boiling_point_C:
            return self.fuel_liquid_key
        return self.fuel_gas_key

    @staticmethod
    def _water_key(T_C: float) -> str:
        """Product water is liquid below 100 degC, vapour above."""
        return "H2O_l" if T_C < 100.0 else "H2O_g"

    # ------------------------------------------------------------------ #
    def delta_H_S(self, T_C: float) -> tuple[float, float]:
        """Reaction enthalpy dH [J/mol] and entropy dS [J/(mol*K)] at `T_C`.

        Computed from standard formation values with the fuel and product
        water assigned to the correct phase for the given temperature.
        """
        fkey = self._fuel_species_key(T_C)
        wkey = self._water_key(T_C)

        Hf_fuel, S_fuel = SPECIES[fkey]
        Hf_w,    S_w    = SPECIES[wkey]
        Hf_o2,   S_o2   = SPECIES["O2_g"]
        Hf_co2,  S_co2  = SPECIES["CO2_g"]

        # Products - reactants
        dH = (self.co2 * Hf_co2 + self.h2o * Hf_w) \
             - (Hf_fuel + self.o2 * Hf_o2)
        dS = (self.co2 * S_co2 + self.h2o * S_w) \
             - (S_fuel + self.o2 * S_o2)
        return dH, dS

    def delta_G(self, T_C: float) -> float:
        """Gibbs free energy of reaction dG = dH - T*dS  [J/mol]."""
        dH, dS = self.delta_H_S(T_C)
        T_K = T_C + T0
        return dH - T_K * dS

    def reversible_voltage(self, T_C: float) -> float:
        """Reversible (Nernst, open-circuit) cell voltage E_rev [V]."""
        return -self.delta_G(T_C) / (self.n_electrons * F)

    def thermodynamic_efficiency(self, T_C: float) -> float:
        """Maximum thermodynamic efficiency eta_thermo = dG/dH [-]."""
        dH, _ = self.delta_H_S(T_C)
        return self.delta_G(T_C) / dH


# ----------------------------------------------------------------------------
# Fuel library  (the assignment requires hydrogen + one other; methanol,
# ethanol and natural gas are all provided so the team can choose / compare)
# ----------------------------------------------------------------------------
HYDROGEN = Fuel(
    name="Hydrogen", formula="H2", M=2.016, n_electrons=2,
    boiling_point_C=-252.9, o2=0.5, co2=0.0, h2o=1.0,
    fuel_liquid_key=None, fuel_gas_key="H2_g",
    recommended_cell="PEMFC (Proton Exchange Membrane Fuel Cell)",
)

METHANOL = Fuel(
    name="Methanol", formula="CH3OH", M=32.04, n_electrons=6,
    boiling_point_C=64.7, o2=1.5, co2=1.0, h2o=2.0,
    fuel_liquid_key="CH3OH_l", fuel_gas_key="CH3OH_g",
    recommended_cell="DMFC (Direct Methanol Fuel Cell)",
)

ETHANOL = Fuel(
    name="Ethanol", formula="C2H5OH", M=46.07, n_electrons=12,
    boiling_point_C=78.4, o2=3.0, co2=2.0, h2o=3.0,
    fuel_liquid_key="C2H5OH_l", fuel_gas_key="C2H5OH_g",
    recommended_cell="DEFC (Direct Ethanol Fuel Cell)",
)

NATURAL_GAS = Fuel(   # modelled as methane
    name="Natural Gas (CH4)", formula="CH4", M=16.04, n_electrons=8,
    boiling_point_C=-161.5, o2=2.0, co2=1.0, h2o=2.0,
    fuel_liquid_key=None, fuel_gas_key="CH4_g",
    recommended_cell="SOFC (Solid Oxide Fuel Cell)",
)

FUELS = {f.name: f for f in (HYDROGEN, METHANOL, ETHANOL, NATURAL_GAS)}


# ----------------------------------------------------------------------------
# System-level model
# ----------------------------------------------------------------------------
@dataclass
class FuelCellSystem:
    """A fuel-cell *system* built from series/parallel arrangements of cells.

    Design assumptions (typical first-order values; change here to retune):
        voltage_efficiency : operating cell voltage as a fraction of E_rev
        fuel_utilisation   : fraction of supplied fuel that reacts
        current_density    : nominal operating current density   [A/cm^2]
        cell_area          : active area of a single cell         [cm^2]
    """
    fuel: Fuel
    voltage_efficiency: float = 0.70
    fuel_utilisation: float = 0.95
    current_density: float = 0.6      # A/cm^2  (typical operating point)
    cell_area: float = 250.0          # cm^2

    # ------------------------------------------------------------------ #
    def analyze(self, T_C: float, power_W: float, voltage_V: float) -> "Result":
        """Run the full performance calculation at one operating point."""
        E_rev = self.fuel.reversible_voltage(T_C)
        eta_thermo = self.fuel.thermodynamic_efficiency(T_C)

        # Operating voltage of a single cell
        V_cell = E_rev * self.voltage_efficiency

        # Overall efficiency
        eta = eta_thermo * self.voltage_efficiency * self.fuel_utilisation

        # ---- Electrical configuration --------------------------------
        # Cells in series to reach the required bus voltage -> one stack
        if V_cell > 0:
            n_series = max(1, round(voltage_V / V_cell))
        else:
            n_series = float("inf")          # cell cannot produce voltage
        V_stack = n_series * V_cell

        # Single-cell operating point
        I_cell = self.current_density * self.cell_area     # A
        P_cell = V_cell * I_cell                           # W
        P_stack = n_series * P_cell                        # W per stack

        if P_stack > 0:
            n_stacks = math.ceil(power_W / P_stack)
        else:
            n_stacks = float("inf")

        # Current drawn by the load from the bus
        I_load = power_W / voltage_V

        # ---- Fuel consumption (energy balance) -----------------------
        dH, _ = self.fuel.delta_H_S(T_C)
        if eta > 0:
            # mol/s of fuel.  |dH| is the (phase-appropriate) heating value.
            n_dot = power_W / (eta * abs(dH))
            m_dot_g_s = n_dot * self.fuel.M          # g/s
        else:
            n_dot = float("nan")
            m_dot_g_s = float("nan")

        return Result(
            fuel=self.fuel.name, T_C=T_C, power_W=power_W, voltage_req_V=voltage_V,
            E_rev=E_rev, V_cell=V_cell, V_stack=V_stack,
            eta_thermo=eta_thermo, eta_voltage=self.voltage_efficiency,
            eta_util=self.fuel_utilisation, efficiency=eta,
            I_load=I_load, I_cell=I_cell,
            n_series=n_series, n_stacks=n_stacks,
            mol_per_s=n_dot, g_per_s=m_dot_g_s,
            phase=("liquid" if (self.fuel.fuel_liquid_key is not None
                                and T_C < self.fuel.boiling_point_C) else "gas"),
        )


@dataclass
class Result:
    """Container for the outputs of a single operating-point analysis."""
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
    mol_per_s: float
    g_per_s: float
    phase: str

    @property
    def kg_per_h(self) -> float:
        return self.g_per_s * 3.6   # g/s -> kg/h

    # -- pretty printer ------------------------------------------------- #
    def report(self) -> str:
        lines = [
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
            f"     thermo / voltage / util = "
            f"{self.eta_thermo*100:.1f}% / {self.eta_voltage*100:.0f}% / {self.eta_util*100:.0f}%",
            f"  Cells in series ..... {self.n_series}",
            f"  Stacks required ..... {self.n_stacks}",
            f"  Fuel consumption .... {self.g_per_s:8.4f} g/s  "
            f"({self.kg_per_h:.3f} kg/h)",
        ]
        return "\n".join(lines)
