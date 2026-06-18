# Fuel Cell Analysis Software — BS Engineering, Project 2

Thermodynamic performance model for hydrogen- and hydrocarbon/alcohol-fuelled
fuel cells.

## Files
| File | Purpose |
|------|---------|
| `fuel_cell.py`   | Core model: thermodynamic data, `Fuel`, `FuelCellSystem`, `Result`. |
| `run_analysis.py`| Demonstration driver: prints worked examples and saves the six graphs. |
| `requirements.txt` | Python dependencies (numpy, matplotlib). |

## Install & run
```bash
pip install -r requirements.txt

python run_analysis.py                 # Hydrogen + Methanol (default)
python run_analysis.py Ethanol         # Hydrogen + Ethanol
python run_analysis.py "Natural Gas (CH4)"
```
Each run prints two fully worked single-temperature examples (for hand-checking)
and saves a 3×2 grid of graphs (`fuel_cell_graphs_H2_vs_*.png`):

* Row 1 — efficiency vs temperature
* Row 2 — cell voltage vs temperature
* Row 3 — fuel consumption vs temperature

for hydrogen (left) and the chosen second fuel (right) → **six graphs**, for a
50 kW / 12 V system over 10–250 °C.

## Using the model directly
```python
from fuel_cell import FuelCellSystem, HYDROGEN

system = FuelCellSystem(HYDROGEN)
r = system.analyze(T_C=80, power_W=50_000, voltage_V=12)
print(r.report())
# r.efficiency, r.I_load, r.V_stack, r.n_stacks, r.g_per_s, r.kg_per_h ...
```

### Inputs
- Fuel cell temperature `T_C` [°C]
- Required power `power_W` [W]
- Required voltage `voltage_V` [V]

### Outputs (`Result`)
- `I_load` — fuel cell / load current [A]
- `V_stack` — fuel cell stack voltage [V]
- `efficiency` — overall fuel cell efficiency [–]
- `n_stacks` — number of stacks required
- `g_per_s` / `kg_per_h` — fuel consumption rate

## Model (one paragraph)
The reversible cell voltage is `E_rev = -ΔG/(nF)` with `ΔG = ΔH − TΔS`, where
ΔH and ΔS come from standard formation data and are **switched between liquid
and gas** for the fuel and the product water depending on temperature relative
to each boiling point. Overall efficiency is
`η = η_thermo · η_voltage · η_util` with `η_thermo = ΔG/ΔH`. The number of
series cells meets the bus voltage, parallel stacks meet the power, and fuel
flow follows from an energy balance `ṅ = P/(η·|ΔH|)`. Full derivation and all
assumptions are documented in the docstring at the top of `fuel_cell.py`.

The visible steps in the curves at the boiling points are real
(latent-heat / HHV↔LHV transition), not numerical artefacts.
