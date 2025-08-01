# Thermocline-identification

This repository provides a unified toolkit for detecting ocean thermoclines using **six different methods**. All methods are designed to work with `.csv` profile data containing depth-temperature (and optionally salinity) values.

---

##  Supported Methods

| Method Name                               | Function/Class Name                 | Description |
|-------------------------------------------|-------------------------------------|-------------|
| Entropy Value Method                      | `calculate_entropy()`               | Calculates entropy-based weights from multiple physical features. |
| Tensor Analysis Method                    | `ThermoclineDetector` class         | using tensor slicing + SVD to detect structural changes. |
| Inflection Point Method                   | `calculate_inflection_point_method()` | Identifies thermocline by detecting abrupt curvature shifts in the temperature profile. |
| Variable Representative Isotherm (VRI)    | `calculate_vri_thermocline()`      | Uses surface and 400m temperatures to infer thermocline parameters. |
| Sigmoid Function Fitting Method           | `thermocline_sigmoid()`            | Fits sigmoid curve to temperature profile and derives thermocline bounds. |
| Hyperbolic Tangent (Tanh) Fitting Method  | `thermocline_tanh()`               | Fits tanh curve to the profile and extracts thermocline depth, thickness, and gradient. |

---

##  Input Data Format

All methods work with `.csv` files containing **vertical profile data**.

- Required columns: `Latitude`, `Longitude`, `depth`, `temp`
- Optional columns (for entropy analysis or N²): `salt`, `pres`, `ctemp`, `asal`
- File naming convention (for batch mode):  
  `BOA_Argo_YYYY_MM.csv` (e.g., `BOA_Argo_2010_01.csv`)
- Files should be located in a directory such as `input_data/`

---

##  Installation

```bash
pip install numpy pandas scipy scikit-learn gsw
