# Supplementary S1: methods notes and tables

All values in this document are derived from the CSV files in this dataset; the generating script is `supplementary_analysis/make_supplementary_tables.py`. E_H is the annual sum of the archived daily heating series, with no boiler efficiency or fuel conversion applied; D24 counts days with the selected zone's daily-mean air temperature below 24 °C.

## S1.0 Methods notes

**Surrogate feature encoding.** The six model inputs (weather-case label; window U-value; ground-floor, façade, and roof thermal resistances; infiltration design flow) enter the feature vector with the values stored in the reduced dataset. For the opaque states these are the dataset values listed in Table S1.2; for example, ground-floor state 1 enters as 4.8 m²K/W and roof state 0 as 0.48 m²K/W, so the released out-of-fold predictions are exactly reproducible from the released files. The window U-values are identical in both sources.

**Random-forest comparator.** 300 trees, unlimited depth, minimum leaf size 1, all features considered at each split, bootstrap resampling; fitted on the same standardised inputs and targets and the same inner-training rows as the networks, with the early-stopping partition unused.


**Gradient-boosting comparator.** One HistGradientBoostingRegressor per objective: squared-error loss, 300 boosting iterations, learning rate 0.1, no early stopping, seeded per fold and objective; fitted on the same standardised inputs and inner-training rows as the networks.

**Trainers and task designs.** The joint network is trained under equal weighting (unweighted mean of the four standardised MSEs) and under gradient balancing with the MGDA implementation (Table S1.12). Two task designs are evaluated: all four objectives predicted (predicted fronts use predicted cost/carbon) and the exact-index design (only the two simulated outcomes predicted; cost/carbon taken from the component tables). The exact reference is identical in both.

**Unique-selectability LP.** A Pareto member is *uniquely selectable* when some admissible weight vector makes it the unique minimiser of the Eq. (7) weighted sum over the front's Eq. (6)-normalised objectives, under the nonnegative normalised weighted-sum model with weight floor w_j ≥ 10⁻⁶. For each member the LP maximises the worst-case score margin t\* over admissible weights; a member is classified uniquely selectable when t\* > 10⁻⁹ and not uniquely selectable otherwise. No finer boundary/unsupported split is reported, because several margins sit at solver-tolerance scale (e.g. −4.9×10⁻⁸) and certifying such a split would need exact rather than floating-point arithmetic. Per-member t\* values (scientific notation) are in `mcda_outputs.csv`.

**Tie conventions.** The five named-profile selections (weighted sum and achievement scalarisation) treat scores within an absolute tolerance of 10⁻¹² as tied (`np.isclose` with rtol = 0.0) and break ties towards the lowest simulation identifier. The 10,000-draw sweeps use exact-equality argmin over identifier-sorted scores, with exact ties broken by the lowest identifier.

**Achievement-scalarising comparison.** The alternative rule minimises max_i w_i f̄_i(x) + ρ Σ_i w_i f̄_i(x) with ρ = 10⁻⁶ over each front, with f̄_i the Eq. (6)-normalised objectives (ideal reference point at the origin) and the same named-profile tie rule. It selects a different package from the weighted sum in 14 of the 15 weather case–profile cases, the mid-century D24-priority case alone agreeing; its equal-weight choice, package 138, is not uniquely selectable under the nonnegative normalised weighted-sum model with w_j ≥ 10⁻⁶ and the stated solver tolerance, in any weather case. The 15 achievement-scalarising selections are unchanged for ρ ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²} (per-case selections in `mcda_profile_sensitivity.csv`).

**Weight-space selection agreement.** For each task design, formulation and repeat–weather case composite, the predicted front is the Pareto set of the out-of-fold predicted objective vectors over all 625 packages (predicted D24 clipped to [0, 365] and, in the all-predicted design, the predicted cost and GWP indices clipped at zero; in the exact-index design the two tabulated indices enter exactly). For each of the 10,000 Section 3.4 weight draws, both the predicted and the exact front are normalised over their own ideal and nadir (Eq. 6), scored with the weighted sum (Eq. 7) under the sweep tie rule, and the agreement is the share of draws on which both fronts select the same simulation identifier (Table S1.14).

**Normalisation-anchor check.** The 15 profile selections are recomputed with the Eq. (6) scaling constants taken over all 625 packages of the weather case instead of the front's ideal and nadir; the candidate set and tie rule are unchanged. Results follow Table S1.7.

**Warm-side baseline safeguard.** Front members whose D24 falls below the same-case baseline count are removed before scoring; the surviving members are scored with the front's original Eq. (6) normalisation (a feasibility screen, not a new scaling) and the named-profile tie rule. Re-normalising over the screened subset instead also moves further near-tied selections (three at mid-century, four at late century), consistent with the scale dependence documented above; the reported safeguard keeps the original scaling. Results follow Table S1.7.

**First-rank acceptability and central weight vectors.** One million Dirichlet(1,1,1,1) draws (seed 42, chunked) are scored per weather case under Eq. (6)–(7) with the sweep tie rule; every ever-winning package's first-rank acceptability a_i and central weight vector w^c (the mean of its winning draws) are in `mcda_central_weights.csv`, generated by `supplementary_analysis/mcda_central_weights.py`. 9,604 draws already bound the Monte Carlo error on first-rank shares to ±0.01 at 95% confidence (Tervonen and Figueira, 2008); the leading packages' one-million-draw shares differ from the 10,000-draw estimates by at most 0.4 percentage points, with the same leading package in every weather case, and the larger sample stabilises the central vectors of low-acceptability packages. Sampled minimum and maximum winning weights are not reported: sample extrema are not theoretical weight bounds.

## Table S1.1: Simulation assumptions

| Setting | Value |
|---|---|
| Geometry | 4.7 m x 7.0 m footprint; ground-storey, first-storey, and roof/attic thermal zones |
| Side boundaries | One ground-storey side surface adiabatic; remaining side surfaces exposed to outdoor air without solar or wind exposure |
| Windows | 32% window-to-wall ratio on north and south facades; generator SHGC 0.70, while reduced-dataset metadata records 0.50 |
| Space heating | Hydronic baseboards in all three zones connected to a natural-gas hot-water boiler |
| Thermostats | 17.5 degrees C heating setpoint throughout the year in all zones; 100 degrees C cooling setpoint, effectively disabling cooling |
| Numerical controls | Six timesteps per hour (10-minute timestep); ConductionTransferFunction heat-balance algorithm |
| Run period | Full calendar year using the selected weather file |
| Occupancy | Prescribed schedule with a density of 0.0565 person/m2 applied to all three zones |
| Other internal gains | Residential lighting at 5 W/m2 and equipment at 3 W/m2, applied to the ground-storey zone only |
| Ventilation | Scheduled natural ZoneVentilation flow of 0.0137 m3/s per zone; no heat recovery represented by this object |
| Infiltration | Configuration-dependent Flow/Zone design flow of 0.0135--0.025 m3/s per zone |
| Solar control | Exterior blinds on four of eight windows, scheduled off October--March and on April--September |
| Ground boundary | Monthly Site:GroundTemperature:BuildingSurface values, held fixed across all three weather cases |
| Thermal bridges | Not modelled |

## Table S1.2: Retrofit state specifications and bundled infiltration factors

`generator_value` is the thickness/conductivity-derived property in the generator script; `archive_metadata_value` is the value stored in the reduced dataset and used as the surrogate input. The infiltration factor multiplies the 0.025 m³/s baseline Flow/Zone rate through the component-weighted generator equation (clipped to 0.013–0.025 m³/s).

| Component | State | Source option label | Property | Generator | Archive | Cost rate (€/m²) | GWP A1–A3 (kgCO₂e/m²) | Infiltration factor |
|---|---|---|---|---|---|---|---|---|
| window | 0 | Wooden double glazing | U-factor | 2.9 | 2.9 | 0.0 | 0.0 | 1.0 |
| window | 1 | HR++ double, plastic frame | U-factor | 1.2 | 1.2 | 184.0 | 70.0 | 0.8 |
| window | 2 | HR++ double, wooden frame | U-factor | 1.21 | 1.21 | 485.0 | 50.0 | 0.7 |
| window | 3 | Triple, plastic frame | U-factor | 0.8 | 0.8 | 295.0 | 150.0 | 0.6 |
| window | 4 | Triple, wooden frame | U-factor | 0.81 | 0.81 | 622.0 | 120.0 | 0.5 |
| ground_floor | 0 | Uninsulated timber | R-value | 0.41 | 0.41 | 0.0 | 0.0 | 1.0 |
| ground_floor | 1 | PIR | R-value | 4.0 | 4.8 | 59.7 | 10.0 | 0.9 |
| ground_floor | 2 | Hemp fibre | R-value | 5.0 | 5.0 | 77.0 | 5.92 | 0.8 |
| ground_floor | 3 | Resol | R-value | 5.5 | 5.5 | 87.9 | 11.0 | 0.7 |
| ground_floor | 4 | Hemp fibre | R-value | 5.67 | 5.6 | 108.0 | 7.0 | 0.7 |
| windowed_facade | 0 | Solid clay brick | R-value | 0.45 | 0.45 | 0.0 | 0.0 | 1.0 |
| windowed_facade | 1 | EPS | R-value | 4.2 | 4.2 | 182.0 | 9.36 | 0.8 |
| windowed_facade | 2 | Hemp fibre | R-value | 4.4 | 4.4 | 179.0 | 4.83 | 0.6 |
| windowed_facade | 3 | EPS | R-value | 6.5 | 6.5 | 200.0 | 17.16 | 0.5 |
| windowed_facade | 4 | Hemp fibre | R-value | 6.75 | 6.7 | 222.0 | 8.5 | 0.5 |
| roof | 0 | Uninsulated tile | R-value | 0.49 | 0.48 | 0.0 | 0.0 | 1.0 |
| roof | 1 | Mineral wool | R-value | 4.5 | 4.5 | 89.5 | 23.29 | 0.9 |
| roof | 2 | Hemp fibre | R-value | 4.75 | 4.7 | 105.0 | 4.76 | 0.8 |
| roof | 3 | PIR | R-value | 8.5 | 8.5 | 101.0 | 18.5 | 0.7 |
| roof | 4 | Hemp fibre | R-value | 8.75 | 8.7 | 139.0 | 10.68 | 0.7 |

## Table S1.3: Out-of-fold prediction accuracy, pooled over weather cases (all-predicted task design)

Mean ± SD over the 15 package-grouped outer folds (3 seeds × 5 folds); folds share packages across seeds and are not independent replicates. All four objectives are predicted; exact-index accuracies for the two simulated outcomes are in Table S1.11.

| Model | Target | MAE | RMSE | R² |
|---|---|---|---|---|
| Joint NN, equal weighting | E_H (GJ/year) | 0.120 ± 0.023 | 0.189 ± 0.067 | 0.9986 ± 0.0013 |
|  | D24 (days/year) | 0.887 ± 0.134 | 1.295 ± 0.336 | 0.9985 ± 0.0009 |
|  | I_C (index points) | 29.0 ± 4.5 | 39.7 ± 8.6 | 0.9714 ± 0.0127 |
|  | I_G (index points) | 6.27 ± 0.32 | 7.53 ± 0.39 | 0.9802 ± 0.0022 |
| Joint NN, gradient-balanced | E_H (GJ/year) | 0.085 ± 0.019 | 0.127 ± 0.034 | 0.9994 ± 0.0003 |
|  | D24 (days/year) | 0.620 ± 0.120 | 0.920 ± 0.213 | 0.9993 ± 0.0004 |
|  | I_C (index points) | 59.0 ± 4.9 | 86.5 ± 5.9 | 0.8687 ± 0.0193 |
|  | I_G (index points) | 7.78 ± 0.49 | 9.49 ± 0.56 | 0.9685 ± 0.0041 |
| Single-task NNs | E_H (GJ/year) | 0.083 ± 0.009 | 0.121 ± 0.017 | 0.9995 ± 0.0002 |
|  | D24 (days/year) | 0.562 ± 0.039 | 0.847 ± 0.064 | 0.9994 ± 0.0001 |
|  | I_C (index points) | 29.8 ± 8.5 | 42.4 ± 14.1 | 0.9655 ± 0.0267 |
|  | I_G (index points) | 7.27 ± 0.38 | 8.83 ± 0.57 | 0.9726 ± 0.0043 |
| Random forest | E_H (GJ/year) | 0.283 ± 0.053 | 0.579 ± 0.103 | 0.9885 ± 0.0038 |
|  | D24 (days/year) | 0.663 ± 0.117 | 1.282 ± 0.332 | 0.9986 ± 0.0008 |
|  | I_C (index points) | 17.9 ± 1.7 | 24.1 ± 3.6 | 0.9898 ± 0.0030 |
|  | I_G (index points) | 5.61 ± 0.36 | 7.05 ± 0.43 | 0.9826 ± 0.0020 |
| Gradient-boosted trees | E_H (GJ/year) | 0.128 ± 0.014 | 0.228 ± 0.035 | 0.9982 ± 0.0005 |
|  | D24 (days/year) | 0.353 ± 0.023 | 0.574 ± 0.058 | 0.9997 ± 0.0001 |
|  | I_C (index points) | 9.6 ± 1.8 | 15.9 ± 3.1 | 0.9955 ± 0.0016 |
|  | I_G (index points) | 1.11 ± 0.15 | 1.53 ± 0.20 | 0.9992 ± 0.0002 |

## Table S1.4: Weather-case-specific prediction accuracy, both task designs

Mean ± descriptive SD over 15 package-grouped folds. The cost and carbon indices are identical across weather cases, so only the two simulated outcomes are shown per case. E_H MAE is in GJ/year and D24 MAE in days.

**Exact-index task design**

| Model | Weather case | E_H MAE | E_H R² | D24 MAE | D24 R² |
|---|---|---|---|---|---|
| Joint NN, equal weighting | Present | 0.093 ± 0.013 | 0.9989 ± 0.0005 | 0.346 ± 0.066 | 0.8666 ± 0.0881 |
|  | Mid-century | 0.081 ± 0.008 | 0.9988 ± 0.0005 | 0.577 ± 0.056 | 0.9975 ± 0.0007 |
|  | Late-century | 0.069 ± 0.010 | 0.9984 ± 0.0009 | 0.793 ± 0.060 | 0.9971 ± 0.0006 |
| Joint NN, gradient-balanced | Present | 0.095 ± 0.013 | 0.9985 ± 0.0009 | 0.373 ± 0.094 | 0.8329 ± 0.1743 |
|  | Mid-century | 0.083 ± 0.014 | 0.9985 ± 0.0009 | 0.584 ± 0.072 | 0.9974 ± 0.0008 |
|  | Late-century | 0.069 ± 0.014 | 0.9981 ± 0.0011 | 0.823 ± 0.097 | 0.9969 ± 0.0008 |
| Single-task NNs | Present | 0.097 ± 0.011 | 0.9986 ± 0.0005 | 0.336 ± 0.064 | 0.8791 ± 0.0823 |
|  | Mid-century | 0.084 ± 0.010 | 0.9988 ± 0.0004 | 0.565 ± 0.041 | 0.9974 ± 0.0007 |
|  | Late-century | 0.070 ± 0.011 | 0.9983 ± 0.0010 | 0.784 ± 0.068 | 0.9971 ± 0.0005 |
| Random forest | Present | 0.208 ± 0.037 | 0.9892 ± 0.0058 | 0.152 ± 0.029 | 0.9424 ± 0.0226 |
|  | Mid-century | 0.203 ± 0.042 | 0.9809 ± 0.0094 | 0.398 ± 0.082 | 0.9975 ± 0.0017 |
|  | Late-century | 0.162 ± 0.033 | 0.9804 ± 0.0095 | 0.597 ± 0.122 | 0.9974 ± 0.0019 |
| Gradient-boosted trees | Present | 0.164 ± 0.019 | 0.9944 ± 0.0020 | 0.150 ± 0.021 | 0.9762 ± 0.0066 |
|  | Mid-century | 0.123 ± 0.017 | 0.9956 ± 0.0018 | 0.354 ± 0.033 | 0.9988 ± 0.0004 |
|  | Late-century | 0.097 ± 0.011 | 0.9955 ± 0.0015 | 0.555 ± 0.042 | 0.9985 ± 0.0004 |

**All-predicted task design**

| Model | Weather case | E_H MAE | E_H R² | D24 MAE | D24 R² |
|---|---|---|---|---|---|
| Joint NN, equal weighting | Present | 0.132 ± 0.027 | 0.9967 ± 0.0034 | 0.748 ± 0.180 | 0.2829 ± 0.7700 |
|  | Mid-century | 0.123 ± 0.028 | 0.9960 ± 0.0048 | 0.887 ± 0.147 | 0.9934 ± 0.0037 |
|  | Late-century | 0.106 ± 0.018 | 0.9954 ± 0.0031 | 1.025 ± 0.105 | 0.9953 ± 0.0015 |
| Joint NN, gradient-balanced | Present | 0.094 ± 0.020 | 0.9987 ± 0.0009 | 0.411 ± 0.158 | 0.7588 ± 0.3863 |
|  | Mid-century | 0.089 ± 0.023 | 0.9982 ± 0.0011 | 0.619 ± 0.120 | 0.9969 ± 0.0013 |
|  | Late-century | 0.073 ± 0.018 | 0.9980 ± 0.0014 | 0.829 ± 0.105 | 0.9969 ± 0.0009 |
| Single-task NNs | Present | 0.097 ± 0.011 | 0.9986 ± 0.0005 | 0.336 ± 0.064 | 0.8791 ± 0.0823 |
|  | Mid-century | 0.084 ± 0.010 | 0.9988 ± 0.0004 | 0.565 ± 0.041 | 0.9974 ± 0.0007 |
|  | Late-century | 0.070 ± 0.011 | 0.9983 ± 0.0010 | 0.784 ± 0.068 | 0.9971 ± 0.0005 |
| Random forest | Present | 0.313 ± 0.057 | 0.9742 ± 0.0087 | 0.244 ± 0.048 | 0.8926 ± 0.0481 |
|  | Mid-century | 0.298 ± 0.061 | 0.9638 ± 0.0118 | 0.878 ± 0.222 | 0.9869 ± 0.0080 |
|  | Late-century | 0.237 ± 0.044 | 0.9609 ± 0.0124 | 0.867 ± 0.125 | 0.9962 ± 0.0016 |
| Gradient-boosted trees | Present | 0.164 ± 0.019 | 0.9944 ± 0.0020 | 0.150 ± 0.021 | 0.9762 ± 0.0066 |
|  | Mid-century | 0.123 ± 0.017 | 0.9956 ± 0.0018 | 0.354 ± 0.033 | 0.9988 ± 0.0004 |
|  | Late-century | 0.097 ± 0.011 | 0.9955 ± 0.0015 | 0.555 ± 0.042 | 0.9985 ± 0.0004 |


## Table S1.5: Decision-fidelity summary, both task designs

Pareto precision/recall/F1: mean ± SD over the 9 seed–weather case composite fronts. Exact agreement: profile selections (of 15 unique weather case–profile cases) reproduced exactly, per repeat seed. Regret: Eq. 10 true-score regret over all 45 selection cases. In the all-predicted design the predicted fronts use predicted cost/carbon; the exact reference never changes.

| Task design | Model | Precision | Recall | F1 | Exact agreement (seeds 17/29/43) | On exact front | Regret median | Regret p90 | Regret max |
|---|---|---|---|---|---|---|---|---|---|
| all-predicted | Joint NN, equal weighting | 0.665 ± 0.018 | 0.782 ± 0.032 | 0.718 ± 0.011 | 4/2/2 of 15 | 41 of 45 | 0.0210 | 0.064 | 0.472 |
|  | Joint NN, gradient-balanced | 0.605 ± 0.028 | 0.802 ± 0.029 | 0.689 ± 0.023 | 1/2/0 of 15 | 36 of 45 | 0.0141 | 0.054 | 0.121 |
|  | Single-task NNs | 0.690 ± 0.029 | 0.759 ± 0.044 | 0.723 ± 0.032 | 3/2/4 of 15 | 40 of 45 | 0.0083 | 0.050 | 0.114 |
|  | Random forest | 0.697 ± 0.022 | 0.922 ± 0.010 | 0.793 ± 0.015 | 1/5/2 of 15 | 45 of 45 | 0.0069 | 0.047 | 0.065 |
|  | Gradient-boosted trees | 0.777 ± 0.031 | 0.897 ± 0.011 | 0.832 ± 0.017 | 6/5/6 of 15 | 43 of 45 | 0.0013 | 0.017 | 0.113 |
| exact-index | Joint NN, equal weighting | 0.845 ± 0.020 | 0.944 ± 0.026 | 0.891 ± 0.008 | 5/6/7 of 15 | 43 of 45 | 0.0016 | 0.022 | 0.114 |
|  | Joint NN, gradient-balanced | 0.843 ± 0.030 | 0.937 ± 0.024 | 0.887 ± 0.017 | 7/4/5 of 15 | 45 of 45 | 0.0013 | 0.029 | 0.048 |
|  | Single-task NNs | 0.841 ± 0.029 | 0.948 ± 0.017 | 0.891 ± 0.015 | 7/6/6 of 15 | 41 of 45 | 0.0013 | 0.025 | 0.117 |
|  | Random forest | 0.837 ± 0.026 | 0.894 ± 0.036 | 0.864 ± 0.026 | 6/8/6 of 15 | 45 of 45 | 0.0002 | 0.017 | 0.113 |
|  | Gradient-boosted trees | 0.837 ± 0.036 | 0.911 ± 0.017 | 0.872 ± 0.019 | 6/6/7 of 15 | 45 of 45 | 0.0004 | 0.017 | 0.113 |

## Table S1.6: Weather-file descriptors

| Weather case | EPW header | SHA-256 (first 16) | Annual mean T (°C) | JJA mean T (°C) | HDD18 (°C·day) | CDD18 (°C·day) | JJA daily-mean p95 (°C) |
|---|---|---|---|---|---|---|---|
| Present TMYx | Lelystad Airport; Climate.OneBuilding SRC-TMYx; 1994-2023 | 15fd5f381b3e01dc… | 10.46 | 17.15 | 2826.6 | 76.2 | 22.24 |
| Mid-century | Future Weather Generator v1.4.0; 11-model ensemble; SSP2-4.5/2050 | bd00dc4d989609c8… | 12.57 | 20.40 | 2239.4 | 258.1 | 25.13 |
| Late-century | Future Weather Generator v1.4.0; 11-model ensemble; SSP5-8.5/2080 | 7a2cbb882de69077… | 14.50 | 21.88 | 1708.1 | 430.3 | 26.68 |

Full 64-character SHA-256 hashes are in `weather_file_descriptors.csv`. The hashes document the surviving EPW files themselves.


## Table S1.7: MCDA robustness summary per weather case

Unique selectability at weight floor 10⁻⁶ and threshold t\* > 10⁻⁹ (see S1.0); sweep = 10,000 uniform Dirichlet(1,1,1,1) draws, seed 42; joint sweep additionally samples T_u uniformly over {23, 24, 25, 26} °C. Full per-member results are in `supplementary_analysis/mcda_outputs.csv` and `mcda_joint_outputs.csv`.

| Weather case | Pareto members | Not uniquely selectable | Sweep: distinct winners | Sweep: 90% coverage | Joint: distinct winners | Joint: 90% coverage |
|---|---|---|---|---|---|---|
| Present | 242 | 189 of 242 | 47 | 15 | 62 | 26 |
| Mid-century | 258 | 207 of 258 | 46 | 13 | 73 | 13 |
| Late-century | 267 | 202 of 267 | 55 | 17 | 72 | 19 |

*The cumulative-coverage curves behind the 90% columns are drawn in the manuscript's decision-robustness figure, produced by `supplementary_analysis/make_decision_robustness_figure.py`.*

**Anchor and safeguard results.** With Eq. (6) scaling constants taken over all 625 packages, all 15 profile selections are unchanged. Under the warm-side baseline safeguard the only change is the late-century E_H-priority selection, package 250 → 400: the safeguarded selection recovers 54 below-threshold days (327 against 273) for +1.184 GJ of E_H, +3 cost-index points and +73 GWP-index points; all 14 other selections, including every present-weather and mid-century selection, are unchanged.


## Table S1.8: Pearson correlations between objectives, all 625 packages per weather case

| Weather case | E_H–I_C | E_H–I_G | I_C–I_G | E_H–D24 |
|---|---|---|---|---|
| Present | -0.750 | -0.606 | +0.548 | -0.334 |
| Mid-century | -0.745 | -0.592 | +0.548 | +0.478 |
| Late-century | -0.743 | -0.594 | +0.548 | +0.528 |

## Table S1.9: Per repeat–weather case Pareto precision and recall

Values are per-composite (seed × weather case); seeds ordered 17/29/43.

**All-predicted task design**

| Model | Weather case | Precision (seeds 17/29/43) | Recall (seeds 17/29/43) |
|---|---|---|---|
| Joint NN, equal weighting | Present | 0.640/0.646/0.658 | 0.793/0.793/0.802 |
|  | Mid-century | 0.672/0.666/0.648 | 0.787/0.787/0.829 |
|  | Late-century | 0.689/0.677/0.685 | 0.772/0.715/0.757 |
| Joint NN, gradient-balanced | Present | 0.589/0.589/0.561 | 0.810/0.810/0.760 |
|  | Mid-century | 0.624/0.582/0.602 | 0.841/0.837/0.764 |
|  | Late-century | 0.639/0.617/0.644 | 0.801/0.779/0.813 |
| Single-task NNs | Present | 0.658/0.694/0.664 | 0.661/0.769/0.744 |
|  | Mid-century | 0.737/0.703/0.669 | 0.795/0.798/0.775 |
|  | Late-century | 0.704/0.721/0.660 | 0.723/0.794/0.772 |
| Random forest | Present | 0.717/0.705/0.704 | 0.921/0.930/0.926 |
|  | Mid-century | 0.675/0.668/0.666 | 0.926/0.926/0.903 |
|  | Late-century | 0.695/0.718/0.721 | 0.921/0.933/0.910 |
| Gradient-boosted trees | Present | 0.742/0.758/0.757 | 0.905/0.905/0.888 |
|  | Mid-century | 0.774/0.776/0.779 | 0.903/0.911/0.899 |
|  | Late-century | 0.813/0.840/0.751 | 0.895/0.884/0.880 |

**Exact-index task design**

| Model | Weather case | Precision (seeds 17/29/43) | Recall (seeds 17/29/43) |
|---|---|---|---|
| Joint NN, equal weighting | Present | 0.822/0.818/0.832 | 0.975/0.967/0.942 |
|  | Mid-century | 0.835/0.852/0.837 | 0.942/0.957/0.957 |
|  | Late-century | 0.871/0.869/0.866 | 0.906/0.948/0.899 |
| Joint NN, gradient-balanced | Present | 0.829/0.828/0.836 | 0.963/0.975/0.950 |
|  | Mid-century | 0.809/0.843/0.838 | 0.938/0.938/0.922 |
|  | Late-century | 0.854/0.915/0.830 | 0.921/0.929/0.895 |
| Single-task NNs | Present | 0.813/0.819/0.816 | 0.955/0.975/0.955 |
|  | Mid-century | 0.866/0.844/0.803 | 0.950/0.965/0.930 |
|  | Late-century | 0.882/0.872/0.849 | 0.925/0.948/0.929 |
| Random forest | Present | 0.851/0.856/0.852 | 0.921/0.909/0.905 |
|  | Mid-century | 0.814/0.816/0.806 | 0.915/0.891/0.853 |
|  | Late-century | 0.817/0.883/0.835 | 0.903/0.929/0.816 |
| Gradient-boosted trees | Present | 0.786/0.820/0.808 | 0.913/0.921/0.921 |
|  | Mid-century | 0.843/0.837/0.837 | 0.919/0.934/0.915 |
|  | Late-century | 0.875/0.906/0.824 | 0.895/0.903/0.876 |


## Table S1.10: Complete weighted-sum profile selections (all five profiles × three weather cases)

States and materials follow
main-text Table 1 (model-generation values); R in m²K/W, window U in W/m²K. Reduction is in E_H
relative to the same-weather-case baseline. Sim-id is the simulation identifier.

| Weather case | Profile | Selected package (states, materials, executed R/U) | Sim-id | E_H red. (%) | D24 (d) |
|---|---|---|---|---|---|
| Present | E_H | Windows S4 triple wooden (U 0.81); floor S1 PIR (R 4.00); façade S4 hemp (R 6.75); roof S4 hemp (R 8.75) | 550 | 52.4 | 363 |
| | Cost | None (baseline, all state 0) | 1 | 0.0 | 361 |
| | GWP | Façade S2 hemp fibre (R 4.40) | 3 | 14.1 | 364 |
| | D24 | Façade S4 hemp (R 6.75) + roof S4 hemp (R 8.75) | 25 | 29.7 | 365 |
| | Equal-weight | Façade S4 hemp (R 6.75) + roof S4 hemp (R 8.75) | 25 | 29.7 | 365 |
| Mid-century | E_H | Windows S3 triple plastic (U 0.80); façade S4 hemp (R 6.75); roof S3 PIR (R 8.50) | 395 | 48.0 | 346 |
| | Cost | None (baseline) | 1 | 0.0 | 339 |
| | GWP | Roof S2 hemp fibre (R 4.75) | 11 | 13.9 | 340 |
| | D24 | Façade S4 hemp (R 6.75) + roof S2 hemp (R 4.75) | 15 | 31.4 | 343 |
| | Equal-weight | Façade S2 hemp (R 4.40) + roof S2 hemp (R 4.75) | 13 | 29.7 | 342 |
| Late-century | E_H | Windows S1 HR++ plastic (U 1.20); floor S4 hemp (R 5.67); façade S4 hemp (R 6.75); roof S4 hemp (R 8.75) | 250 | 52.5 | 273 |
| | Cost | None (baseline) | 1 | 0.0 | 321 |
| | GWP | None (baseline) | 1 | 0.0 | 321 |
| | D24 | Façade S4 hemp (R 6.75) + roof S2 hemp (R 4.75) | 15 | 30.0 | 326 |
| | Equal-weight | Façade S2 hemp (R 4.40) + roof S2 hemp (R 4.75) | 13 | 27.6 | 325 |

Under both warmer files the D24-priority profile stops short of the maximum attainable count
(343 of 346 days at mid-century; 326 of 329 late-century), because every maximum-count package
carries a far higher cost rate sum (the cheapest at mid-century sums to I_C = 584.5 against 327
for the selection), so the profile's residual 0.1 weights on the two indices outweigh the last
few days. At mid-century the E_H-priority profile overtakes the D24-priority profile on that count.

## Figure S1.1: Two ternary slices of the four-weight simplex

Weighted-sum winner regions, drawn as filled regions with white boundaries on a fine weight grid,
on ternary slices at w_G = 0.10 and 0.25 for each weather case; on a slice the three plotted
weights allocate the remaining 1 − w_G. Stars mark the weight profiles lying on each slice (E, C,
D; Bal = equal weights), and the plotted winner at every star point is verified against the Table S1.10
selections. Strong hues are reserved for the baseline (1), the equal-weight selections (25, 13),
the warmer-case D24-priority selection (15) and the E_H-priority winners (550/395/250, the small
regions at the E star); other regions take muted fills, with simulation ids printed at the region
medoid for regions of at least about 3% of a panel; smaller semantic winners keep leader-line
callouts. Region areas are conditional on the displayed
slice and are not full-simplex acceptability probabilities (Table S1.16 carries those). In the
present-weather w_G = 0.10 slice the D star sits on the boundary between packages 25 and 20, whose
scores are near-tied at that profile; the winner at the exact profile point is package 25. Every
region belongs to a uniquely selectable package; packages that are not uniquely selectable win no
region.
*(figures/fig_weight_regions.pdf)*

## Figure S1.2: Rank-acceptability heatmap

Rank-acceptability indices b_i^r for ranks 1–5 over the 10,000 uniformly sampled weight vectors, as
a single-hue heatmap for the eight packages with the highest first-rank acceptability per weather
case (cell values in percent, one decimal below 10; rows do NOT sum to 100% — the omitted
probability lies beyond rank 5). The do-nothing baseline is marked "base" and each case's
equal-weight selection "eq-wt" on the row labels. Only 47, 46 and 55 packages per weather case are ever selected first, and 13 to 17
cover 90% of the sampled weight space. The do-nothing baseline takes the largest first-rank share
under the warmer files (28% and 32%), largely because both incremental indices are zero at
baseline, reinforced by a favourable warm-side count there; it remains worst on heating.
*(figures/fig_rank_acceptability.pdf)*

## Figure S1.3: Shared-trunk network architectures by task design

The two separately fitted joint-model routes into the decision layer. In each route six inputs feed
a 128–128–64 shared trunk with SiLU activations. The all-predicted design has four 64–32–1 heads,
one per objective. The exact-index design is a separate refit with only the two simulated-outcome
heads; the component rate tables supply I_C and I_G. Both routes end in the same four-coordinate
objective vector from which fronts and selections are computed.
*(figures/fig_mtl_architecture.pdf)*

## Figure S1.4: Value paths of the exact Pareto fronts

Parallel-coordinate value paths over all four objectives, one panel per weather case: every exact
Pareto member is a light polyline over the four axes, each axis min–max scaled over that case's
front and oriented so that up is better (real best/worst values at the axis ends). Highlighted
paths: the do-nothing baseline (tops the two index axes, bottom of the heating axis), the
equal-weight selection (high on every axis, collapsing on none) and the E_H-priority selection
(tops the heating axis while dropping on cost and, at late century, on the warm-side axis; at
mid-century it also attains the front's maximum D24, the overtake noted under Table S1.10). This
is the standard many-objective view that no two-objective projection can carry.
*(figures/fig_value_paths.pdf; generated by `supplementary_analysis/make_value_paths_figure.py`)*

## Figure S1.5: The clipping mechanism in the worst-regret composite

Raw out-of-fold D24 predictions of the equally weighted joint network against the simulated range
(present weather, repeat seed 29, all-predicted design; n = 625, the worst-regret composite of
Section 3.3, not a representative fold). 84 packages are simulated at the 365-day maximum; 51 raw
predictions fall at or above 365 and are clipped there for the decision analysis (shaded). The
highlighted package (simulation id 16, simulated at 361 days) is predicted at 376.8 days; within
the clipped group its other predicted objectives make it the D24-priority winner, the
largest-regret selection (0.47). Neural and boosted-tree predictions show substantive exceedances
of the upper bound; the random forest’s exceedances are limited to floating-point roundoff
(Table S1.18).
*(figures/fig_d24_mechanism.pdf)*

## Figure S1.6: Cumulative coverage of first-rank outcomes

Cumulative first-rank share over the most-frequent winning packages per weather case, (a) under
weight sampling alone and (b) under joint weight and threshold sampling, with the 90%-coverage
crossings marked (15/13/17 and 26/13/19). The manuscript summarises these curves through the
coverage bars in Figure 5(d); the full curves are kept here.
*(figures/fig_coverage_curves.pdf; generated by `supplementary_analysis/make_decision_robustness_figure.py`)*

## Figure S1.7: Attainable heating–temperature trade-offs by ground-floor state

(a) Each curve gives the largest $\Delta D_{24}=D_{24}(\mathbf{x})-D_{24}(\mathrm{baseline})$
among packages delivering at least $r$ GJ/year of heating savings, separately for
packages with the ground floor at state 0 and with floor insulation. The baseline
uses the same weather case. Negative values mean more days with first-storey
daily-mean air temperature at or above 24 °C. The higher family curve gives the
unrestricted result. End markers show each family's maximum heating savings.
Callouts compare the best attainable $D_{24}$ at the limit without floor
insulation with the best attainable value when strictly greater heating savings
are required. These discrete family switches reduce attainable $D_{24}$ by
2, 39 and 48 days under present, mid-century and late-century weather.
They are not matched changes to a single package.

(b, c) Matched effects of replacing state 0 with the lowest-U window state or
highest-R opaque-component state, including the assigned infiltration change,
while holding the other components fixed. Heating savings are state 0 minus the
strongest state; day-count changes are the strongest state minus state 0.
Positive values indicate improvement in both panels. Dots show medians, thick
bars interquartile ranges and capped thin bars full ranges over 125 combinations
of the other components per weather case. These ranges describe the finite
configuration set, not statistical confidence intervals. Manuscript Figures 2
and 3 instead display additional above-threshold days, with the opposite sign
to $\Delta D_{24}$. Figure 3 further groups window, façade and roof comparisons
by the unchanged floor option, with 25 pairs for the existing option and 100
for insulated options. The pooled summaries here retain all 125 pairs together.

*(figures/fig_attainment_curves.pdf; generated by
`supplementary_analysis/make_attainment_s1_figure.py`.)*

## Table S1.11: Full model screen, both task designs (nine formulations)

Pooled out-of-fold MAE (mean over 15 folds) per objective, with Pareto recall (mean over 9 composites) and exact selection agreement (of 45). The Separate and Deep-balanced rows use the alternative architectures described in Table S1.12.

**All-predicted task design**

| Model | E_H (GJ/year) MAE | D24 (days/year) MAE | I_C (index points) MAE | I_G (index points) MAE | Pareto recall | Agreement |
|---|---|---|---|---|---|---|
| Joint NN, equal weighting | 0.120 | 0.887 | 29.0 | 6.27 | 0.782 | 8 of 45 |
| Joint NN, gradient-balanced | 0.085 | 0.620 | 59.0 | 7.78 | 0.802 | 3 of 45 |
| Separate family, equal weighting | 0.104 | 0.800 | 53.0 | 7.30 | 0.792 | 5 of 45 |
| Separate family, gradient-balanced | 0.097 | 0.707 | 53.7 | 7.35 | 0.780 | 3 of 45 |
| Deep-balanced family, equal weighting | 0.999 | 3.072 | 114.2 | 12.15 | 0.642 | 4 of 45 |
| Deep-balanced family, gradient-balanced | 0.997 | 2.460 | 117.3 | 13.09 | 0.568 | 4 of 45 |
| Single-task NNs | 0.083 | 0.562 | 29.8 | 7.27 | 0.759 | 9 of 45 |
| Random forest | 0.283 | 0.663 | 17.9 | 5.61 | 0.922 | 8 of 45 |
| Gradient-boosted trees | 0.128 | 0.353 | 9.6 | 1.11 | 0.897 | 17 of 45 |

**Exact-index task design**

| Model | E_H (GJ/year) MAE | D24 (days/year) MAE | Pareto recall | Agreement |
|---|---|---|---|---|
| Joint NN, equal weighting | 0.081 | 0.572 | 0.944 | 18 of 45 |
| Joint NN, gradient-balanced | 0.082 | 0.593 | 0.937 | 16 of 45 |
| Separate family, equal weighting | 0.090 | 0.688 | 0.934 | 15 of 45 |
| Separate family, gradient-balanced | 0.091 | 0.688 | 0.929 | 13 of 45 |
| Deep-balanced family, equal weighting | 1.329 | 4.516 | 0.634 | 15 of 45 |
| Deep-balanced family, gradient-balanced | 1.382 | 4.347 | 0.601 | 12 of 45 |
| Single-task NNs | 0.083 | 0.562 | 0.948 | 19 of 45 |
| Random forest | 0.191 | 0.383 | 0.894 | 20 of 45 |
| Gradient-boosted trees | 0.128 | 0.353 | 0.911 | 19 of 45 |


## Table S1.12: Architecture screen and trainer notes

Two further architecture families were screened on identical folds (full rows in Table S1.11): the Separate family (one shared layer, deep task branches, linear outputs for standardised targets) and the Deep-balanced family (200–100 shared trunk with dropout 0.5, asymmetric task heads). The deeper architecture did not consistently improve prediction accuracy or selection agreement relative to the shared-trunk network. The Separate family with equal weighting improves on the shared trunk on the two simulated outcomes in the all-predicted design (heating MAE 0.104 against 0.120 GJ/year; day-count MAE 0.800 against 0.887 days/year) but is worse on the two rate-table indices (cost 53.0 against 29.0; GWP 7.30 against 6.27 index points) and reproduces 5 of the 45 exact selections against 8; in the exact-index design it is behind the shared trunk on both simulated outcomes and on exact selection agreement. Neither family displaces the main formulation. The Deep-balanced collapse is consistent with excessive dropout for the inner-training sample, 0.5 against 425 training packages (1,275 rows): a one-seed diagnostic at dropout 0.1 recovers its exact-index heating MAE to 0.334 GJ/year and its day-count MAE to 1.423 days/year, still clearly behind the shared trunk.

**Trainer implementation note.** MGDA replaces the shared-trunk gradient with a minimum-norm convex combination of task gradients, while each output head retains its own gradient. Combination weights are computed in closed form for two tasks and by deterministic Frank–Wolfe iteration for four.

## Table S1.13: Substitution-only ablation

The exact-index design changes two things at once for the joint networks and the random forest: the two indices are supplied from the component tables, and the model is refitted on two targets instead of four. The substitution-only arm changes only the first. It retains the four-target models' predicted $E_H$ and $D_{24}$ and replaces the two indices by their table values, with no model refitted, so the two effects can be read separately. The single-task networks and the per-objective boosted trees fit each target independently, so for them the substitution-only and exact-index arms coincide by construction.

Agreement is the mean over the nine repeat-weather case composites, per cent; selections are exact profile selections of 45.

| Formulation | all-predicted agr. | all-predicted sel. | substitution-only agr. | substitution-only sel. | exact-index agr. | exact-index sel. |
|---|---|---|---|---|---|---|
| Joint NN, equal weighting | 9.7 | 8 | 37.7 | 17 | 49.6 | 18 |
| Joint NN, gradient-balanced | 10.0 | 3 | 46.8 | 19 | 46.4 | 16 |
| Single-task NNs | 17.8 | 9 | 50.7 | 19 | 50.7 | 19 |
| Random forest | 13.9 | 8 | 47.2 | 22 | 46.5 | 20 |
| Gradient-boosted trees | 47.2 | 17 | 52.9 | 19 | 52.9 | 19 |
| **Mean over the five / total of 225** | **19.7** | **45** | **47.0** | **96** | **49.2** | **92** |

Substituting the indices accounts for 93 per cent of the mean weight-space gain between the all-predicted and exact-index arms (19.7 to 47.0 against 49.2 per cent). Exact selections move the same way, 45 of 225 to 96 of 225 under substitution alone and 92 of 225 after refitting. The improvement therefore came from substituting the deterministic indices, while refitting contributed the remainder.

## Table S1.14: Weight-space selection agreement, all nine formulations

Share of the 10,000 uniform Dirichlet(1,1,1,1) weight draws (seed 42, the Section 3.4
draws) for which the reconstructed and exact fronts select the same package under the
identical Eq. (6)-(7) protocol with the sweep tie rule. Predictions are returned to their
feasible ranges before decision analysis: $D_{24}$ to [0, 365] and the predicted cost and
GWP indices to [0, inf). Mean ± SD over the nine repeat-weather case composites; the SD
describes between-composite variation, not a confidence interval, since all composites
share the same draws. Generated by `mcda_weight_agreement.py`; per-composite values in
`mcda_weight_agreement.csv`. The first five rows of each arm are the main-text
formulations (Table 3).

| Task design | Model | Weight-space agreement (%) |
|---|---|---|
| all-predicted | Joint NN, equal weighting | 9.7 ± 8.0 |
|  | Joint NN, gradient-balanced | 10.0 ± 5.4 |
|  | Single-task NNs | 17.8 ± 8.7 |
|  | Random forest | 13.9 ± 8.3 |
|  | Gradient-boosted trees | 47.2 ± 15.8 |
|  | Separate family, equal weighting | 7.9 ± 3.6 |
|  | Separate family, gradient-balanced | 8.6 ± 7.0 |
|  | Deep-balanced family, equal weighting | 7.2 ± 5.8 |
|  | Deep-balanced family, gradient-balanced | 9.7 ± 11.4 |
| substitution-only | Joint NN, equal weighting | 37.7 ± 13.9 |
|  | Joint NN, gradient-balanced | 46.8 ± 11.3 |
|  | Single-task NNs | 50.7 ± 11.7 |
|  | Random forest | 47.2 ± 10.1 |
|  | Gradient-boosted trees | 52.9 ± 16.2 |
|  | Separate family, equal weighting | 38.4 ± 7.9 |
|  | Separate family, gradient-balanced | 35.3 ± 13.7 |
|  | Deep-balanced family, equal weighting | 28.2 ± 13.2 |
|  | Deep-balanced family, gradient-balanced | 26.4 ± 13.4 |
| exact-index | Joint NN, equal weighting | 49.6 ± 13.2 |
|  | Joint NN, gradient-balanced | 46.4 ± 8.7 |
|  | Single-task NNs | 50.7 ± 11.7 |
|  | Random forest | 46.5 ± 11.8 |
|  | Gradient-boosted trees | 52.9 ± 16.2 |
|  | Separate family, equal weighting | 41.2 ± 16.7 |
|  | Separate family, gradient-balanced | 41.7 ± 12.8 |
|  | Deep-balanced family, equal weighting | 26.3 ± 11.4 |
|  | Deep-balanced family, gradient-balanced | 24.9 ± 12.8 |

## Table S1.15: Equal-weight exchange equivalents of one D24 day

Eq. (6) spans of each weather case's Pareto front (its ideal-to-nadir range per objective) and the
resulting equal-weight exchange rates: sacrificing one below-threshold day is score-neutral against
the tabulated amounts of each other objective. The cost- and carbon-index spans are identical in all
three cases (I_C 1091.0 points, I_G 200.45 points), since the indices do not depend on weather. For
the D24-priority stress profile (weight 0.7 against 0.1), multiply each equivalent by seven.

| Weather case | D24 span (d) | E_H span (GJ) | 1 day ≡ E_H (GJ) | 1 day ≡ I_C (pts) | 1 day ≡ I_G (pts) |
|---|---|---|---|---|---|
| Present | 6 | 20.10 | 3.35 | 181.8 | 33.4 |
| Mid-century | 46 | 17.21 | 0.374 | 23.7 | 4.36 |
| Late-century | 66 | 13.07 | 0.198 | 16.5 | 3.04 |

## Table S1.16: First-rank acceptability and central weight vectors (1,000,000 draws)

Top three packages per weather case by first-rank acceptability a_i, with central weight vectors
w^c = (E_H, I_C, I_G, D24), both from the same 1,000,000-draw run (see S1.0); all ever-winning
packages are in `mcda_central_weights.csv`. The present-weather leader is the equal-weight
selection (package 25); under both warmer files the leader is the do-nothing baseline (package 1),
most often selected where cost receives the largest individual weight while the environmental and
warm-side objectives retain substantial combined weight.

| Weather case | Package | a_i (%) | w^c (E_H, I_C, I_G, D24) |
|---|---|---|---|
| Present | 25 | 25.19 | (0.197, 0.113, 0.331, 0.359) |
|  | 19 | 14.61 | (0.202, 0.381, 0.072, 0.345) |
|  | 3 | 11.46 | (0.070, 0.346, 0.389, 0.196) |
| Mid-century | 1 | 28.03 | (0.074, 0.425, 0.302, 0.199) |
|  | 15 | 16.69 | (0.242, 0.107, 0.292, 0.359) |
|  | 13 | 13.69 | (0.220, 0.188, 0.436, 0.155) |
| Late-century | 1 | 32.14 | (0.081, 0.412, 0.297, 0.210) |
|  | 15 | 18.15 | (0.241, 0.111, 0.279, 0.369) |
|  | 13 | 11.32 | (0.184, 0.170, 0.438, 0.208) |

## Table S1.17: Weighted-sum and achievement-scalarising selections per case and profile

Per-case per-profile selections under the two scalarising rules (from
`mcda_profile_sensitivity.csv`); the achievement-scalarising selections are identical for
ρ ∈ {10⁻⁸, 10⁻⁶, 10⁻⁴, 10⁻²}. The two rules agree in 1 of the 15 cases.

| Weather case | Profile | Weighted-sum package | ASF package | Same? |
|---|---|---|---|---|
| Present | Cost | 1 | 16 | no |
|  | GWP | 3 | 15 | no |
|  | Energy | 550 | 319 | no |
|  | D24 | 25 | 138 | no |
|  | Equal-weight | 25 | 138 | no |
| Mid-century | Cost | 1 | 16 | no |
|  | GWP | 11 | 15 | no |
|  | Energy | 395 | 294 | no |
|  | D24 | 15 | 15 | yes |
|  | Equal-weight | 13 | 138 | no |
| Late-century | Cost | 1 | 16 | no |
|  | GWP | 1 | 15 | no |
|  | Energy | 250 | 295 | no |
|  | D24 | 15 | 138 | no |
|  | Equal-weight | 13 | 138 | no |

## Table S1.18: Raw day-count predictions outside the physical bound

Share of the raw out-of-fold D24 predictions above 365 days, per formulation, task design and
weather case (percent of the 1,875 predictions per cell: 625 packages × 3 repeats). The reported
percentages use a strict comparison with 365 days. The forest’s exceedances are floating-point
roundoff, with a maximum prediction of 365.0000000000002 days; the neural and boosted-tree
formulations produce larger exceedances. All predictions are clipped to [0, 365] before
any decision analysis. No prediction falls below 0 in either design. From
`surrogate_pareto_validation.csv` of both designs.

| Task design | Model | Present | Mid-century | Late-century |
|---|---|---|---|---|
| all-predicted | Joint NN, equal weighting | 8.1% | 0.0% | 0.0% |
|  | Joint NN, gradient-balanced | 7.7% | 0.0% | 0.0% |
|  | Single-task NNs | 6.8% | 0.0% | 0.0% |
|  | Random forest | 0.1% | 0.0% | 0.0% |
|  | Gradient-boosted trees | 5.9% | 0.0% | 0.0% |
| exact-index | Joint NN, equal weighting | 7.9% | 0.0% | 0.0% |
|  | Joint NN, gradient-balanced | 8.4% | 0.0% | 0.0% |
|  | Single-task NNs | 6.8% | 0.0% | 0.0% |
|  | Random forest | 1.9% | 0.0% | 0.0% |
|  | Gradient-boosted trees | 5.9% | 0.0% | 0.0% |

## Table S1.19: Warm-side threshold sensitivity of the headline selections

Change in the below-threshold day count relative to the same-case unrenovated baseline (simulation
1), for the minimum-E_H package and for the equal-weight profile selection, evaluated at upper
thresholds of 23, 24, 25 and 26 °C. Both packages are chosen once, using the 24 °C objectives, and
then re-evaluated unchanged at every threshold; negative values mean more days at or above the
threshold. Recomputed from the released daily temperature series in
`inputs/{2020,2050,2100}_merged_simulation_results.csv`.

| Weather case | Package | Sim id | 23 °C | 24 °C | 25 °C | 26 °C |
|---|---|---|---|---|---|---|
| Present | Minimum-E_H | 625 | −3 | +1 | +1 | 0 |
|  | Equal-weight | 25 | +2 | +4 | +1 | 0 |
| Mid-century | Minimum-E_H | 625 | −41 | −39 | −28 | −18 |
|  | Equal-weight | 13 | −2 | +3 | +3 | +2 |
| Late-century | Minimum-E_H | 599 | −56 | −57 | −47 | −35 |
|  | Equal-weight | 13 | +4 | +4 | +2 | +2 |

The sign of the minimum-E_H penalty is unchanged across thresholds in both warmer cases, while its
magnitude falls as the threshold rises. Under present weather the same package stays within three
days of its baseline at every threshold. The equal-weight selection stays at or above its baseline
in all three cases at 24, 25 and 26 °C; the single exception across the twelve case–threshold
combinations is the mid-century selection at 23 °C, two days below.

## Table S1.20: Normalisation anchors and the training cap

**Normalisation anchors.** The exact and reconstructed fronts are each normalised over
their own ideal and nadir, so a reconstructed front with a different range could in
principle disagree for reasons of scaling rather than of package ordering. Repeating the
weight-space agreement with common anchors --- the exact front's ideal and nadir applied
to both --- moves agreement by 0.89 points on average across the 90 composites. The
largest formulation-level mean shift is 2.52 points and the largest single-composite
shift is 4.20 points. Written by `mcda_common_anchor.py` to `mcda_common_anchor.csv`.

| Task design | Model | Own anchors (%) | Common anchors (%) | Difference |
|---|---|---|---|---|
| all-predicted | Joint NN, equal weighting | 9.7 | 9.3 | -0.38 |
|  | Joint NN, gradient-balanced | 10.0 | 10.2 | +0.18 |
|  | Single-task NNs | 17.8 | 17.7 | -0.07 |
|  | Random forest | 13.9 | 14.3 | +0.34 |
|  | Gradient-boosted trees | 47.2 | 47.9 | +0.69 |
| exact-index | Joint NN, equal weighting | 49.6 | 50.2 | +0.55 |
|  | Joint NN, gradient-balanced | 46.4 | 46.5 | +0.17 |
|  | Single-task NNs | 50.7 | 50.7 | +0.01 |
|  | Random forest | 46.5 | 49.1 | +2.52 |
|  | Gradient-boosted trees | 52.9 | 53.8 | +0.95 |

**Training cap.** Neural training stops early on the inner validation partition or at
300 epochs, whichever comes first. The runs that reached the cap are:

| Formulation | Runs at the cap |
|---|---|
| Joint NN, equal weighting | 9 of 15 |
| Joint NN, gradient-balanced | 1 of 15 |
| Single-task NNs | 3 of 60 |
| Separate family, equal weighting | 3 of 15 |
| Separate family, gradient-balanced | 2 of 15 |

Nine of the fifteen equally weighted joint-network runs reached the 300-epoch cap; the
reported results for that formulation are conditional on this training budget.

## Table S1.21. Decision fidelity within held-out folds

Decision fidelity within each held-out fold. Every comparison uses the same 125 candidates for exact and predicted decisions. Each formulation and task design has 45 repeat–fold–weather evaluations and 225 named-profile comparisons. F1 and sampled-weight agreement are means with sample standard deviations over the 45 evaluations; these evaluations share data. Nondominance and regret refer to the exact front of the 125 candidates. The nondominated percentage and median regret include mismatches only.

Each formulation and task design has 3 repeat seeds × 5 held-out folds × 3 weather cases = 45 candidate-set evaluations, with 5 profiles each (225 profile comparisons). Sampled-weight agreement uses the same 10,000 Dirichlet(1,1,1,1) draws (seed 42) in every evaluation. Both fronts use their own ideal/nadir normalisation; selected packages are scored with the exact-front normalisation. Raw predictions are clipped for decision analysis as in Methods. These repeated comparisons are dependent, and the reported SDs describe variation across the 45 evaluations rather than uncertainty from independent deployment trials.

| Design | Formulation | Pareto F1 (mean ± SD) | Profile matches / 225 (%) | Weight agreement, % (mean ± SD) | Mismatches on subset exact front (%) | Median mismatch regret | Maximum regret |
|---|---|---:|---:|---:|---:|---:|---:|
| all-predicted | Equally weighted joint network | 0.860 ± 0.028 | 88/225 (39.1) | 39.9 ± 15.0 | 95.6 | 0.0159 | 0.4719 |
| all-predicted | Gradient-balanced joint network | 0.854 ± 0.028 | 80/225 (35.6) | 38.1 ± 14.6 | 95.2 | 0.0122 | 0.4441 |
| all-predicted | Single-task networks | 0.865 ± 0.030 | 95/225 (42.2) | 42.1 ± 12.6 | 100.0 | 0.0116 | 0.1158 |
| all-predicted | Random forest | 0.882 ± 0.023 | 123/225 (54.7) | 52.2 ± 12.5 | 100.0 | 0.0083 | 0.0706 |
| all-predicted | Gradient-boosted trees | 0.918 ± 0.023 | 147/225 (65.3) | 72.1 ± 11.3 | 100.0 | 0.0053 | 0.1109 |
| exact-index | Equally weighted joint network | 0.942 ± 0.023 | 148/225 (65.8) | 75.4 ± 9.3 | 98.7 | 0.0033 | 0.1215 |
| exact-index | Gradient-balanced joint network | 0.941 ± 0.021 | 147/225 (65.3) | 73.7 ± 12.4 | 97.4 | 0.0039 | 0.4441 |
| exact-index | Single-task networks | 0.939 ± 0.024 | 143/225 (63.6) | 72.5 ± 15.2 | 100.0 | 0.0045 | 0.1158 |
| exact-index | Random forest | 0.934 ± 0.022 | 156/225 (69.3) | 70.8 ± 11.9 | 100.0 | 0.0045 | 0.0782 |
| exact-index | Gradient-boosted trees | 0.939 ± 0.025 | 160/225 (71.1) | 77.7 ± 9.6 | 100.0 | 0.0039 | 0.1109 |

Pooling the five exact-index formulations, 368 of 371 mismatches (99.2%) remain nondominated within their 125-candidate set; their median regret is 0.0042. Only 80.3% are also on the 625-package exact front. The subset and full-grid references therefore should not be interchanged.

## Table S1.22. Exact winner-to-runner-up score margins

Margins are the second-lowest minus the lowest weighted score among exact-front members, using each candidate set’s own exact-front ideal/nadir. The full-grid calculation counts 15 weather–profile decisions once each; the held-out calculation counts 225 repeat–fold–weather–profile decisions once each. Values are not duplicated across formulations or task designs.

| Candidate set | Profile | Decisions | Margin < 0.004 (%) | Median margin |
|---|---|---:|---:|---:|
| Full grid (625) | All profiles | 15 | 80.0 | 0.00171 |
| Full grid (625) | Equal weight | 3 | 100.0 | 0.00102 |
| Full grid (625) | GWP priority | 3 | 100.0 | 0.00179 |
| Full grid (625) | Cost priority | 3 | 0.0 | 0.04331 |
| Full grid (625) | Warm-side priority | 3 | 100.0 | 0.00185 |
| Full grid (625) | Heating priority | 3 | 100.0 | 0.00105 |
| Held-out (125) | All profiles | 225 | 37.8 | 0.00663 |
| Held-out (125) | Equal weight | 45 | 17.8 | 0.00663 |
| Held-out (125) | GWP priority | 45 | 20.0 | 0.01913 |
| Held-out (125) | Cost priority | 45 | 13.3 | 0.02087 |
| Held-out (125) | Warm-side priority | 45 | 51.1 | 0.00370 |
| Held-out (125) | Heating priority | 45 | 86.7 | 0.00191 |

The full-grid median is 0.001705 (0.0017 rounded), and the held-out-subset median is 0.006630. Among exact-index comparisons within held-out sets, identity agreement is 37.4% when the exact margin is below 0.004 and 85.0% otherwise. This is a descriptive association: reducing the candidate count also changes the front and its normalisation, so the comparison does not isolate a causal effect of score margins or cross-fitting. The diagnostic CSV also reports the runner-up among all candidates under the same exact-front scaling; this differs from the front-only runner-up in one of the 225 held-out decisions, without changing its classification at 0.004.

## Table S1.23. Regret tails for full-grid and held-out decisions

Values pool the five main formulations. All-profile rows include exact package agreements (zero regret); mismatch rows exclude them. Each formulation contributes 45 full-grid profile comparisons (3 repeats × 3 weather cases × 5 profiles) or 225 held-out profile comparisons (3 repeats × 5 folds × 3 weather cases × 5 profiles). The 95th percentile uses linear interpolation. Regret uses the exact-front scaling of the corresponding 625- or 125-candidate set, so magnitudes across candidate-set sizes use different anchors.

In full-grid reconstruction, pooling the five formulations within the exact-index
design gives 127 of 133 mismatches (95.5%) on the exact front and median mismatch
regret 0.0047. The all-predicted design gives 160 of 180 (88.9%) and median 0.0164.
Pooling across both task designs gives 287 of 313 (91.7%) and median 0.0078; these
pooled values combine the two designs rather than describe either one separately.

| Candidate set | Design | Population | Comparisons | Median | 95th percentile | Maximum | Regret > 0.05, count (%) |
|---|---|---|---:|---:|---:|---:|---:|
| Full grid (625) | all-predicted | All profiles | 225 | 0.0078 | 0.0696 | 0.4723 | 26 (11.6) |
| Full grid (625) | all-predicted | Mismatches | 180 | 0.0164 | 0.0939 | 0.4723 | 26 (14.4) |
| Full grid (625) | exact-index | All profiles | 225 | 0.0010 | 0.0400 | 0.1174 | 8 (3.6) |
| Full grid (625) | exact-index | Mismatches | 133 | 0.0047 | 0.0767 | 0.1174 | 8 (6.0) |
| Held-out (125) | all-predicted | All profiles | 1125 | 0.0008 | 0.0542 | 0.4719 | 72 (6.4) |
| Held-out (125) | all-predicted | Mismatches | 592 | 0.0111 | 0.0747 | 0.4719 | 72 (12.2) |
| Held-out (125) | exact-index | All profiles | 1125 | 0.0000 | 0.0209 | 0.4441 | 20 (1.8) |
| Held-out (125) | exact-index | Mismatches | 371 | 0.0042 | 0.0563 | 0.4441 | 20 (5.4) |

## Table S1.24. Physical outcomes of the largest exact-index held-out regret

Physical outcomes and weighted contributions for the largest held-out-subset regret under the exact-index design. The gradient-balanced joint network (repeat seed 43, fold 1) selects package 1 instead of package 145 under present weather and the warm-side priority profile. Positive contributions increase regret; index savings partially offset the heating and warm-side penalties.

| Quantity | Exact selection (145) | Surrogate selection (1) | Weighted contribution to regret |
|---|---:|---:|---:|
| Annual heating, $E_H$ (GJ/year) | 22.712 | 37.571 | +0.07406 |
| Cost index, $I_C$ (points) | 507.000 | 0.000 | -0.04734 |
| GWP index, $I_G$ (points) | 97.000 | 0.000 | -0.04932 |
| Warm-side count, $D_{24}$ (days) | 365.000 | 361.000 | +0.46667 |
| Total regret | | | 0.44406 |

The unrenovated package selected by the surrogate uses 14.86 GJ/year more heating and has four fewer days below 24 °C than the exact selection. The exact subset front spans six $D_{24}$ days, giving the warm-side contribution $0.7 × 4/6 = 0.46667$. Cost and GWP index savings partly offset that penalty. Both selected packages are nondominated within the subset and within the full grid. Raw $D_{24}$ predictions for the exact and surrogate selections are 365.68 and 371.77 days, respectively, and both enter the decision analysis at the clipped value of 365 days.

The retained exact-output checks in `physical_reference_checks.json` also confirm that the late-century façade/roof warm-side-priority package (ID 15) cuts heating by 30.02% and gains five $D_{24}$ days relative to the unrenovated package. Among the minimum-heating selections across the three weather cases, the late-century selection has the largest warm-side penalty (57 additional warm days). Across all 625 late-century packages, the largest penalty is 59 days (IDs 580, 604 and 605).

Tables S1.21–S1.24 are generated from retained outputs by `supplementary_analysis/within_fold_selection_check.py`. Machine-readable tables, case details, input SHA256 fingerprints and LaTeX tables are in `supplementary_analysis/decision_fidelity_diagnostics/`. No model fitting or EnergyPlus execution is performed.
