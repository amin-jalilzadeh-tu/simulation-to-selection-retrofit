"""Component A1--A3 GWP rates and the comparative GWP-rate index.

The values below are component-specific kg CO2e/m2 rates. The archived study
sums one selected rate per component without multiplying by component area.
The result is therefore a comparative index, not whole-building embodied GWP.
"""


def calculate_window_carbon(window_U_Factor):
    rates = {2.90: 0, 1.20: 70, 1.21: 50, 0.80: 150, 0.81: 120}
    try:
        return rates[window_U_Factor]
    except KeyError as exc:
        raise ValueError("windows_U_Factor not valid") from exc


def calculate_floor_carbon(groundfloor_thermal_resistance):
    rates = {0.41: 0, 4.8: 10, 5.0: 5.92, 5.5: 11, 5.6: 7}
    try:
        return rates[groundfloor_thermal_resistance]
    except KeyError as exc:
        raise ValueError("groundfloor_thermal_resistance not valid") from exc


def calculate_facade_carbon(ext_walls_thermal_resistance):
    rates = {0.45: 0, 4.2: 9.36, 4.4: 4.83, 6.5: 17.16, 6.7: 8.5}
    try:
        return rates[ext_walls_thermal_resistance]
    except KeyError as exc:
        raise ValueError("ext_walls_thermal_resistance not valid") from exc


def calculate_roof_carbon(roof_thermal_resistance):
    rates = {0.48: 0, 4.5: 23.29, 4.7: 4.76, 8.5: 18.5, 8.7: 10.68}
    try:
        return rates[roof_thermal_resistance]
    except KeyError as exc:
        raise ValueError("roof_thermal_resistance not valid") from exc


def calculate_gwp_rate_index(
    window_U_Factor,
    groundfloor_thermal_resistance,
    ext_walls_thermal_resistance,
    roof_thermal_resistance,
):
    """Return the unweighted sum of four component A1--A3 kg CO2e/m2 rates."""
    return (
        calculate_window_carbon(window_U_Factor)
        + calculate_floor_carbon(groundfloor_thermal_resistance)
        + calculate_facade_carbon(ext_walls_thermal_resistance)
        + calculate_roof_carbon(roof_thermal_resistance)
    )


def calculate_total_carbon(
    window_U_Factor,
    groundfloor_thermal_resistance,
    ext_walls_thermal_resistance,
    roof_thermal_resistance,
):
    """Backward-compatible alias for :func:`calculate_gwp_rate_index`.

    The alias is kept for saved pipelines. Its return value
    is not whole-building embodied GWP because envelope areas are not applied.
    """
    return calculate_gwp_rate_index(
        window_U_Factor,
        groundfloor_thermal_resistance,
        ext_walls_thermal_resistance,
        roof_thermal_resistance,
    )
