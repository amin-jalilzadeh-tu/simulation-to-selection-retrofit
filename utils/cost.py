"""Component unit-cost rates and the manuscript's comparative cost index.

The values below are rates in EUR/m2 for four different envelope components.
The archived study sums one selected rate per component without multiplying by
the corresponding component area. The result is therefore a comparative
component-rate index, not a project cost in euros.
"""


def calculate_window_cost(window_U_Factor):
    rates = {2.90: 0, 1.20: 184, 1.21: 485, 0.80: 295, 0.81: 622}
    try:
        return rates[window_U_Factor]
    except KeyError as exc:
        raise ValueError("Value of windows_U_Factor not valid") from exc


def calculate_floor_cost(groundfloor_thermal_resistance):
    rates = {0.41: 0, 4.8: 59.7, 5.0: 77, 5.5: 87.9, 5.6: 108}
    try:
        return rates[groundfloor_thermal_resistance]
    except KeyError as exc:
        raise ValueError(
            "Value of groundfloor_thermal_resistance not valid"
        ) from exc


def calculate_facade_cost(ext_walls_thermal_resistance):
    rates = {0.45: 0, 4.2: 182, 4.4: 179, 6.5: 200, 6.7: 222}
    try:
        return rates[ext_walls_thermal_resistance]
    except KeyError as exc:
        raise ValueError(
            "Value of ext_walls_thermal_resistance not valid"
        ) from exc


def calculate_roof_cost(roof_thermal_resistance):
    rates = {0.48: 0, 4.5: 89.5, 4.7: 105, 8.5: 101, 8.7: 139}
    try:
        return rates[roof_thermal_resistance]
    except KeyError as exc:
        raise ValueError("Value of roof_thermal_resistance not valid") from exc


def calculate_cost_rate_index(
    window_U_Factor,
    groundfloor_thermal_resistance,
    ext_walls_thermal_resistance,
    roof_thermal_resistance,
):
    """Return the unweighted sum of four component-specific EUR/m2 rates."""
    return (
        calculate_window_cost(window_U_Factor)
        + calculate_floor_cost(groundfloor_thermal_resistance)
        + calculate_facade_cost(ext_walls_thermal_resistance)
        + calculate_roof_cost(roof_thermal_resistance)
    )


def calculate_total_cost(
    window_U_Factor,
    groundfloor_thermal_resistance,
    ext_walls_thermal_resistance,
    roof_thermal_resistance,
):
    """Backward-compatible alias for :func:`calculate_cost_rate_index`.

    The alias is kept for saved pipelines. Its return value
    is not a whole-building cost because envelope areas are not applied.
    """
    return calculate_cost_rate_index(
        window_U_Factor,
        groundfloor_thermal_resistance,
        ext_walls_thermal_resistance,
        roof_thermal_resistance,
    )
