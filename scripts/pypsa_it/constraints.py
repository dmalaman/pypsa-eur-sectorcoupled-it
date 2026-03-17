# SPDX-FileCopyrightText: Contributors to PyPSA-Eur-SectorCoupled-IT
# SPDX-License-Identifier: MIT

"""
PyPSA-IT custom constraints added through solve_network.extra_functionality.

Current implementation supports:
- total installed capacity constraints for Generators by carrier (Italy only)
- total produced energy constraints for Generators by carrier (Italy only)
- total electricity exchange constraints between Italy and neighbouring countries
  using AC lines and DC links connecting electric buses

Conventions:
- capacity targets are in MW
- energy/exchange targets are in MWh
- exchange target sign is from Italy perspective:
    positive -> net export from Italy to neighbour
    negative -> net import into Italy from neighbour
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Validation
# =============================================================================


def validate_pypsait_config(pypsait_cfg: dict[str, Any]) -> None:
    """Validate the PyPSA-IT config structure."""
    if not isinstance(pypsait_cfg, dict):
        raise TypeError("pypsa_it config must be a dictionary.")

    allowed_senses = {"==", "<=", ">="}

    for section_name in [
        "capacity_constraints",
        "energy_constraints",
        "exchange_constraints",
    ]:
        section = pypsait_cfg.get(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(f"pypsa_it.{section_name} must be a dictionary.")

        if not section.get("enable", False):
            continue

        if section_name in {"capacity_constraints", "energy_constraints"}:
            for subgroup in ["generators", "hydrogen", "gas"]:
                values = section.get(subgroup, {})
                if values is None:
                    continue
                if not isinstance(values, dict):
                    raise TypeError(
                        f"pypsa_it.{section_name}.{subgroup} must be a dictionary."
                    )

                for key, spec in values.items():
                    _validate_compact_constraint_spec(
                        spec=spec,
                        location=f"pypsa_it.{section_name}.{subgroup}.{key}",
                        allowed_senses=allowed_senses,
                    )

        elif section_name == "exchange_constraints":
            electricity = section.get("electricity", {})
            if electricity is None:
                continue
            if not isinstance(electricity, dict):
                raise TypeError(
                    "pypsa_it.exchange_constraints.electricity must be a dictionary."
                )

            for country, spec in electricity.items():
                _validate_compact_constraint_spec(
                    spec=spec,
                    location=f"pypsa_it.exchange_constraints.electricity.{country}",
                    allowed_senses=allowed_senses,
                )


def _validate_compact_constraint_spec(
    spec: Any,
    location: str,
    allowed_senses: set[str],
) -> None:
    """Validate compact constraint syntax: ['==', 123.0]."""
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError(
            f"{location} must be a list/tuple of length 2 like ['==', 123.0]."
        )

    sense, value = spec

    if sense not in allowed_senses:
        raise ValueError(
            f"{location}: invalid sense {sense!r}. Allowed: {sorted(allowed_senses)}."
        )

    if not isinstance(value, (int, float)):
        raise TypeError(f"{location}: target value must be numeric, got {type(value)}.")


# =============================================================================
# Public entry point
# =============================================================================


def add_pypsait_constraints(n, snapshots: pd.Index, pypsait_cfg: dict[str, Any]) -> None:
    """Add all enabled PyPSA-IT constraints to the current optimization model."""
    validate_pypsait_config(pypsait_cfg)

    cap_cfg = pypsait_cfg.get("capacity_constraints", {})
    ene_cfg = pypsait_cfg.get("energy_constraints", {})
    exc_cfg = pypsait_cfg.get("exchange_constraints", {})

    if cap_cfg.get("enable", False):
        add_capacity_constraints(n, snapshots, cap_cfg)

    if ene_cfg.get("enable", False):
        add_energy_constraints(n, snapshots, ene_cfg)

    if exc_cfg.get("enable", False):
        add_exchange_constraints(n, snapshots, exc_cfg)


# =============================================================================
# Capacity constraints
# =============================================================================


def add_capacity_constraints(n, snapshots: pd.Index, cap_cfg: dict[str, Any]) -> None:
    """Add installed capacity constraints."""
    _ = snapshots  # not used for capacity constraints

    # -------------------------
    # Generator carriers
    # -------------------------
    generator_constraints = cap_cfg.get("generators", {}) or {}
    for carrier, (sense, target) in generator_constraints.items():
        assets = _get_generator_assets_by_carrier(n, carrier)
        lhs = _build_generator_capacity_expression(n, assets)

        name = f"pypsait_capacity_generators_{_sanitize_name(carrier)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT capacity constraint on generators for carrier '%s': %s %s MW",
            carrier,
            sense,
            target,
        )

    # -------------------------
    # Hydrogen links
    # -------------------------
    hydrogen_constraints = cap_cfg.get("hydrogen", {}) or {}
    for key, (sense, target) in hydrogen_constraints.items():
        assets = _get_hydrogen_link_assets(n, key)
        lhs = _build_link_capacity_expression(n, assets)

        name = f"pypsait_capacity_hydrogen_{_sanitize_name(key)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT capacity constraint on hydrogen links for key '%s': %s %s MW",
            key,
            sense,
            target,
        )

    # -------------------------
    # Gas links
    # -------------------------
    gas_constraints = cap_cfg.get("gas", {}) or {}
    for key, (sense, target) in gas_constraints.items():
        assets = _get_gas_link_assets(n, key)
        lhs = _build_link_capacity_expression(n, assets)

        name = f"pypsait_capacity_gas_{_sanitize_name(key)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT capacity constraint on gas links for key '%s': %s %s MW",
            key,
            sense,
            target,
        )


# =============================================================================
# Energy constraints
# =============================================================================


def add_energy_constraints(n, snapshots: pd.Index, ene_cfg: dict[str, Any]) -> None:
    """Add total energy constraints."""
    weightings = _get_snapshot_weightings(n, snapshots)

    # -------------------------
    # Generator carriers
    # -------------------------
    generator_constraints = ene_cfg.get("generators", {}) or {}
    for carrier, (sense, target) in generator_constraints.items():
        assets = _get_generator_assets_by_carrier(n, carrier)
        lhs = _build_generator_energy_expression(n, snapshots, assets, weightings)

        name = f"pypsait_energy_generators_{_sanitize_name(carrier)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT energy constraint on generators for carrier '%s': %s %s MWh",
            carrier,
            sense,
            target,
        )

    # -------------------------
    # Hydrogen links
    # -------------------------
    hydrogen_constraints = ene_cfg.get("hydrogen", {}) or {}
    for key, (sense, target) in hydrogen_constraints.items():
        assets = _get_hydrogen_link_assets(n, key)
        lhs = _build_link_energy_expression(n, snapshots, assets, weightings)

        name = f"pypsait_energy_hydrogen_{_sanitize_name(key)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT energy constraint on hydrogen links for key '%s': %s %s MWh",
            key,
            sense,
            target,
        )

    # -------------------------
    # Gas links
    # -------------------------
    gas_constraints = ene_cfg.get("gas", {}) or {}
    for key, (sense, target) in gas_constraints.items():
        assets = _get_gas_link_assets(n, key)
        lhs = _build_link_energy_expression(n, snapshots, assets, weightings)

        name = f"pypsait_energy_gas_{_sanitize_name(key)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT energy constraint on gas links for key '%s': %s %s MWh",
            key,
            sense,
            target,
        )


# =============================================================================
# Exchange constraints
# =============================================================================


def add_exchange_constraints(n, snapshots: pd.Index, exc_cfg: dict[str, Any]) -> None:
    """Add total electricity exchange constraints between Italy and neighbour countries."""
    weightings = _get_snapshot_weightings(n, snapshots)
    electricity_constraints = exc_cfg.get("electricity", {}) or {}

    for neighbour, (sense, target) in electricity_constraints.items():
        lhs = _build_electricity_exchange_expression(
            n=n,
            snapshots=snapshots,
            neighbour=neighbour,
            weightings=weightings,
        )

        name = f"pypsait_exchange_electricity_{_sanitize_name(neighbour)}"
        n.model.add_constraints(lhs, sense, float(target), name=name)

        logger.info(
            "Added PyPSA-IT electricity exchange constraint for IT-%s: %s %s MWh",
            neighbour,
            sense,
            target,
        )


# =============================================================================
# Selectors
# =============================================================================


def _get_generator_assets_by_carrier(n, carrier: str) -> pd.Index:
    """Select Italian generators by exact carrier name."""
    bus_country = n.generators.bus.map(n.buses.country)
    mask = (n.generators.carrier.astype(str) == str(carrier)) & (bus_country == "IT")
    assets = n.generators.index[mask]

    if assets.empty:
        raise ValueError(
            f"No Italian generators found for carrier {carrier!r} in PyPSA-IT constraint."
        )

    return assets


def _get_hydrogen_link_assets(n, key: str) -> pd.Index:
    """
    Select hydrogen-related Italian links for a semantic key.

    Both bus0 and bus1 must be in Italy.
    Extend this function case by case as needed.
    """
    carrier = str(key).strip().lower()

    bus0_country = n.links.bus0.map(n.buses.country)
    bus1_country = n.links.bus1.map(n.buses.country)
    is_internal_it = (bus0_country == "IT") & (bus1_country == "IT")

    if carrier == "electrolysis":
        carrier_mask = n.links.carrier.astype(str).str.lower().isin(
            {
                "h2 electrolysis",
                "h2 electrolyzer",
                "electrolysis",
            }
        )
    else:
        raise NotImplementedError(
            f"Hydrogen link selector for key {key!r} is not implemented yet."
        )

    assets = n.links.index[carrier_mask & is_internal_it]
    if assets.empty:
        raise ValueError(f"No Italian hydrogen links found for key {key!r}.")

    return assets


def _get_gas_link_assets(n, key: str) -> pd.Index:
    """
    Select gas-related Italian links for a semantic key.

    Both bus0 and bus1 must be in Italy.
    Extend this function case by case as needed.
    """
    k = str(key).strip().lower()

    bus0_country = n.links.bus0.map(n.buses.country)
    bus1_country = n.links.bus1.map(n.buses.country)
    is_internal_it = (bus0_country == "IT") & (bus1_country == "IT")

    carrier_series = n.links.carrier.astype(str).str.lower()

    if k == "imports":
        carrier_mask = carrier_series.str.contains("import", na=False)
    elif k == "biomethane":
        carrier_mask = carrier_series.str.contains("biomethane", na=False)
    elif k == "internal_production":
        carrier_mask = carrier_series.str.contains("gas production", na=False)
    else:
        raise NotImplementedError(
            f"Gas link selector for key {key!r} is not implemented yet."
        )

    assets = n.links.index[carrier_mask & is_internal_it]
    if assets.empty:
        raise ValueError(f"No Italian gas links found for key {key!r}.")

    return assets


def _select_crossborder_ac_lines(n, neighbour: str) -> tuple[pd.Index, pd.Series]:
    """
    Select AC lines connecting Italy and neighbour.

    Returns
    -------
    assets : pd.Index
        Selected line names.
    sign_from_italy : pd.Series
        +1 if positive p0 means export from Italy
        -1 if positive p0 means import into Italy
    """
    if n.lines.empty:
        return pd.Index([]), pd.Series(dtype=float)

    bus0_country = n.lines.bus0.map(n.buses.country)
    bus1_country = n.lines.bus1.map(n.buses.country)

    mask = ((bus0_country == "IT") & (bus1_country == neighbour)) | (
        (bus0_country == neighbour) & (bus1_country == "IT")
    )

    assets = n.lines.index[mask]
    if assets.empty:
        return assets, pd.Series(dtype=float)

    sign_from_italy = pd.Series(index=assets, dtype=float)
    sign_from_italy.loc[bus0_country.loc[assets] == "IT"] = 1.0
    sign_from_italy.loc[bus1_country.loc[assets] == "IT"] = -1.0

    return assets, sign_from_italy


def _select_crossborder_dc_links(n, neighbour: str) -> tuple[pd.Index, pd.Series]:
    """
    Select DC links connecting Italy and neighbour through electric buses.

    Returns
    -------
    assets : pd.Index
        Selected link names.
    sign_from_italy : pd.Series
        +1 if positive p0 means export from Italy
        -1 if positive p0 means import into Italy
    """
    if n.links.empty:
        return pd.Index([]), pd.Series(dtype=float)

    carrier = n.links.carrier.astype(str).str.lower()
    is_dc = carrier == "dc"

    bus0_country = n.links.bus0.map(n.buses.country)
    bus1_country = n.links.bus1.map(n.buses.country)

    bus0_carrier = n.links.bus0.map(n.buses.carrier).astype(str).str.lower()
    bus1_carrier = n.links.bus1.map(n.buses.carrier).astype(str).str.lower()

    is_electric = bus0_carrier.eq("ac") & bus1_carrier.eq("ac")

    mask = is_dc & is_electric & (
        ((bus0_country == "IT") & (bus1_country == neighbour))
        | ((bus0_country == neighbour) & (bus1_country == "IT"))
    )

    assets = n.links.index[mask]
    if assets.empty:
        return assets, pd.Series(dtype=float)

    sign_from_italy = pd.Series(index=assets, dtype=float)
    sign_from_italy.loc[bus0_country.loc[assets] == "IT"] = 1.0
    sign_from_italy.loc[bus1_country.loc[assets] == "IT"] = -1.0

    return assets, sign_from_italy


# =============================================================================
# Expression builders
# =============================================================================


def _build_generator_capacity_expression(n, assets: pd.Index):
    """Build sum of optimized generator capacities."""
    extendable = assets.intersection(n.generators.index[n.generators.p_nom_extendable])

    if extendable.empty:
        raise ValueError(
            "Selected generators for capacity constraint are not extendable. "
            "Current implementation only supports p_nom_opt decision variables."
        )

    if len(extendable) != len(assets):
        missing = list(set(assets) - set(extendable))
        raise ValueError(
            "Some selected generators are not extendable, so a p_nom_opt aggregate "
            f"constraint is ambiguous. Non-extendable assets: {missing}"
        )

    return n.model["Generator-p_nom"].loc[extendable].sum()


def _build_link_capacity_expression(n, assets: pd.Index):
    """Build sum of optimized link capacities."""
    extendable = assets.intersection(n.links.index[n.links.p_nom_extendable])

    if extendable.empty:
        raise ValueError(
            "Selected links for capacity constraint are not extendable. "
            "Current implementation only supports p_nom_opt decision variables."
        )

    if len(extendable) != len(assets):
        missing = list(set(assets) - set(extendable))
        raise ValueError(
            "Some selected links are not extendable, so a p_nom_opt aggregate "
            f"constraint is ambiguous. Non-extendable assets: {missing}"
        )

    return n.model["Link-p_nom"].loc[extendable].sum()


def _build_generator_energy_expression(
    n,
    snapshots: pd.Index,
    assets: pd.Index,
    weightings: pd.Series,
):
    """Build weighted total generator production."""
    dispatch = n.model["Generator-p"].loc[snapshots, assets]
    return (dispatch * weightings).sum()


def _build_link_energy_expression(
    n,
    snapshots: pd.Index,
    assets: pd.Index,
    weightings: pd.Series,
):
    """
    Build weighted total link output based on Link-p at bus0 side.

    Positive Link-p is flow from bus0 to bus1.
    For now this function sums raw p0-side flow. This is fine only for cases
    where the sign convention of the selected links is well understood.
    """
    dispatch = n.model["Link-p"].loc[snapshots, assets]
    return (dispatch * weightings).sum()


def _build_electricity_exchange_expression(
    n,
    snapshots: pd.Index,
    neighbour: str,
    weightings: pd.Series,
):
    """
    Build weighted net electricity exchange between Italy and neighbour.

    Positive expression means net export from Italy to neighbour.
    """
    lhs_terms = []

    # AC lines
    ac_assets, ac_sign = _select_crossborder_ac_lines(n, neighbour)
    if not ac_assets.empty:
        p0 = n.model["Line-s"].loc[snapshots, ac_assets]
        lhs_terms.append((p0 * ac_sign * weightings).sum())

    # DC links
    dc_assets, dc_sign = _select_crossborder_dc_links(n, neighbour)
    if not dc_assets.empty:
        p0 = n.model["Link-p"].loc[snapshots, dc_assets]
        lhs_terms.append((p0 * dc_sign * weightings).sum())

    if not lhs_terms:
        raise ValueError(
            f"No AC lines or DC links found between IT and {neighbour!r} "
            "for electricity exchange constraint."
        )

    lhs = lhs_terms[0]
    for term in lhs_terms[1:]:
        lhs = lhs + term

    return lhs


# =============================================================================
# Helpers
# =============================================================================


def _get_snapshot_weightings(n, snapshots: pd.Index) -> pd.Series:
    """
    Return snapshot weightings to convert power to energy.

    Prefer generators weighting, which is the standard choice for dispatch-based
    aggregate energy calculations in solve_network.
    """
    if "generators" in n.snapshot_weightings.columns:
        return n.snapshot_weightings.loc[snapshots, "generators"]

    return pd.Series(1.0, index=snapshots)


def _sanitize_name(value: str) -> str:
    """Sanitize a string for model constraint names."""
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )