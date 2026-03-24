#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare an Italian-focused PyPSA-Eur network against DE-IT electricity balance data.

Main features
-------------
- Italian scope only
- Generators counted only if connected to Italian buses
- Links counted only if they inject/withdraw on Italian buses
- Cross-border electricity imports counted only for branches with one Italian side
- Storage losses computed for StorageUnits and configured storage links
- Extra diagnostic rows for:
    * H2-to-power electricity production
    * Hydrogen production by electrolysis
    * Gas imports

Outputs
-------
- CSV summary
- Detailed CSV tables
- Comparison plots

Important
---------
This script is intentionally configurable. Review CONFIG carefully before using
results in a report or publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa


# =============================================================================
# CONFIG
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # Reference values from DE-IT (TWh)
    # Use np.nan for rows where you want only the PyPSA value.
    # -------------------------------------------------------------------------
    "reference_twh": {
        "Fabbisogno elettrico totale": 439.0,
        "Produzione nazionale": 400.0,
        "Produzione rinnovabile (RES)": 336.0,
        "Idroelettrico": 46.0,
        "Solare": 168.0,
        "Eolico": 121.0,
        "Altre RES": 17.0,
        "Sovragenerazione (curtailment)": -16.0,
        "Produzione termoelettrica (convenzionale)": 65.0,
        "Gas naturale": 59.0,
        "Altra non rinnovabile": 6.0,
        "Saldo estero (import-export)": 47.0,
        "Perdite di accumulo": -9.0,
        "Produzione elettrica da H2-to-power": np.nan,
        "Produzione H2 da elettrolizzatori": np.nan,
        "Import gas": np.nan,
    },

    # -------------------------------------------------------------------------
    # Italy detection
    # -------------------------------------------------------------------------
    "italy_country_codes": {"IT"},
    "italy_bus_prefixes": ("IT",),

    # -------------------------------------------------------------------------
    # Electric bus carriers
    # -------------------------------------------------------------------------
    "electric_bus_carriers": {"AC", "DC", "electricity", "low voltage"},

    # -------------------------------------------------------------------------
    # Gas / hydrogen bus carrier hints for imports and electrolysis
    # These may need to be adapted to your exact network.
    # -------------------------------------------------------------------------
    "gas_bus_carriers": {"gas"},
    "hydrogen_bus_carriers": {"H2", "hydrogen"},

    # -------------------------------------------------------------------------
    # Options
    # -------------------------------------------------------------------------
    "include_phs_in_hydro": True,
    "prefer_optimized_nominal": True,

    # -------------------------------------------------------------------------
    # Main category mapping
    # -------------------------------------------------------------------------
    "category_map": {
        "Idroelettrico": {
            "generator_carriers": {"ror"},
            "link_output_carriers": set(),
            "storageunit_dispatch_carriers": {"hydro", "PHS"},
        },
        "Solare": {
            "generator_carriers": {"solar", "solar-hsat", "solar rooftop"},
            "link_output_carriers": set(),
            "storageunit_dispatch_carriers": set(),
        },
        "Eolico": {
            "generator_carriers": {"onwind", "offwind-ac", "offwind-dc", "offwind-float"},
            "link_output_carriers": set(),
            "storageunit_dispatch_carriers": set(),
        },
        "Altre RES": {
            "generator_carriers": {"biogas", "solid biomass"},
            "link_output_carriers": {
                "urban central solid biomass CHP",
                "urban central solid biomass CHP CC",
            },
            "storageunit_dispatch_carriers": set(),
        },
        "Gas naturale": {
            "generator_carriers": {"gas"},
            "link_output_carriers": {
                "OCGT",
                "CCGT",
                "urban central gas CHP",
                "urban central gas CHP CC",
            },
            "storageunit_dispatch_carriers": set(),
        },
        "Altra non rinnovabile": {
            "generator_carriers": {"nuclear", "oil primary"},
            "link_output_carriers": set(),
            "storageunit_dispatch_carriers": set(),
        },
        "Produzione elettrica da H2-to-power": {
            "generator_carriers": set(),
            "link_output_carriers": {"H2 turbine", "H2 Fuel Cell"},
            "storageunit_dispatch_carriers": set(),
        },
    },

    # -------------------------------------------------------------------------
    # Generator carriers used for curtailment
    # -------------------------------------------------------------------------
    "curtailment_generator_carriers": {
        "solar",
        "solar-hsat",
        "solar rooftop",
        "onwind",
        "offwind-ac",
        "offwind-dc",
        "offwind-float",
        "ror",
    },

    # -------------------------------------------------------------------------
    # Link-based storage losses
    # losses = electric charge input - electric discharge output
    # -------------------------------------------------------------------------
    "link_storage_groups": [
        {
            "name": "battery",
            "charge_link_carriers": {"battery charger", "home battery charger"},
            "discharge_link_carriers": {"battery discharger", "home battery discharger"},
        },
        {
            "name": "hydrogen_power_storage",
            "charge_link_carriers": {"H2 Electrolysis"},
            "discharge_link_carriers": {"H2 Fuel Cell", "H2 turbine"},
        },
    ],

    # -------------------------------------------------------------------------
    # Electrolysis carriers for H2 production
    # -------------------------------------------------------------------------
    "electrolysis_link_carriers": {"H2 Electrolysis"},

    # -------------------------------------------------------------------------
    # Gas import links
    # -------------------------------------------------------------------------
    "gas_import_link_carriers": {"gas pipeline", "gas pipeline new"},
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def normalize_carrier(x) -> str:
    """Return a normalized carrier string."""
    if pd.isna(x):
        return ""
    return str(x).strip()


def find_weight_series(n: pypsa.Network, preferred: Sequence[str]) -> pd.Series:
    """Return a suitable snapshot weighting series."""
    sw = n.snapshot_weightings.copy()

    if isinstance(sw, pd.Series):
        return sw.astype(float)

    for col in preferred:
        if col in sw.columns:
            return sw[col].astype(float)

    return sw.iloc[:, 0].astype(float)


def weighted_sum_twh(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute annual energy in TWh from MW time series."""
    aligned_weights = weights.reindex(df.index)
    return df.mul(aligned_weights, axis=0).sum(axis=0) / 1e6


def get_active_nominal(df: pd.DataFrame, base_col: str, opt_col: str, prefer_opt: bool = True) -> pd.Series:
    """Return active nominal values."""
    if prefer_opt and opt_col in df.columns:
        out = df[opt_col].copy()
        if base_col in df.columns:
            out = out.fillna(df[base_col])
        return out.astype(float)
    return df[base_col].astype(float)


def get_bus_carrier(n: pypsa.Network) -> pd.Series:
    """Return bus carrier series."""
    if "carrier" in n.buses.columns:
        return n.buses["carrier"].astype(str)
    return pd.Series("", index=n.buses.index, dtype=object)


def detect_italian_buses(n: pypsa.Network, config: dict) -> pd.Index:
    """Detect Italian buses."""
    buses = n.buses.copy()
    mask = pd.Series(False, index=buses.index)

    if "country" in buses.columns:
        mask |= buses["country"].astype(str).isin(config["italy_country_codes"])

    if "location" in buses.columns:
        mask |= buses["location"].astype(str).str.startswith(config["italy_bus_prefixes"], na=False)

    mask |= buses.index.to_series().astype(str).str.startswith(config["italy_bus_prefixes"], na=False)

    return buses.index[mask]


def is_bus_of_type(bus: str, bus_carrier: pd.Series, allowed_carriers: Set[str]) -> bool:
    """Return True if bus has a carrier in allowed_carriers."""
    if bus not in bus_carrier.index:
        return False
    return normalize_carrier(bus_carrier.loc[bus]) in allowed_carriers


def get_component_timeseries(n: pypsa.Network, component: str, field: str) -> Optional[pd.DataFrame]:
    """Safely get component time series."""
    container_name = f"{component}_t"
    if not hasattr(n, container_name):
        return None
    container = getattr(n, container_name)
    if not hasattr(container, field):
        return None
    return getattr(container, field)


def get_link_terminal_bus(row: pd.Series, terminal: int) -> Optional[str]:
    """Return the bus name at a given link terminal."""
    col = f"bus{terminal}"
    if col in row.index:
        val = row[col]
        if isinstance(val, str) and val != "":
            return val
    return None


def get_link_terminal_power_df(n: pypsa.Network, terminal: int) -> Optional[pd.DataFrame]:
    """Return links_t.p{terminal} if available."""
    return get_component_timeseries(n, "links", f"p{terminal}")


def group_sum(series: pd.Series, carriers: pd.Series, carrier_set: Set[str]) -> float:
    """Sum series entries whose carrier belongs to carrier_set."""
    if len(series) == 0:
        return 0.0
    mask = carriers.isin(carrier_set)
    if mask.sum() == 0:
        return 0.0
    return float(series.loc[mask].sum())


# =============================================================================
# LOADS
# =============================================================================

def get_load_dispatch(n: pypsa.Network) -> pd.DataFrame:
    """Return load time series in MW."""
    p = get_component_timeseries(n, "loads", "p")
    if p is not None and not p.empty:
        return p

    p_set = get_component_timeseries(n, "loads", "p_set")
    if p_set is not None and not p_set.empty:
        return p_set

    if "p_set" in n.loads.columns:
        static = n.loads["p_set"].astype(float)
        return pd.DataFrame(
            np.repeat(static.values.reshape(1, -1), len(n.snapshots), axis=0),
            index=n.snapshots,
            columns=n.loads.index,
        )

    raise ValueError("No load time series found.")


# =============================================================================
# LINK INPUT / OUTPUT ON ITALIAN ELECTRIC BUSES
# =============================================================================

def collect_link_electric_output(
    n: pypsa.Network,
    italian_buses: Set[str],
    electric_bus_carriers: Set[str],
    bus_carrier: pd.Series,
) -> pd.Series:
    """
    Annual electricity injection from each link into Italian electric buses.

    At a terminal:
    injection into bus = max(-p_terminal, 0)
    """
    weights = find_weight_series(n, ["objective", "generators", "stores"])
    annual_output = pd.Series(0.0, index=n.links.index, dtype=float)

    for t in range(5):
        p_df = get_link_terminal_power_df(n, t)
        if p_df is None or p_df.empty:
            continue

        cols = p_df.columns.intersection(n.links.index)
        if len(cols) == 0:
            continue

        buses_t = n.links.loc[cols, f"bus{t}"]
        mask = buses_t.isin(italian_buses) & buses_t.map(
            lambda b: is_bus_of_type(b, bus_carrier, electric_bus_carriers)
        )
        if not mask.any():
            continue

        selected = mask.index[mask]
        inj = (-p_df[selected]).clip(lower=0.0)
        annual_output.loc[selected] += weighted_sum_twh(inj, weights)

    return annual_output


def collect_link_electric_input(
    n: pypsa.Network,
    italian_buses: Set[str],
    electric_bus_carriers: Set[str],
    bus_carrier: pd.Series,
) -> pd.Series:
    """
    Annual electricity withdrawal by each link from Italian electric buses.

    At a terminal:
    withdrawal from bus = max(p_terminal, 0)
    """
    weights = find_weight_series(n, ["objective", "generators", "stores"])
    annual_input = pd.Series(0.0, index=n.links.index, dtype=float)

    for t in range(5):
        p_df = get_link_terminal_power_df(n, t)
        if p_df is None or p_df.empty:
            continue

        cols = p_df.columns.intersection(n.links.index)
        if len(cols) == 0:
            continue

        buses_t = n.links.loc[cols, f"bus{t}"]
        mask = buses_t.isin(italian_buses) & buses_t.map(
            lambda b: is_bus_of_type(b, bus_carrier, electric_bus_carriers)
        )
        if not mask.any():
            continue

        selected = mask.index[mask]
        absorp = p_df[selected].clip(lower=0.0)
        annual_input.loc[selected] += weighted_sum_twh(absorp, weights)

    return annual_input


# =============================================================================
# GENERATOR CURTAILMENT
# =============================================================================

def compute_generator_curtailment_twh(
    n: pypsa.Network,
    italian_generators: pd.Index,
    curtailment_carriers: Set[str],
    prefer_opt: bool = True,
) -> pd.Series:
    """Compute annual generator curtailment in TWh."""
    gens = n.generators.loc[italian_generators].copy()
    carriers = gens["carrier"].map(normalize_carrier)
    selected = carriers[carriers.isin(curtailment_carriers)].index

    if len(selected) == 0:
        return pd.Series(dtype=float)

    weights = find_weight_series(n, ["generators", "objective", "stores"])
    p = n.generators_t.p[selected]
    p_max_pu = n.generators_t.p_max_pu[selected]

    p_nom_active = get_active_nominal(
        gens.loc[selected],
        base_col="p_nom",
        opt_col="p_nom_opt",
        prefer_opt=prefer_opt,
    )

    available = p_max_pu.mul(p_nom_active, axis=1)
    curtailed = (available - p).clip(lower=0.0)

    return weighted_sum_twh(curtailed, weights)


# =============================================================================
# STORAGEUNITS
# =============================================================================

def compute_storageunit_dispatch_twh(
    n: pypsa.Network,
    italian_storageunits: pd.Index,
) -> pd.Series:
    """Return annual StorageUnit discharge in TWh."""
    if len(italian_storageunits) == 0:
        return pd.Series(dtype=float)

    if not hasattr(n.storage_units_t, "p_dispatch") or n.storage_units_t.p_dispatch.empty:
        return pd.Series(0.0, index=italian_storageunits)

    weights = find_weight_series(n, ["stores", "objective", "generators"])
    return weighted_sum_twh(n.storage_units_t.p_dispatch[italian_storageunits], weights)


def compute_storageunit_losses_twh(
    n: pypsa.Network,
    italian_storageunits: pd.Index,
) -> float:
    """Compute StorageUnit losses in TWh."""
    if len(italian_storageunits) == 0:
        return 0.0

    weights = find_weight_series(n, ["stores", "objective", "generators"])

    charge = 0.0
    discharge = 0.0

    if hasattr(n.storage_units_t, "p_store") and not n.storage_units_t.p_store.empty:
        charge = float(weighted_sum_twh(n.storage_units_t.p_store[italian_storageunits], weights).sum())

    if hasattr(n.storage_units_t, "p_dispatch") and not n.storage_units_t.p_dispatch.empty:
        discharge = float(weighted_sum_twh(n.storage_units_t.p_dispatch[italian_storageunits], weights).sum())

    return max(charge - discharge, 0.0)


# =============================================================================
# LINK STORAGE LOSSES
# =============================================================================

def compute_link_storage_losses_twh(
    n: pypsa.Network,
    annual_link_input_twh: pd.Series,
    annual_link_output_twh: pd.Series,
    config: dict,
) -> Tuple[float, pd.DataFrame]:
    """Compute losses for configured link-based storage groups."""
    link_carriers = n.links["carrier"].map(normalize_carrier)
    records = []
    total_loss = 0.0

    for group in config["link_storage_groups"]:
        charge = float(annual_link_input_twh.loc[link_carriers.isin(group["charge_link_carriers"])].sum())
        discharge = float(annual_link_output_twh.loc[link_carriers.isin(group["discharge_link_carriers"])].sum())
        loss = max(charge - discharge, 0.0)

        total_loss += loss
        records.append({
            "group": group["name"],
            "charge_twh": charge,
            "discharge_twh": discharge,
            "loss_twh": loss,
        })

    return total_loss, pd.DataFrame(records)


# =============================================================================
# BORDER ELECTRICITY IMPORTS
# =============================================================================

def compute_border_net_imports_twh(
    n: pypsa.Network,
    italian_buses: Set[str],
    electric_bus_carriers: Set[str],
    bus_carrier: pd.Series,
) -> Tuple[float, pd.DataFrame]:
    """
    Compute net electricity imports into Italy in TWh.

    Positive means net imports into Italy.
    """
    weights = find_weight_series(n, ["objective", "generators", "stores"])
    records = []

    # Lines
    if not n.lines.empty and hasattr(n.lines_t, "p0") and hasattr(n.lines_t, "p1"):
        for name, row in n.lines.iterrows():
            bus0 = row["bus0"]
            bus1 = row["bus1"]

            bus0_it = (bus0 in italian_buses) and is_bus_of_type(bus0, bus_carrier, electric_bus_carriers)
            bus1_it = (bus1 in italian_buses) and is_bus_of_type(bus1, bus_carrier, electric_bus_carriers)

            if bus0_it == bus1_it:
                continue

            if bus0_it:
                flow_import = -n.lines_t.p0[name]
                italy_bus = bus0
                foreign_bus = bus1
                italy_side = "bus0"
            else:
                flow_import = -n.lines_t.p1[name]
                italy_bus = bus1
                foreign_bus = bus0
                italy_side = "bus1"

            annual_twh = float((flow_import * weights.reindex(flow_import.index)).sum() / 1e6)

            records.append({
                "component": "line",
                "name": name,
                "carrier": normalize_carrier(row.get("carrier", "")),
                "italy_side": italy_side,
                "italy_bus": italy_bus,
                "foreign_bus": foreign_bus,
                "net_import_twh": annual_twh,
            })

    # Links
    if not n.links.empty:
        for name, row in n.links.iterrows():
            terminals = []
            for t in range(5):
                bus = get_link_terminal_bus(row, t)
                if bus is not None:
                    terminals.append((t, bus))

            italian_terminals = [
                (t, b) for t, b in terminals
                if (b in italian_buses) and is_bus_of_type(b, bus_carrier, electric_bus_carriers)
            ]
            foreign_terminals = [
                (t, b) for t, b in terminals
                if (b not in italian_buses) and is_bus_of_type(b, bus_carrier, electric_bus_carriers)
            ]

            if len(italian_terminals) != 1 or len(foreign_terminals) < 1:
                continue

            t_it, italy_bus = italian_terminals[0]
            p_df = get_link_terminal_power_df(n, t_it)
            if p_df is None or name not in p_df.columns:
                continue

            flow_import = -p_df[name]
            annual_twh = float((flow_import * weights.reindex(flow_import.index)).sum() / 1e6)

            records.append({
                "component": "link",
                "name": name,
                "carrier": normalize_carrier(row.get("carrier", "")),
                "italy_side": f"bus{t_it}",
                "italy_bus": italy_bus,
                "foreign_bus": ",".join(b for _, b in foreign_terminals),
                "net_import_twh": annual_twh,
            })

    details = pd.DataFrame(records)
    total = float(details["net_import_twh"].sum()) if not details.empty else 0.0
    return total, details


# =============================================================================
# H2 PRODUCTION FROM ELECTROLYSIS
# =============================================================================

def compute_h2_production_from_electrolysis_twh(
    n: pypsa.Network,
    italian_buses: Set[str],
    hydrogen_bus_carriers: Set[str],
    bus_carrier: pd.Series,
    electrolysis_link_carriers: Set[str],
) -> Tuple[float, pd.DataFrame]:
    """
    Compute hydrogen production from electrolysis in TWh_H2.

    We count the hydrogen output side of electrolysis links on Italian H2 buses.
    """
    weights = find_weight_series(n, ["objective", "generators", "stores"])
    link_carriers = n.links["carrier"].map(normalize_carrier)

    selected_links = link_carriers[link_carriers.isin(electrolysis_link_carriers)].index
    records = []

    total_twh = 0.0

    for name in selected_links:
        row = n.links.loc[name]

        # Look for hydrogen-output terminal(s)
        for t in range(5):
            bus = get_link_terminal_bus(row, t)
            if bus is None:
                continue

            if (bus in italian_buses) and is_bus_of_type(bus, bus_carrier, hydrogen_bus_carriers):
                p_df = get_link_terminal_power_df(n, t)
                if p_df is None or name not in p_df.columns:
                    continue

                # Injection into H2 bus
                output = (-p_df[name]).clip(lower=0.0)
                annual_twh = float((output * weights.reindex(output.index)).sum() / 1e6)

                total_twh += annual_twh
                records.append({
                    "link": name,
                    "carrier": normalize_carrier(row.get("carrier", "")),
                    "h2_bus": bus,
                    "annual_h2_output_twh": annual_twh,
                })

    return total_twh, pd.DataFrame(records)


# =============================================================================
# GAS IMPORTS
# =============================================================================

def compute_gas_imports_twh(
    n: pypsa.Network,
    italian_buses: Set[str],
    gas_bus_carriers: Set[str],
    bus_carrier: pd.Series,
    gas_import_link_carriers: Set[str],
) -> Tuple[float, pd.DataFrame]:
    """
    Compute gas imports into Italy in TWh from gas pipeline links.

    Positive means net gas imports into Italy.
    """
    weights = find_weight_series(n, ["objective", "generators", "stores"])
    link_carriers = n.links["carrier"].map(normalize_carrier)

    selected_links = link_carriers[link_carriers.isin(gas_import_link_carriers)].index
    records = []

    total_twh = 0.0

    for name in selected_links:
        row = n.links.loc[name]

        terminals = []
        for t in range(5):
            bus = get_link_terminal_bus(row, t)
            if bus is not None:
                terminals.append((t, bus))

        italian_gas_terminals = [
            (t, b) for t, b in terminals
            if (b in italian_buses) and is_bus_of_type(b, bus_carrier, gas_bus_carriers)
        ]
        foreign_gas_terminals = [
            (t, b) for t, b in terminals
            if (b not in italian_buses) and is_bus_of_type(b, bus_carrier, gas_bus_carriers)
        ]

        if len(italian_gas_terminals) != 1 or len(foreign_gas_terminals) < 1:
            continue

        t_it, italy_bus = italian_gas_terminals[0]
        p_df = get_link_terminal_power_df(n, t_it)
        if p_df is None or name not in p_df.columns:
            continue

        flow_import = -p_df[name]
        annual_twh = float((flow_import * weights.reindex(flow_import.index)).sum() / 1e6)

        total_twh += annual_twh
        records.append({
            "link": name,
            "carrier": normalize_carrier(row.get("carrier", "")),
            "italy_bus": italy_bus,
            "foreign_bus": ",".join(b for _, b in foreign_gas_terminals),
            "annual_import_twh": annual_twh,
        })

    return total_twh, pd.DataFrame(records)


# =============================================================================
# CATEGORY AGGREGATION
# =============================================================================

def compute_category_totals(
    n: pypsa.Network,
    italian_generators: pd.Index,
    italian_storageunits: pd.Index,
    gen_dispatch_twh: pd.Series,
    su_dispatch_twh: pd.Series,
    annual_link_output_twh: pd.Series,
    config: dict,
) -> Tuple[Dict[str, float], Dict[str, pd.DataFrame]]:
    """
    Aggregate category totals from generators, storage units, and links.
    """
    gen_carriers = n.generators.loc[italian_generators, "carrier"].map(normalize_carrier)
    link_carriers = n.links["carrier"].map(normalize_carrier)

    su_carriers = pd.Series(dtype=object)
    if len(italian_storageunits) > 0:
        su_carriers = n.storage_units.loc[italian_storageunits, "carrier"].map(normalize_carrier)

    totals = {}
    details = {}

    include_phs = config["include_phs_in_hydro"]

    for category, mapping in config["category_map"].items():
        gen_set = set(mapping["generator_carriers"])
        link_set = set(mapping["link_output_carriers"])
        su_set = set(mapping["storageunit_dispatch_carriers"])

        if category == "Idroelettrico" and not include_phs:
            su_set = su_set - {"PHS"}

        gen_val = group_sum(gen_dispatch_twh, gen_carriers, gen_set)
        link_val = group_sum(annual_link_output_twh, link_carriers, link_set)

        su_val = 0.0
        if len(su_dispatch_twh) > 0 and len(su_set) > 0:
            su_val = group_sum(su_dispatch_twh, su_carriers, su_set)

        totals[category] = gen_val + link_val + su_val

        details[category] = pd.DataFrame({
            "source": ["generators", "links", "storage_units"],
            "value_twh": [gen_val, link_val, su_val],
        })

    return totals, details


# =============================================================================
# MAIN BALANCE
# =============================================================================

def compute_balance(n: pypsa.Network, config: dict) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compute the Italian balance and diagnostic tables."""
    italian_buses = detect_italian_buses(n, config)
    italian_buses_set = set(italian_buses)
    bus_carrier = get_bus_carrier(n)

    electric_bus_carriers = set(config["electric_bus_carriers"])
    gas_bus_carriers = set(config["gas_bus_carriers"])
    hydrogen_bus_carriers = set(config["hydrogen_bus_carriers"])

    # -------------------------------------------------------------------------
    # Loads
    # -------------------------------------------------------------------------
    load_weights = find_weight_series(n, ["objective", "generators", "stores"])
    italian_loads = n.loads.index[n.loads["bus"].isin(italian_buses)]
    loads_dispatch = get_load_dispatch(n)
    load_twh = weighted_sum_twh(loads_dispatch[italian_loads], load_weights)
    total_demand_twh = float(load_twh.sum())

    load_details = pd.DataFrame({
        "bus": n.loads.loc[italian_loads, "bus"],
        "carrier": n.loads.loc[italian_loads, "carrier"].map(normalize_carrier) if "carrier" in n.loads.columns else "",
        "annual_twh": load_twh,
    }).sort_values("annual_twh", ascending=False)

    # -------------------------------------------------------------------------
    # Generators
    # -------------------------------------------------------------------------
    gen_weights = find_weight_series(n, ["generators", "objective", "stores"])
    italian_generators = n.generators.index[n.generators["bus"].isin(italian_buses)]
    gen_dispatch_twh = weighted_sum_twh(n.generators_t.p[italian_generators], gen_weights)

    generator_details = pd.DataFrame({
        "bus": n.generators.loc[italian_generators, "bus"],
        "carrier": n.generators.loc[italian_generators, "carrier"].map(normalize_carrier),
        "annual_twh": gen_dispatch_twh,
    }).sort_values("annual_twh", ascending=False)

    # -------------------------------------------------------------------------
    # Links
    # -------------------------------------------------------------------------
    annual_link_output_twh = collect_link_electric_output(
        n=n,
        italian_buses=italian_buses_set,
        electric_bus_carriers=electric_bus_carriers,
        bus_carrier=bus_carrier,
    )
    annual_link_input_twh = collect_link_electric_input(
        n=n,
        italian_buses=italian_buses_set,
        electric_bus_carriers=electric_bus_carriers,
        bus_carrier=bus_carrier,
    )

    link_details = pd.DataFrame({
        "carrier": n.links["carrier"].map(normalize_carrier),
        "annual_electric_output_twh": annual_link_output_twh,
        "annual_electric_input_twh": annual_link_input_twh,
    }).sort_values("annual_electric_output_twh", ascending=False)

    # -------------------------------------------------------------------------
    # StorageUnits
    # -------------------------------------------------------------------------
    italian_storageunits = n.storage_units.index[n.storage_units["bus"].isin(italian_buses)]
    su_dispatch_twh = compute_storageunit_dispatch_twh(n, italian_storageunits)

    su_details = pd.DataFrame(index=italian_storageunits)
    if len(italian_storageunits) > 0:
        su_details["bus"] = n.storage_units.loc[italian_storageunits, "bus"]
        su_details["carrier"] = n.storage_units.loc[italian_storageunits, "carrier"].map(normalize_carrier)

        if hasattr(n.storage_units_t, "p_store") and not n.storage_units_t.p_store.empty:
            su_details["charge_twh"] = weighted_sum_twh(
                n.storage_units_t.p_store[italian_storageunits],
                find_weight_series(n, ["stores", "objective", "generators"])
            )
        else:
            su_details["charge_twh"] = 0.0

        su_details["discharge_twh"] = su_dispatch_twh
        su_details["loss_twh"] = (su_details["charge_twh"] - su_details["discharge_twh"]).clip(lower=0.0)
        su_details = su_details.sort_values("discharge_twh", ascending=False)

    # -------------------------------------------------------------------------
    # Main categories
    # -------------------------------------------------------------------------
    category_totals, category_details = compute_category_totals(
        n=n,
        italian_generators=italian_generators,
        italian_storageunits=italian_storageunits,
        gen_dispatch_twh=gen_dispatch_twh,
        su_dispatch_twh=su_dispatch_twh,
        annual_link_output_twh=annual_link_output_twh,
        config=config,
    )

    hydro_twh = category_totals["Idroelettrico"]
    solar_twh = category_totals["Solare"]
    wind_twh = category_totals["Eolico"]
    other_res_twh = category_totals["Altre RES"]
    gas_twh = category_totals["Gas naturale"]
    other_non_res_twh = category_totals["Altra non rinnovabile"]
    h2_to_power_twh = category_totals["Produzione elettrica da H2-to-power"]

    res_twh = hydro_twh + solar_twh + wind_twh + other_res_twh
    thermo_twh = gas_twh + other_non_res_twh
    national_production_twh = res_twh + thermo_twh

    # -------------------------------------------------------------------------
    # Curtailment
    # -------------------------------------------------------------------------
    curtailment_by_gen = compute_generator_curtailment_twh(
        n=n,
        italian_generators=italian_generators,
        curtailment_carriers=set(config["curtailment_generator_carriers"]),
        prefer_opt=config["prefer_optimized_nominal"],
    )
    curtailment_twh = float(curtailment_by_gen.sum())

    curtailment_details = pd.DataFrame({
        "bus": n.generators.loc[curtailment_by_gen.index, "bus"],
        "carrier": n.generators.loc[curtailment_by_gen.index, "carrier"].map(normalize_carrier),
        "curtailment_twh": curtailment_by_gen,
    }).sort_values("curtailment_twh", ascending=False)

    # -------------------------------------------------------------------------
    # Electricity imports
    # -------------------------------------------------------------------------
    border_imports_twh, border_details = compute_border_net_imports_twh(
        n=n,
        italian_buses=italian_buses_set,
        electric_bus_carriers=electric_bus_carriers,
        bus_carrier=bus_carrier,
    )

    # -------------------------------------------------------------------------
    # Storage losses
    # -------------------------------------------------------------------------
    su_losses_twh = compute_storageunit_losses_twh(n, italian_storageunits)
    link_storage_losses_twh, link_storage_details = compute_link_storage_losses_twh(
        n=n,
        annual_link_input_twh=annual_link_input_twh,
        annual_link_output_twh=annual_link_output_twh,
        config=config,
    )
    total_storage_losses_twh = su_losses_twh + link_storage_losses_twh

    # -------------------------------------------------------------------------
    # Hydrogen production by electrolysis
    # -------------------------------------------------------------------------
    h2_prod_twh, electrolysis_details = compute_h2_production_from_electrolysis_twh(
        n=n,
        italian_buses=italian_buses_set,
        hydrogen_bus_carriers=hydrogen_bus_carriers,
        bus_carrier=bus_carrier,
        electrolysis_link_carriers=set(config["electrolysis_link_carriers"]),
    )

    # -------------------------------------------------------------------------
    # Gas imports
    # -------------------------------------------------------------------------
    gas_import_twh, gas_import_details = compute_gas_imports_twh(
        n=n,
        italian_buses=italian_buses_set,
        gas_bus_carriers=gas_bus_carriers,
        bus_carrier=bus_carrier,
        gas_import_link_carriers=set(config["gas_import_link_carriers"]),
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    pypsa_values = {
        "Fabbisogno elettrico totale": total_demand_twh,
        "Produzione nazionale": national_production_twh,
        "Produzione rinnovabile (RES)": res_twh,
        "Idroelettrico": hydro_twh,
        "Solare": solar_twh,
        "Eolico": wind_twh,
        "Altre RES": other_res_twh,
        "Sovragenerazione (curtailment)": -curtailment_twh,
        "Produzione termoelettrica (convenzionale)": thermo_twh,
        "Gas naturale": gas_twh,
        "Altra non rinnovabile": other_non_res_twh,
        "Saldo estero (import-export)": border_imports_twh,
        "Perdite di accumulo": -total_storage_losses_twh,
        "Produzione elettrica da H2-to-power": h2_to_power_twh,
        "Produzione H2 da elettrolizzatori": h2_prod_twh,
        "Import gas": gas_import_twh,
    }

    summary = pd.DataFrame({
        "Voce": list(config["reference_twh"].keys()),
        "DE-IT_TWh": list(config["reference_twh"].values()),
    })
    summary["PyPSA_TWh"] = summary["Voce"].map(pypsa_values)
    summary["Diff_PyPSA_minus_DEIT_TWh"] = summary["PyPSA_TWh"] - summary["DE-IT_TWh"]
    summary["Abs_Diff_TWh"] = summary["Diff_PyPSA_minus_DEIT_TWh"].abs()
    summary["Rel_Diff_%_vs_DEIT"] = np.where(
        summary["DE-IT_TWh"].notna() & (summary["DE-IT_TWh"].abs() > 1e-12),
        100.0 * summary["Diff_PyPSA_minus_DEIT_TWh"] / summary["DE-IT_TWh"],
        np.nan,
    )

    details = {
        "loads_detail": load_details,
        "generators_detail": generator_details,
        "links_detail": link_details,
        "storageunits_detail": su_details,
        "curtailment_detail": curtailment_details,
        "border_electricity_detail": border_details,
        "link_storage_detail": link_storage_details,
        "electrolysis_detail": electrolysis_details,
        "gas_import_detail": gas_import_details,
    }

    for key, df in category_details.items():
        details[f"category_{key.lower().replace(' ', '_').replace('-', '_')}_detail"] = df

    return summary, details


# =============================================================================
# OUTPUT
# =============================================================================

def save_outputs(summary: pd.DataFrame, details: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save all CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_dir / "deit_vs_pypsa_balance_summary.csv", index=False)

    for name, df in details.items():
        if df is not None and not df.empty:
            df.to_csv(output_dir / f"{name}.csv")


def plot_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot only rows with a DE-IT reference for the main grouped comparison
    plot_df = summary[summary["DE-IT_TWh"].notna()].copy()

    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(plot_df))
    width = 0.38

    ax.bar(x - width / 2, plot_df["DE-IT_TWh"], width=width, label="DE-IT")
    ax.bar(x + width / 2, plot_df["PyPSA_TWh"], width=width, label="PyPSA")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["Voce"], rotation=45, ha="right")
    ax.set_ylabel("TWh")
    ax.set_title("Confronto bilancio elettrico Italia: DE-IT vs PyPSA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "deit_vs_pypsa_balance_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(plot_df["Voce"], plot_df["Diff_PyPSA_minus_DEIT_TWh"])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("TWh")
    ax.set_title("Differenza PyPSA - DE-IT")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "deit_vs_pypsa_balance_difference.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Extra plot for rows without DE-IT reference
    extra_df = summary[summary["DE-IT_TWh"].isna()].copy()
    if not extra_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(extra_df["Voce"], extra_df["PyPSA_TWh"])
        ax.set_ylabel("TWh")
        ax.set_title("Indicatori extra calcolati da PyPSA")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "deit_vs_pypsa_extra_indicators.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    """Print compact summary."""
    cols = ["Voce", "DE-IT_TWh", "PyPSA_TWh", "Diff_PyPSA_minus_DEIT_TWh"]
    print("\n=== DE-IT vs PyPSA balance summary (TWh) ===")
    print(summary[cols].to_string(index=False))


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Italian PyPSA network outputs against DE-IT balance values."
    )
    parser.add_argument(
        "network",
        type=str,
        help="Path to input PyPSA network (.nc).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="deit_balance_comparison",
        help="Output directory.",
    )
    parser.add_argument(
        "--exclude-phs-from-hydro",
        action="store_true",
        help="Exclude PHS StorageUnit discharge from hydro production.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    network_path = Path(args.network)
    output_dir = Path(args.output_dir)

    if not network_path.exists():
        raise FileNotFoundError(f"Network file not found: {network_path}")

    config = dict(CONFIG)
    config["include_phs_in_hydro"] = not args.exclude_phs_from_hydro

    print(f"Loading network: {network_path}")
    n = pypsa.Network(network_path)

    print("Computing balance...")
    summary, details = compute_balance(n, config)

    print("Saving outputs...")
    save_outputs(summary, details, output_dir)
    plot_comparison(summary, output_dir)
    print_summary(summary)

    print(f"\nDone. Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()