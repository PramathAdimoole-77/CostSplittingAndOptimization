"""
Runs both the baseline matcher and the VRPPD/VRPTW optimizer on the same
scenario and prints a side-by-side comparison. This is the benchmarking
step the hackathon brief requires before any optimization claim is made --
see baseline_matcher.py and vrppd_solver.py.
"""
from .baseline_matcher import run_baseline
from .data_generator import generate_scenario
from .vrppd_solver import solve


def _baseline_summary(shipments, vehicles):
    assignments = run_baseline(shipments, vehicles)
    matched = [a for a in assignments if a.vehicle is not None]
    total_cost = sum(a.cost_inr for a in matched)
    total_weight = sum(a.shipment.weight_kg for a in matched)
    total_capacity_used = sum(a.vehicle.capacity_kg for a in matched) or 1
    return {
        "matched": len(matched),
        "unmatched": len(assignments) - len(matched),
        "total_cost_inr": round(total_cost, 2),
        "vehicles_used": len({a.vehicle.id for a in matched}),
        "avg_load_factor": round(total_weight / total_capacity_used, 2),
    }


def _optimized_summary(shipments, vehicles):
    result = solve(shipments, vehicles)

    total_cost = 0.0
    return_leg_matches = 0
    for route in result.routes:
        rate = route.vehicle.base_rate_per_km * (
            1 - route.vehicle.return_leg_discount if route.vehicle.is_return_leg else 1.0
        )
        total_cost += route.distance_km * rate
        if route.vehicle.is_return_leg:
            return_leg_matches += len(route.shipment_ids)

    matched_weight = sum(s.weight_kg for s in shipments if s.id not in result.unassigned_shipment_ids)
    total_capacity_used = sum(r.vehicle.capacity_kg for r in result.routes) or 1

    summary = {
        "matched": len(shipments) - len(result.unassigned_shipment_ids),
        "unmatched": len(result.unassigned_shipment_ids),
        "total_cost_inr": round(total_cost, 2),
        "vehicles_used": len(result.routes),
        "avg_load_factor": round(matched_weight / total_capacity_used, 2),
        "return_leg_matches": return_leg_matches,
    }
    return summary, result


def run_benchmark(n_shipments: int = 18, n_vehicles: int = 10, density: str = "sparse", seed: int = 42):
    shipments, vehicles = generate_scenario(n_shipments, n_vehicles, density, seed)
    baseline = _baseline_summary(shipments, vehicles)
    optimized, result = _optimized_summary(shipments, vehicles)

    print(f"\n=== Scenario: {density}, {n_shipments} shipments, {n_vehicles} vehicles (seed={seed}) ===")
    print(f"{'Metric':<22}{'Baseline':<15}{'Optimized':<15}")
    print(f"{'Matched shipments':<22}{baseline['matched']:<15}{optimized['matched']:<15}")
    print(f"{'Total cost (INR)':<22}{baseline['total_cost_inr']:<15}{optimized['total_cost_inr']:<15}")
    print(f"{'Vehicles used':<22}{baseline['vehicles_used']:<15}{optimized['vehicles_used']:<15}")
    print(f"{'Avg load factor':<22}{baseline['avg_load_factor']:<15}{optimized['avg_load_factor']:<15}")
    print(f"{'Return-leg matches':<22}{'0 (n/a)':<15}{optimized['return_leg_matches']:<15}")

    if baseline["total_cost_inr"] > 0:
        pct = 100 * (1 - optimized["total_cost_inr"] / baseline["total_cost_inr"])
        print(f"\nCost difference vs. baseline: {pct:.1f}% (synthetic data -- see README before citing this)")

    return baseline, optimized, result, shipments, vehicles
