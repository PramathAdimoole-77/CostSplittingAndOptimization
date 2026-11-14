"""
End-to-end demo for the Saajha matching engine.

Run with (from the project root, after `pip install -r requirements.txt`):

    python -m demo.run_demo
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark import run_benchmark
from src.fare_splitter import explain_fares


def print_fare_breakdown(result, shipments):
    shipments_by_id = {s.id: s for s in shipments}
    pooled = [r for r in result.routes if len(r.shipment_ids) > 1]
    if not pooled:
        print("\nNo pooled (multi-shipment) trip in this run to show a fare breakdown for.")
        return

    route = pooled[0]
    rate = route.vehicle.base_rate_per_km * (
        1 - route.vehicle.return_leg_discount if route.vehicle.is_return_leg else 1.0
    )
    total_cost = route.distance_km * rate

    tag = " (return-leg, discounted)" if route.vehicle.is_return_leg else ""
    print(f"\n--- Fare Explainer: vehicle {route.vehicle.id}{tag}, "
          f"{len(route.shipment_ids)} shipments pooled, {route.distance_km} km, "
          f"total trip cost INR {total_cost:.2f} ---")
    for line in explain_fares(route, shipments_by_id, total_cost):
        print(f"  {line.shipment_id} ({line.shipper_name}): "
              f"distance INR {line.distance_component_inr} + "
              f"weight INR {line.weight_component_inr} + "
              f"detour INR {line.detour_component_inr} "
              f"= INR {line.total_inr}")


def main():
    print("Saajha matching engine -- demo run")
    print("Synthetic data only. See README.md before citing any of these numbers elsewhere.")

    print("\n" + "#" * 60)
    print("# Scenario 1: sparse, everyday demand")
    print("#" * 60)
    _, _, result_sparse, shipments_sparse, _ = run_benchmark(
        n_shipments=14, n_vehicles=9, density="sparse", seed=42
    )
    print_fare_breakdown(result_sparse, shipments_sparse)

    print("\n" + "#" * 60)
    print("# Scenario 2: dense, harvest-season-like demand")
    print("#" * 60)
    _, _, result_dense, shipments_dense, _ = run_benchmark(
        n_shipments=28, n_vehicles=12, density="dense", seed=7
    )
    print_fare_breakdown(result_dense, shipments_dense)


if __name__ == "__main__":
    main()
