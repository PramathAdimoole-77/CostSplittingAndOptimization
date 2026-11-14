"""
Basic sanity tests.

test_models_*, test_generate_*, test_baseline_*, and test_fare_* run with no
external dependencies. test_solver_* needs Google OR-Tools installed
(`pip install -r requirements.txt`) and is skipped automatically if it
isn't available, so `pytest tests/` still passes without it.

Run with:  pytest tests/ -v
      or:  python tests/test_basic.py   (plain-assert runner, no pytest needed)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import Location, Shipment, Vehicle, Stop, VehicleRoute, haversine_km
from src.data_generator import generate_scenario
from src.baseline_matcher import run_baseline
from src.fare_splitter import explain_fares

try:
    import ortools  # noqa: F401
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False


def test_haversine_zero_distance():
    a = Location("a", 21.0, 79.0)
    assert haversine_km(a, a) == 0.0


def test_haversine_known_distance():
    # One degree of latitude is approximately 111 km.
    a = Location("a", 0.0, 0.0)
    b = Location("b", 1.0, 0.0)
    d = haversine_km(a, b)
    assert 108 <= d <= 112, f"expected ~111 km, got {d}"


def test_generate_scenario_sizes_and_windows():
    shipments, vehicles = generate_scenario(10, 6, density="sparse", seed=1)
    assert len(shipments) == 10
    assert len(vehicles) == 6
    for s in shipments:
        assert s.pickup_window[0] < s.pickup_window[1]
        assert s.dropoff_window[0] < s.dropoff_window[1]
        assert s.weight_kg > 0
    for v in vehicles:
        assert v.capacity_kg > 0
        assert v.shift_window[0] < v.shift_window[1]


def test_generate_scenario_is_reproducible():
    s1, v1 = generate_scenario(8, 5, density="sparse", seed=99)
    s2, v2 = generate_scenario(8, 5, density="sparse", seed=99)
    assert [s.id for s in s1] == [s.id for s in s2]
    assert [round(s.weight_kg) for s in s1] == [round(s.weight_kg) for s in s2]


def test_baseline_no_pooling_one_vehicle_per_shipment():
    shipments, vehicles = generate_scenario(8, 5, density="sparse", seed=2)
    assignments = run_baseline(shipments, vehicles)
    matched = [a for a in assignments if a.vehicle is not None]
    assert len(matched) <= len(vehicles)
    # Each vehicle should be used for at most one shipment (no pooling, no reuse).
    vehicle_ids_used = [a.vehicle.id for a in matched]
    assert len(vehicle_ids_used) == len(set(vehicle_ids_used))


def test_baseline_respects_capacity():
    shipments, vehicles = generate_scenario(8, 5, density="dense", seed=3)
    assignments = run_baseline(shipments, vehicles)
    for a in assignments:
        if a.vehicle is not None:
            assert a.shipment.weight_kg <= a.vehicle.capacity_kg


def test_fare_splitter_lines_sum_to_total_cost():
    vehicle = Vehicle(id="V1", start=Location("depot", 21.00, 79.00), capacity_kg=1000, shift_window=(0, 400))
    s1 = Shipment(id="S1", pickup=Location("p1", 21.01, 79.00), dropoff=Location("d1", 21.05, 79.05),
                  weight_kg=200, pickup_window=(0, 60), dropoff_window=(30, 120), shipper_name="Farmer A")
    s2 = Shipment(id="S2", pickup=Location("p2", 21.02, 79.01), dropoff=Location("d2", 21.06, 79.06),
                  weight_kg=300, pickup_window=(0, 60), dropoff_window=(30, 120), shipper_name="Farmer B")

    stops = [
        Stop("pickup", "S1", s1.pickup, 10),
        Stop("pickup", "S2", s2.pickup, 15),
        Stop("delivery", "S1", s1.dropoff, 50),
        Stop("delivery", "S2", s2.dropoff, 60),
    ]
    route = VehicleRoute(vehicle=vehicle, stops=stops, shipment_ids=["S1", "S2"], distance_km=12.0)

    lines = explain_fares(route, {"S1": s1, "S2": s2}, total_trip_cost_inr=240.0)
    assert len(lines) == 2
    assert abs(sum(l.total_inr for l in lines) - 240.0) < 0.5


def test_fare_splitter_handles_single_shipment_route():
    vehicle = Vehicle(id="V2", start=Location("depot", 21.00, 79.00), capacity_kg=1000, shift_window=(0, 400))
    s1 = Shipment(id="S1", pickup=Location("p1", 21.01, 79.00), dropoff=Location("d1", 21.05, 79.05),
                  weight_kg=200, pickup_window=(0, 60), dropoff_window=(30, 120), shipper_name="Farmer A")
    stops = [Stop("pickup", "S1", s1.pickup, 10), Stop("delivery", "S1", s1.dropoff, 50)]
    route = VehicleRoute(vehicle=vehicle, stops=stops, shipment_ids=["S1"], distance_km=8.0)

    lines = explain_fares(route, {"S1": s1}, total_trip_cost_inr=160.0)
    assert len(lines) == 1
    assert abs(lines[0].total_inr - 160.0) < 0.5


def test_solver_respects_capacity_and_time_windows():
    if not HAS_ORTOOLS:
        return "skipped"
    from src.vrppd_solver import solve

    shipments, vehicles = generate_scenario(10, 6, density="sparse", seed=5)
    result = solve(shipments, vehicles, time_limit_seconds=5)

    for route in result.routes:
        carried = 0
        for stop in route.stops:
            weight = next(s.weight_kg for s in shipments if s.id == stop.shipment_id)
            carried += weight if stop.kind == "pickup" else -weight
            assert carried <= route.vehicle.capacity_kg + 1e-6
        for stop in route.stops:
            shipment = next(s for s in shipments if s.id == stop.shipment_id)
            window = shipment.pickup_window if stop.kind == "pickup" else shipment.dropoff_window
            assert window[0] <= stop.arrival_min <= window[1]


ALL_TESTS = [
    test_haversine_zero_distance,
    test_haversine_known_distance,
    test_generate_scenario_sizes_and_windows,
    test_generate_scenario_is_reproducible,
    test_baseline_no_pooling_one_vehicle_per_shipment,
    test_baseline_respects_capacity,
    test_fare_splitter_lines_sum_to_total_cost,
    test_fare_splitter_handles_single_shipment_route,
    test_solver_respects_capacity_and_time_windows,
]


if __name__ == "__main__":
    failures = 0
    skipped = 0
    for test in ALL_TESTS:
        try:
            result = test()
            if result == "skipped":
                skipped += 1
                print(f"SKIPPED {test.__name__} (OR-Tools not installed)")
            else:
                print(f"PASSED  {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAILED  {test.__name__}: {e}")
    ran = len(ALL_TESTS) - skipped
    print(f"\n{ran - failures}/{ran} passed ({skipped} skipped)")
    sys.exit(1 if failures else 0)
