"""
Synthetic test-data generator for a village-to-mandi freight corridor.

This produces clearly-labelled SYNTHETIC data for benchmarking the matching
engine end to end. It is not real field data -- see the hackathon
submission's "Expected Impact" section for why that distinction matters,
and why no number produced from this generator should be quoted as a
real-world result.
"""
import math
import random
from typing import List, Tuple

from .models import Location, Shipment, Vehicle, haversine_km

MANDI = Location("Central Mandi", 21.1700, 79.0900)  # an arbitrary rural reference point

VILLAGE_NAMES = [
    "Ambegaon", "Birwadi", "Chandur", "Devgaon", "Ekodi", "Fatehpur",
    "Gondpipri", "Harangul", "Islampur", "Jamgaon", "Kondhali", "Lonar",
    "Mendha", "Nandur", "Osmannagar", "Pimpalgaon",
]


def _point_near(center: Location, min_km: float, max_km: float, rng: random.Random) -> Location:
    r_km = rng.uniform(min_km, max_km)
    theta = rng.uniform(0, 2 * math.pi)
    dlat = (r_km / 111.0) * math.cos(theta)
    dlng = (r_km / (111.0 * math.cos(math.radians(center.lat)))) * math.sin(theta)
    return Location(f"pt_{rng.randint(0, 999999)}", center.lat + dlat, center.lng + dlng)


def _point_near_named(name: str, rng: random.Random) -> Location:
    pt = _point_near(MANDI, 3.0, 18.0, rng)
    return Location(name, pt.lat, pt.lng)


def generate_scenario(
    n_shipments: int,
    n_vehicles: int,
    density: str = "sparse",
    seed: int = 42,
) -> Tuple[List[Shipment], List[Vehicle]]:
    """
    density: "sparse" (everyday demand) or "dense" (harvest-season-like
    demand -- more weight per shipment, used in the demo to show the
    optimizer's advantage more clearly than a thin, low-density scenario
    would).

    Returns (shipments, vehicles).
    """
    rng = random.Random(seed)
    villages = [_point_near_named(name, rng) for name in VILLAGE_NAMES[: max(6, n_shipments // 2)]]

    weight_range = (150, 600) if density == "sparse" else (300, 900)

    shipments: List[Shipment] = []
    for i in range(n_shipments):
        village = rng.choice(villages)
        # Most shipments move produce out to the mandi; a minority move
        # materials from the mandi/input dealers back out to a village.
        if rng.random() < 0.75:
            pickup, dropoff = village, MANDI
        else:
            pickup, dropoff = MANDI, village

        pickup_start = rng.randint(0, 180)
        pickup_end = pickup_start + rng.randint(30, 90)
        travel_min = int(haversine_km(pickup, dropoff) / 30.0 * 60)  # 30 km/h rural-road assumption
        dropoff_start = pickup_start + travel_min
        dropoff_end = dropoff_start + rng.randint(60, 150)

        shipments.append(Shipment(
            id=f"S{i + 1:03d}",
            pickup=pickup,
            dropoff=dropoff,
            weight_kg=rng.randint(*weight_range),
            pickup_window=(pickup_start, pickup_end),
            dropoff_window=(dropoff_start, dropoff_end),
            shipper_name=f"{village.name} farmer",
        ))

    vehicles: List[Vehicle] = []
    for i in range(n_vehicles):
        # Roughly 40% of available capacity in a given window is "recovered"
        # return-leg capacity rather than freshly posted capacity -- an
        # assumption for the demo, not a measured figure.
        is_return = rng.random() < 0.4
        start_loc = MANDI if is_return else rng.choice(villages)
        shift_start = rng.randint(120, 240) if is_return else rng.randint(0, 60)

        vehicles.append(Vehicle(
            id=f"V{i + 1:02d}",
            start=start_loc,
            capacity_kg=rng.choice([750, 1000, 1500, 2000]),
            shift_window=(shift_start, shift_start + 420),
            is_return_leg=is_return,
        ))

    return shipments, vehicles
