"""
Baseline matcher: a simple greedy "call the nearest driver you know" model,
representing how shippers find transport today, per the hackathon brief
("people phone a driver they know, wait, pay for a full trip on a part
load, and the vehicle usually comes back empty").

No pooling. No return-leg reuse. Each matched shipment is billed at the
vehicle's full base rate for its own round trip, because that is what the
brief describes happening today.

This module exists so that vrppd_solver.py can be benchmarked against it --
see benchmark.py -- which is a mandatory requirement of the hackathon brief,
not an optional nice-to-have.
"""
from dataclasses import dataclass
from typing import List, Optional

from .models import Shipment, Vehicle, haversine_km


@dataclass
class BaselineAssignment:
    shipment: Shipment
    vehicle: Optional[Vehicle]   # None if no compatible vehicle was available
    distance_km: float
    cost_inr: float


def run_baseline(shipments: List[Shipment], vehicles: List[Vehicle]) -> List[BaselineAssignment]:
    """
    Processes shipments in pickup-time order (first come, first served, like
    a phone call), assigning each to the nearest still-available, large-
    enough vehicle. Every vehicle is used at most once: no pooling, and a
    vehicle already assigned a shipment is not offered for any other one,
    including on its return leg.
    """
    available = list(vehicles)
    assignments: List[BaselineAssignment] = []

    for shipment in sorted(shipments, key=lambda s: s.pickup_window[0]):
        best_vehicle: Optional[Vehicle] = None
        best_distance = float("inf")

        for vehicle in available:
            if shipment.weight_kg > vehicle.capacity_kg:
                continue
            d = haversine_km(vehicle.start, shipment.pickup)
            if d < best_distance:
                best_distance = d
                best_vehicle = vehicle

        if best_vehicle is None:
            assignments.append(BaselineAssignment(shipment, None, 0.0, 0.0))
            continue

        trip_km = haversine_km(best_vehicle.start, shipment.pickup) + haversine_km(
            shipment.pickup, shipment.dropoff
        )
        # The shipper pays for the whole vehicle trip, full base rate, even
        # though their own load may only partly fill it -- this is the
        # "part load, full price" behaviour the brief describes.
        cost = trip_km * best_vehicle.base_rate_per_km
        assignments.append(BaselineAssignment(shipment, best_vehicle, round(trip_km, 2), round(cost, 2)))
        available.remove(best_vehicle)

    return assignments
