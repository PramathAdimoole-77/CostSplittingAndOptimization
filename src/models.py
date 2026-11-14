"""
Core data models for the Saajha matching engine.

These are plain dataclasses with no external dependencies, so they (and
anything built only from them, such as the baseline matcher, the data
generator, and the fare splitter) can be tested and reused even without
Google OR-Tools installed. Only vrppd_solver.py needs OR-Tools.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lng: float


@dataclass
class Shipment:
    id: str
    pickup: Location
    dropoff: Location
    weight_kg: float
    pickup_window: Tuple[int, int]    # (earliest_minute, latest_minute), minutes from scenario start
    dropoff_window: Tuple[int, int]   # (earliest_minute, latest_minute)
    shipper_name: str = ""


@dataclass
class Vehicle:
    id: str
    start: Location
    capacity_kg: float
    shift_window: Tuple[int, int] = (0, 420)   # vehicle's working window, minutes from scenario start
    is_return_leg: bool = False                # True = "recovered" capacity (the KhaliVapsi case)
    base_rate_per_km: float = 20.0             # INR/km, illustrative
    return_leg_discount: float = 0.4           # fraction off base rate when is_return_leg


@dataclass
class Stop:
    kind: str           # "pickup" or "delivery"
    shipment_id: str
    location: Location
    arrival_min: int


@dataclass
class VehicleRoute:
    vehicle: Vehicle
    stops: List[Stop]          # in visiting order
    shipment_ids: List[str]
    distance_km: float


@dataclass
class SolveResult:
    routes: List[VehicleRoute]
    unassigned_shipment_ids: List[str]
    objective: Optional[float]


def haversine_km(a: Location, b: Location) -> float:
    """Great-circle distance between two locations, in kilometres.

    Used as the distance metric throughout this prototype. Swapping in a
    real road-network distance (e.g. from OSRM) is a drop-in replacement:
    every other module calls this function rather than computing distance
    itself.
    """
    if a.lat == b.lat and a.lng == b.lng:
        return 0.0
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [a.lat, a.lng, b.lat, b.lng])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))
