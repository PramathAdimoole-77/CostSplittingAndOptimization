"""
Fare Explainer: turns a pooled vehicle route into a transparent, itemised
price for every shipper on it. This is what satisfies the hackathon brief's
requirement that every shipper sharing a vehicle can see exactly how their
price was calculated.

Allocation method: a weighted blend of three components, chosen because
each is independently easy to explain to a shipper who has never used the
platform before, rather than one single opaque number:

  * distance share -- this shipment's own pickup-to-dropoff distance, as a
    fraction of all pooled shipments' own distances
  * weight share    -- this shipment's weight, as a fraction of total
    weight carried on the trip
  * detour share    -- the extra distance this shipment's two stops added
    to the route, versus running the same route without them

Each component gets an equal one-third weight by default; that split is a
reasonable, explainable starting point for a hackathon prototype, not a
claim that it is the economically optimal split. A more rigorous
alternative (e.g. a Shapley-value allocation) is noted as future work in
the main submission.
"""
from dataclasses import dataclass
from typing import Dict, List

from .models import Shipment, VehicleRoute, haversine_km

DISTANCE_WEIGHT = 1 / 3
WEIGHT_WEIGHT = 1 / 3
DETOUR_WEIGHT = 1 / 3


@dataclass
class FareLine:
    shipment_id: str
    shipper_name: str
    distance_component_inr: float
    weight_component_inr: float
    detour_component_inr: float
    total_inr: float


def _own_leg_km(shipment: Shipment) -> float:
    return haversine_km(shipment.pickup, shipment.dropoff)


def _route_km_without(route: VehicleRoute, shipment_id: str) -> float:
    """
    Approximate distance of the route if this shipment's pickup and
    delivery stops were removed, keeping the remaining stops in their
    original relative order (not re-optimised). This is a reasonable,
    explainable approximation for a prototype -- not an exact marginal-cost
    recomputation, which would require re-solving the route.
    """
    stops = [s for s in route.stops if s.shipment_id != shipment_id]
    if not stops:
        return 0.0
    points = [route.vehicle.start] + [s.location for s in stops]
    return sum(haversine_km(a, b) for a, b in zip(points, points[1:]))


def explain_fares(
    route: VehicleRoute,
    shipments_by_id: Dict[str, Shipment],
    total_trip_cost_inr: float,
) -> List[FareLine]:
    ids = route.shipment_ids
    if not ids:
        return []

    own_legs = {sid: _own_leg_km(shipments_by_id[sid]) for sid in ids}
    weights = {sid: shipments_by_id[sid].weight_kg for sid in ids}
    detours = {sid: max(0.0, route.distance_km - _route_km_without(route, sid)) for sid in ids}

    total_own_leg = sum(own_legs.values()) or 1.0
    total_weight = sum(weights.values()) or 1.0
    total_detour = sum(detours.values()) or 1.0

    lines: List[FareLine] = []
    for sid in ids:
        distance_share = own_legs[sid] / total_own_leg
        weight_share = weights[sid] / total_weight
        detour_share = detours[sid] / total_detour

        distance_component = total_trip_cost_inr * DISTANCE_WEIGHT * distance_share
        weight_component = total_trip_cost_inr * WEIGHT_WEIGHT * weight_share
        detour_component = total_trip_cost_inr * DETOUR_WEIGHT * detour_share

        lines.append(FareLine(
            shipment_id=sid,
            shipper_name=shipments_by_id[sid].shipper_name,
            distance_component_inr=round(distance_component, 2),
            weight_component_inr=round(weight_component, 2),
            detour_component_inr=round(detour_component, 2),
            total_inr=round(distance_component + weight_component + detour_component, 2),
        ))
    return lines
