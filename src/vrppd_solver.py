"""
Core matching engine: Vehicle Routing Problem with Pickup and Delivery,
Capacity Constraints and Time Windows (VRPPD/VRPTW), solved with Google
OR-Tools.

This is the optimizer side of the benchmark the hackathon brief requires;
see baseline_matcher.py for the simple baseline it is compared against in
benchmark.py.

Design notes (matching the Saajha submission):
  * Fresh capacity (vehicle.is_return_leg = False) and recovered return-leg
    capacity (vehicle.is_return_leg = True) are modelled as the same kind of
    vehicle, just costed at a discounted rate -- so the solver treats both
    as one shared pool, exactly as described in the submission, rather than
    as two separate systems.
  * Demand uncertainty is NOT handled by forecasting inside this module. In
    the full system, solve() is re-invoked on a short rolling window every
    time a new shipment is posted, new capacity becomes available, or a
    transporter cancels (see README.md) -- this function always solves
    exactly one such batch.
  * Known simplification: shipments are assumed to be jointly feasible for
    the given vehicles and time windows (the data generator is tuned to
    keep instances solvable). A production version would add bounded
    disjunctions so an individual hard-to-place shipment can be left
    unmatched, with a penalty, rather than the whole batch failing to solve.

Requires: pip install ortools  (see requirements.txt)
"""
from typing import Dict, List, Tuple

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

from .models import Location, Shipment, Stop, Vehicle, VehicleRoute, SolveResult, haversine_km

AVERAGE_SPEED_KMPH = 30.0      # rural-road assumption; swap in OSRM output in production
TIME_HORIZON_MIN = 24 * 60
WAIT_SLACK_MIN = 180           # how long a vehicle may wait at a node for a window to open
FIXED_COST_PER_VEHICLE = 5000  # relative weight only -- rewards pooling onto fewer vehicles

END_DEPOT = 0  # node 0 is always the shared, zero-cost "end" depot


def _build_nodes(
    shipments: List[Shipment], vehicles: List[Vehicle]
) -> Tuple[List[Location], Dict[str, int], Dict[str, int], Dict[int, Tuple[str, str]]]:
    """
    Node layout:
      0                   -> dummy end depot (zero distance/time to and from everywhere,
                              so vehicles don't have to drive back to their own start)
      1 .. len(vehicles)  -> one start node per vehicle, at its current location
      after that          -> a pickup node then a delivery node, per shipment

    Returns (nodes, pickup_of, delivery_of, node_lookup) where node_lookup
    maps a node id to ("pickup"|"delivery", shipment_id) for every
    shipment-related node.
    """
    nodes: List[Location] = [Location("END_DEPOT", vehicles[0].start.lat, vehicles[0].start.lng)]
    nodes += [v.start for v in vehicles]

    pickup_of: Dict[str, int] = {}
    delivery_of: Dict[str, int] = {}
    node_lookup: Dict[int, Tuple[str, str]] = {}

    for s in shipments:
        pickup_of[s.id] = len(nodes)
        node_lookup[len(nodes)] = ("pickup", s.id)
        nodes.append(s.pickup)

        delivery_of[s.id] = len(nodes)
        node_lookup[len(nodes)] = ("delivery", s.id)
        nodes.append(s.dropoff)

    return nodes, pickup_of, delivery_of, node_lookup


def solve(shipments: List[Shipment], vehicles: List[Vehicle], time_limit_seconds: int = 10) -> SolveResult:
    if not shipments or not vehicles:
        return SolveResult(routes=[], unassigned_shipment_ids=[s.id for s in shipments], objective=0.0)

    nodes, pickup_of, delivery_of, node_lookup = _build_nodes(shipments, vehicles)
    n_vehicles = len(vehicles)
    starts = [1 + i for i in range(n_vehicles)]
    ends = [END_DEPOT] * n_vehicles

    manager = pywrapcp.RoutingIndexManager(len(nodes), n_vehicles, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    def km_between(from_node: int, to_node: int) -> float:
        if from_node == END_DEPOT or to_node == END_DEPOT:
            return 0.0
        return haversine_km(nodes[from_node], nodes[to_node])

    # ---- Objective: per-vehicle cost, discounted for return-leg capacity ----
    for v_idx, vehicle in enumerate(vehicles):
        rate = vehicle.base_rate_per_km * (1 - vehicle.return_leg_discount if vehicle.is_return_leg else 1.0)

        def make_cost_cb(rate: float = rate):
            def cost_cb(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return int(round(km_between(from_node, to_node) * rate * 100))  # scaled integer units
            return cost_cb

        cb_index = routing.RegisterTransitCallback(make_cost_cb())
        routing.SetArcCostEvaluatorOfVehicle(cb_index, v_idx)

    # Rewards the solver for pooling shipments onto fewer vehicles instead of spreading them out.
    routing.SetFixedCostOfAllVehicles(FIXED_COST_PER_VEHICLE)

    # ---- Capacity dimension ----
    weight_of_shipment = {s.id: s.weight_kg for s in shipments}

    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        info = node_lookup.get(node)
        if info is None:
            return 0
        kind, shipment_id = info
        w = weight_of_shipment[shipment_id]
        return int(w) if kind == "pickup" else -int(w)

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    vehicle_capacities = [int(v.capacity_kg) for v in vehicles]
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, vehicle_capacities, True, "Capacity")

    # ---- Time dimension ----
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(km_between(from_node, to_node) / AVERAGE_SPEED_KMPH * 60))

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.AddDimension(time_callback_index, WAIT_SLACK_MIN, TIME_HORIZON_MIN, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for v_idx, vehicle in enumerate(vehicles):
        time_dimension.CumulVar(routing.Start(v_idx)).SetRange(*vehicle.shift_window)

    for s in shipments:
        p_index = manager.NodeToIndex(pickup_of[s.id])
        d_index = manager.NodeToIndex(delivery_of[s.id])

        time_dimension.CumulVar(p_index).SetRange(*s.pickup_window)
        time_dimension.CumulVar(d_index).SetRange(*s.dropoff_window)

        routing.AddPickupAndDelivery(p_index, d_index)
        routing.solver().Add(routing.VehicleVar(p_index) == routing.VehicleVar(d_index))
        routing.solver().Add(time_dimension.CumulVar(p_index) <= time_dimension.CumulVar(d_index))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return SolveResult(routes=[], unassigned_shipment_ids=[s.id for s in shipments], objective=None)

    shipments_by_id = {s.id: s for s in shipments}
    routes: List[VehicleRoute] = []
    assigned_ids = set()

    for v_idx, vehicle in enumerate(vehicles):
        index = routing.Start(v_idx)
        stops: List[Stop] = []
        route_km = 0.0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            info = node_lookup.get(node)
            if info is not None:
                kind, shipment_id = info
                arrival = solution.Value(time_dimension.CumulVar(index))
                loc = shipments_by_id[shipment_id].pickup if kind == "pickup" else shipments_by_id[shipment_id].dropoff
                stops.append(Stop(kind, shipment_id, loc, arrival))
                if kind == "pickup":
                    assigned_ids.add(shipment_id)

            next_index = solution.Value(routing.NextVar(index))
            route_km += km_between(node, manager.IndexToNode(next_index))
            index = next_index

        if stops:
            routes.append(VehicleRoute(
                vehicle=vehicle,
                stops=stops,
                shipment_ids=sorted({st.shipment_id for st in stops}),
                distance_km=round(route_km, 2),
            ))

    unassigned = [s.id for s in shipments if s.id not in assigned_ids]
    return SolveResult(routes=routes, unassigned_shipment_ids=unassigned, objective=solution.ObjectiveValue())
