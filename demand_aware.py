
import numpy as np


def calculate_demand_aware_buffer(
    cancellation_probs,
    capacity,
    predicted_new_demand,
    max_walk_risk=0.05,
    max_buffer=150,
    simulations=10000,
    seed=42,
    adr=150.0,
    contribution_margin=0.70,
    walk_cost_multiplier=2.0,
    future_cancel_rate=None,
):
    """
    Experimental 30-day-ahead demand-aware simulation.

    k = maximum additional reservations the hotel
        is willing to accept after the forecast date.

    Actual new bookings = min(random demand, k).

    Assumptions:
    - One reservation occupies one room.
    - Future demand follows a Poisson distribution.
    - Cancellation outcomes are independent.
    - Extra bookings use an assumed cancellation rate.
    - New reservations can be accepted up to the limit.
    """

    p = np.asarray(cancellation_probs, dtype=float)

    if not np.all(np.isfinite(p)):
        raise ValueError("Cancellation probabilities must be finite.")

    if np.any((p < 0) | (p > 1)):
        raise ValueError("Cancellation probabilities must be 0–1.")

    if capacity <= 0:
        raise ValueError("Capacity must be positive.")

    if predicted_new_demand < 0:
        raise ValueError("Predicted demand cannot be negative.")

    if max_buffer < 0 or simulations <= 0:
        raise ValueError("Invalid simulation settings.")

    if not 0 <= max_walk_risk <= 1:
        raise ValueError("Walk risk must be between 0 and 1.")

    if future_cancel_rate is None:
        future_cancel_rate = float(p.mean()) if len(p) else 0.0

    if not 0 <= future_cancel_rate <= 1:
        raise ValueError("Invalid future cancellation rate.")

    if adr <= 0 or not 0 <= contribution_margin <= 1:
        raise ValueError("Invalid financial assumptions.")

    if walk_cost_multiplier < 0:
        raise ValueError("Walk cost multiplier cannot be negative.")

    rng = np.random.default_rng(seed)

    # Existing bookings active at the forecast date.
    existing_arrivals = (
        rng.random((simulations, len(p))) >= p
    ).sum(axis=1)

    # Future new bookings are uncertain.
    future_demand = rng.poisson(
        lam=float(predicted_new_demand),
        size=simulations
    )

    # Pre-generate cancellation outcomes for new bookings.
    future_showup_draws = (
        rng.random((simulations, max_buffer))
        >= future_cancel_rate
    )

    cumulative_future_showups = np.concatenate(
        [
            np.zeros((simulations, 1), dtype=int),
            np.cumsum(future_showup_draws, axis=1)
        ],
        axis=1
    )

    spoilage_cost_per_room = adr * contribution_margin
    walk_cost_per_room = adr * walk_cost_multiplier

    results = []

    for k in range(max_buffer + 1):

        # Demand may be lower than the booking limit.
        accepted_new = np.minimum(future_demand, k)

        future_arrivals = cumulative_future_showups[
            np.arange(simulations),
            accepted_new
        ]

        total_arrivals = existing_arrivals + future_arrivals

        empty_rooms = np.maximum(
            capacity - total_arrivals, 0
        )

        walked_rooms = np.maximum(
            total_arrivals - capacity, 0
        )

        walk_probability = float(
            np.mean(walked_rooms > 0)
        )

        expected_empty = float(empty_rooms.mean())
        expected_walked = float(walked_rooms.mean())

        expected_cost = (
            expected_empty * spoilage_cost_per_room
            + expected_walked * walk_cost_per_room
        )

        results.append({
            "buffer_k": k,
            "walk_probability": walk_probability,
            "expected_accepted_new": float(
                accepted_new.mean()
            ),
            "expected_empty_rooms": expected_empty,
            "expected_walked_rooms": expected_walked,
            "expected_total_cost": float(expected_cost),
        })

    feasible = [
        row for row in results
        if row["walk_probability"] <= max_walk_risk
    ]

    if feasible:
        recommended = min(
            feasible,
            key=lambda row: (
                row["expected_total_cost"],
                row["buffer_k"]
            )
        ).copy()

        recommended["feasible"] = True

    else:
        # Fallback is not a risk-compliant policy.
        recommended = results[0].copy()
        recommended["feasible"] = False

    recommended["predicted_new_demand"] = float(
        predicted_new_demand
    )

    recommended["future_cancel_rate"] = float(
        future_cancel_rate
    )

    return recommended, results


if __name__ == "__main__":

    # Illustrative test inputs, not actual hotel
    # cancellation predictions.
    example_probs = np.full(22, 0.35)

    recommendation, scenarios = (
        calculate_demand_aware_buffer(
            cancellation_probs=example_probs,
            capacity=156,
            predicted_new_demand=19.4,
            max_walk_risk=0.05,
            max_buffer=100,
            simulations=10000,
            seed=42,
            adr=102.21,
            contribution_margin=0.70,
            walk_cost_multiplier=2.0,
        )
    )

    print("\n=== DEMAND-AWARE SIMULATION ===")
    print(
        "Recommended Additional Booking Limit:",
        recommendation["buffer_k"]
    )
    print(
        "Expected Accepted New Bookings:",
        round(
            recommendation["expected_accepted_new"], 2
        )
    )
    print(
        "Walk Risk:",
        f'{recommendation["walk_probability"]:.2%}'
    )
    print(
        "Expected Total Cost:",
        f'${recommendation["expected_total_cost"]:,.2f}'
    )
    print(
        "Risk Constraint Feasible:",
        recommendation["feasible"]
    )
