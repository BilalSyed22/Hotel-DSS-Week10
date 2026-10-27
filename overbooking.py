
import numpy as np


def calculate_dynamic_buffer(
    cancellation_probs,
    capacity,
    max_walk_risk=0.05,
    max_buffer=100,
    simulations=10000,
    seed=42,
    adr=150.0,
    contribution_margin=0.70,
    walk_cost_multiplier=2.0
):
    p = np.asarray(cancellation_probs, dtype=float)

    if not np.all(np.isfinite(p)):
        raise ValueError("Invalid probabilities")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    if capacity <= 0 or simulations <= 0 or max_buffer < 0:
        raise ValueError("Invalid simulation settings")
    if not 0 <= max_walk_risk <= 1:
        raise ValueError("Invalid walk risk limit")
    if adr < 0 or not 0 <= contribution_margin <= 1:
        raise ValueError("Invalid financial parameters")
    if walk_cost_multiplier < 0:
        raise ValueError("Invalid walk cost multiplier")

    rng = np.random.default_rng(seed)

    # Simulate existing booking arrivals once.
    arrivals = (
        rng.random((simulations, len(p))) >= p
    ).sum(axis=1)

    extra_cancel_risk = float(p.mean()) if len(p) else 0.0

    # Costs per room-night
    spoilage_cost = adr * contribution_margin
    walk_cost = adr * walk_cost_multiplier

    # Reuse the same extra-booking outcomes across k.
    extra_showups = (
        rng.random((simulations, max_buffer))
        >= extra_cancel_risk
    )
    cumulative_extra = np.concatenate(
        [
            np.zeros((simulations, 1), dtype=int),
            np.cumsum(extra_showups, axis=1)
        ],
        axis=1
    )

    results = []

    for k in range(max_buffer + 1):
        total_arrivals = arrivals + cumulative_extra[:, k]

        empty_rooms = np.maximum(capacity - total_arrivals, 0)
        walked_rooms = np.maximum(total_arrivals - capacity, 0)

        walk_probability = float(np.mean(walked_rooms > 0))
        expected_empty = float(np.mean(empty_rooms))
        expected_walked = float(np.mean(walked_rooms))

        expected_spoilage_cost = expected_empty * spoilage_cost
        expected_walk_cost = expected_walked * walk_cost

        results.append({
            "buffer_k": k,
            "walk_probability": walk_probability,
            "expected_empty_rooms": expected_empty,
            "expected_walked_rooms": expected_walked,
            "expected_spoilage_cost": expected_spoilage_cost,
            "expected_walk_cost": expected_walk_cost,
            "expected_total_cost": (
                expected_spoilage_cost + expected_walk_cost
            )
        })

    eligible = [
        r for r in results
        if r["walk_probability"] <= max_walk_risk
    ]

    # Minimize expected cost among feasible choices.
    recommended = min(
        eligible,
        key=lambda r: (r["expected_total_cost"], r["buffer_k"])
    ) if eligible else results[0]

    return recommended, results
