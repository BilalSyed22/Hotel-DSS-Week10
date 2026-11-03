
import numpy as np
import pandas as pd

from demand_aware import calculate_demand_aware_buffer

CAPACITY = 156
MAX_BUFFER = 150
SIMULATIONS = 10000

scenarios = [
    {
        "name": "Low Demand",
        "current": 22,
        "future": 19.4,
    },
    {
        "name": "Moderate Demand",
        "current": 110,
        "future": 45,
    },
    {
        "name": "High Demand",
        "current": 145,
        "future": 65,
    },
]

all_results = []

for scenario in scenarios:

    # Illustrative cancellation assumption.
    probabilities = np.full(
        scenario["current"],
        0.08
    )

    recommended, results = calculate_demand_aware_buffer(
        cancellation_probs=probabilities,
        capacity=CAPACITY,
        predicted_new_demand=scenario["future"],
        future_cancel_rate=0.08,
        max_walk_risk=0.05,
        max_buffer=MAX_BUFFER,
        simulations=SIMULATIONS,
        seed=42,
        adr=150,
        contribution_margin=0.70,
        walk_cost_multiplier=2.0,
    )

    results_by_k = {
        row["buffer_k"]: row
        for row in results
    }

    policies = {
        "Stop New Bookings": 0,
        "No Overbooking": max(
            0, CAPACITY - scenario["current"]
        ),
        "Fixed Overbooking (+5)": max(
            0, CAPACITY + 5 - scenario["current"]
        ),
        "Demand-Aware": int(
            recommended["buffer_k"]
        ),
    }

    for policy_name, k in policies.items():
        result = results_by_k[k]

        all_results.append({
            "Scenario": scenario["name"],
            "Policy": policy_name,
            "Additional Limit": k,
            "Expected Accepted": round(
                result["expected_accepted_new"], 2
            ),
            "Walk Risk (%)": round(
                result["walk_probability"] * 100, 2
            ),
            "Expected Cost ($)": round(
                result["expected_total_cost"], 2
            ),
            "Risk Compliant": (
                result["walk_probability"] <= 0.05
            ),
        })

comparison = pd.DataFrame(all_results)

print("\n=== HIGH-DEMAND STRESS TEST ===")
print(comparison.to_string(index=False))

print("\n=== DEMAND-AWARE RECOMMENDATIONS ===")
print(
    comparison[
        comparison["Policy"] == "Demand-Aware"
    ].to_string(index=False)
)

print("\nStress test completed.")


# ============================================
# WEEK 11 — SENSITIVITY ANALYSIS
# ============================================

print("\n=== SENSITIVITY ANALYSIS ===")

sensitivity_rows = []

high_demand_probs = np.full(145, 0.08)

for risk_limit in [0.01, 0.03, 0.05, 0.10]:
    for walk_multiplier in [1.0, 2.0, 3.0, 5.0]:

        rec, _ = calculate_demand_aware_buffer(
            cancellation_probs=high_demand_probs,
            capacity=156,
            predicted_new_demand=65,
            future_cancel_rate=0.08,
            max_walk_risk=risk_limit,
            max_buffer=100,
            simulations=10000,
            seed=42,
            adr=150,
            contribution_margin=0.70,
            walk_cost_multiplier=walk_multiplier,
        )

        sensitivity_rows.append({
            "Risk Limit (%)": risk_limit * 100,
            "Walk Cost (x ADR)": walk_multiplier,
            "Additional Limit": rec["buffer_k"],
            "Walk Risk (%)": round(
                rec["walk_probability"] * 100, 2
            ),
            "Expected Cost ($)": round(
                rec["expected_total_cost"], 2
            ),
            "Feasible": rec["feasible"],
        })

sensitivity = pd.DataFrame(sensitivity_rows)

print(sensitivity.to_string(index=False))


# ============================================
# WEEK 11 — MONTE CARLO ROBUSTNESS CHECK
# ============================================

print("\n=== MONTE CARLO ROBUSTNESS CHECK ===")

robustness_rows = []

for seed in [11, 21, 42, 88, 123]:

    rec, _ = calculate_demand_aware_buffer(
        cancellation_probs=np.full(145, 0.08),
        capacity=156,
        predicted_new_demand=65,
        future_cancel_rate=0.08,
        max_walk_risk=0.05,
        max_buffer=100,
        simulations=10000,
        seed=seed,
        adr=150,
        contribution_margin=0.70,
        walk_cost_multiplier=2.0,
    )

    robustness_rows.append({
        "Seed": seed,
        "Recommended Limit": rec["buffer_k"],
        "Walk Risk (%)": round(
            rec["walk_probability"] * 100, 3
        ),
        "Expected Cost ($)": round(
            rec["expected_total_cost"], 2
        ),
        "Feasible": rec["feasible"],
    })

robustness = pd.DataFrame(robustness_rows)

print(robustness.to_string(index=False))

print("\n=== ROBUSTNESS SUMMARY ===")

print(
    "Recommended Limit Range:",
    robustness["Recommended Limit"].min(),
    "to",
    robustness["Recommended Limit"].max()
)

print(
    "Walk Risk Range:",
    robustness["Walk Risk (%)"].min(),
    "to",
    robustness["Walk Risk (%)"].max()
)

print(
    "All Feasible:",
    robustness["Feasible"].all()
)
