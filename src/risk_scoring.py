from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskScore:
    probability: float
    level: str
    action: str


def classify_risk(probability: float) -> RiskScore:
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be in range [0, 1].")

    if probability <= 0.30:
        return RiskScore(probability=probability, level="Low Risk", action="Allow")
    if probability <= 0.70:
        return RiskScore(
            probability=probability,
            level="Medium Risk",
            action="Require additional verification",
        )
    return RiskScore(
        probability=probability,
        level="High Risk",
        action="Block or manual review",
    )

