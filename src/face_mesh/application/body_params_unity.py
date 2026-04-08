def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm_beta(beta: float, scale: float = 18.0) -> float:
    return _clamp01(0.5 + scale * beta)


def _r4(x: float) -> float:
    return round(_clamp01(x), 4)


def betas_to_unity_body_params(betas: list[float]) -> dict[str, float]:
    b = list(betas[:10]) + [0.0] * max(0, 10 - len(betas))
    b = b[:10]
    n = [_norm_beta(x) for x in b]
    body_fat = _r4(0.35 * n[1] + 0.33 * n[2] + 0.32 * n[3])
    muscle_mass = _r4(
        0.42 * (1.0 - body_fat) + 0.22 * n[4] + 0.18 * n[5] + 0.18 * n[6]
    )
    shoulder_width = _r4(0.32 * n[2] + 0.33 * n[3] + 0.35 * n[4])
    chest_size = _r4(0.45 * n[1] + 0.30 * n[2] + 0.25 * n[0])
    waist_size = _r4(0.40 * n[1] + 0.35 * n[3] + 0.25 * n[2])
    hip_size = _r4(0.30 * n[2] + 0.35 * n[3] + 0.20 * n[7] + 0.15 * n[1])
    arm_thickness = _r4(0.45 * n[6] + 0.35 * n[7] + 0.20 * n[5])
    leg_thickness = _r4(0.40 * n[7] + 0.40 * n[8] + 0.20 * n[6])
    leg_length = _r4(0.55 * n[0] + 0.25 * n[8] + 0.20 * n[9])
    neck_length = _r4(0.50 * n[0] + 0.30 * n[9] + 0.20 * (1.0 - n[3]))
    return {
        "bodyFat": body_fat,
        "muscleMass": muscle_mass,
        "shoulderWidth": shoulder_width,
        "chestSize": chest_size,
        "waistSize": waist_size,
        "hipSize": hip_size,
        "armThickness": arm_thickness,
        "legThickness": leg_thickness,
        "legLength": leg_length,
        "neckLength": neck_length,
    }
