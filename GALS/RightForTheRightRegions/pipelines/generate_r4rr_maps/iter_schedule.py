import math


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _round_to_multiple(value, multiple):
    return int(multiple * round(float(value) / float(multiple)))


def _ceil_to_multiple(value, multiple):
    return int(multiple * math.ceil(float(value) / float(multiple)))


def count_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def compute_powerlaw_schedule(
    n_train,
    b_eff,
    i_min=8000,
    i_max=100000,
    s_ref=12500,
    i_ref=30000,
    alpha=0.4,
    ensure_full_epoch=True,
):
    if n_train <= 0:
        raise ValueError(f"n_train must be > 0, got {n_train}")
    if b_eff <= 0:
        raise ValueError(f"b_eff must be > 0, got {b_eff}")

    steps_per_epoch = int(math.ceil(float(n_train) / float(b_eff)))
    raw_iters = float(i_ref) * ((float(steps_per_epoch) / float(s_ref)) ** float(alpha))
    clipped_iters = float(_clamp(raw_iters, i_min, i_max))
    max_iters = _round_to_multiple(clipped_iters, 500)
    max_iters = int(_clamp(max_iters, i_min, i_max))

    # Guarantee at least one full pass whenever possible under the global cap.
    if ensure_full_epoch:
        coverage_floor = _ceil_to_multiple(steps_per_epoch, 500)
        if coverage_floor <= i_max:
            max_iters = max(max_iters, coverage_floor)

    cam_iters = int(_clamp(round(0.08 * max_iters), 600, 5000))
    eval_iters = int(_clamp(round(max_iters / 10.0), 500, 4000))

    return {
        "n_train": int(n_train),
        "b_eff": int(b_eff),
        "steps_per_epoch": steps_per_epoch,
        "raw_iters": raw_iters,
        "clipped_iters": clipped_iters,
        "max_iters": int(max_iters),
        "cam_iters": cam_iters,
        "eval_iters": eval_iters,
        "coverage_guaranteed": bool(max_iters >= steps_per_epoch),
    }
