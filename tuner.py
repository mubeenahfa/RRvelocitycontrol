import numpy as np
from controller import simulate_controller


def evaluate_run(history, goal, dt, tolerance=0.02):
    goal = np.array(goal, dtype=float)

    ee_positions = np.array(history["ee"])
    error_norm = np.array(history["error_norm"])

    final_error = error_norm[-1]


    settled_indices = np.where(error_norm < tolerance)[0]

    if len(settled_indices) > 0:
        settling_step = settled_indices[0]
        settling_time = settling_step * dt
    else:
        settling_step = len(error_norm)
        settling_time = settling_step * dt


    start = ee_positions[0]
    path_vector = goal - start
    path_length = np.linalg.norm(path_vector)

    if path_length > 1e-8:
        path_unit = path_vector / path_length
        projections = (ee_positions - start) @ path_unit
        overshoot = max(0.0, np.max(projections) - path_length)
    else:
        overshoot = 0.0


    error_diff = np.diff(error_norm)

    if len(error_diff) > 1:
        oscillations = np.sum(np.diff(np.sign(error_diff)) != 0)
    else:
        oscillations = 0

    singularity_detected = history["stopped_reason"] == "singularity_detected"
    joint_limit_violation = history["stopped_reason"] == "joint_limit_violation"
    max_steps_reached = history["stopped_reason"] == "max_steps_reached"

    unstable = (
        np.any(np.isnan(error_norm))
        or np.any(np.isinf(error_norm))
        or error_norm[-1] > 2.0 * error_norm[0]
    )

    cost = (
        20.0 * final_error
        + 1.0 * settling_time
        + 10.0 * overshoot
        + 0.2 * oscillations
    )

    if singularity_detected:
        cost += 1000.0

    if joint_limit_violation:
        cost += 1000.0

    if max_steps_reached:
        cost += 200.0

    if unstable:
        cost += 1000.0

    return {
        "final_error": final_error,
        "settling_step": settling_step,
        "settling_time": settling_time,
        "overshoot": overshoot,
        "oscillations": oscillations,
        "singularity_detected": singularity_detected,
        "joint_limit_violation": joint_limit_violation,
        "max_steps_reached": max_steps_reached,
        "unstable": unstable,
        "cost": cost,
        "stopped_reason": history["stopped_reason"],
    }


def tune_kp(
    robot,
    theta0,
    goal,
    kp_values,
    dt=0.05,
    max_steps=1000,
    tolerance=0.02,
    max_ee_velocity=2.0,
):
    results = []

    for Kp in kp_values:
        history = simulate_controller(
            robot=robot,
            theta0=theta0,
            goal=goal,
            Kp=Kp,
            dt=dt,
            max_steps=max_steps,
            tolerance=tolerance,
            max_ee_velocity=max_ee_velocity,
        )

        metrics = evaluate_run(
            history=history,
            goal=goal,
            dt=dt,
            tolerance=tolerance,
        )

        metrics["Kp"] = Kp
        results.append(metrics)

    best_result = min(results, key=lambda item: item["cost"])

    return best_result, results