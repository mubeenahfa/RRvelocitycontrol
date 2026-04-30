import numpy as np


def simulate_controller(
    robot,
    theta0,
    goal,
    Kp=1.0,
    dt=0.05,
    max_steps=1000,
    tolerance=0.02,
    max_ee_velocity=2.0,
):
    theta = np.array(theta0, dtype=float)
    goal = np.array(goal, dtype=float)

    history = {
        "theta": [],
        "theta_dot": [],
        "ee": [],
        "error": [],
        "error_norm": [],
        "singular": [],
        "joint_limit_violation": [],
    }

    stopped_reason = "max_steps_reached"

    for _ in range(max_steps):
        ee = robot.forward_kinematics(theta)
        error = goal - ee
        error_norm = np.linalg.norm(error)

        singular = robot.is_singular(theta)
        inside_limits = robot.check_joint_limits(theta)

        history["theta"].append(theta.copy())
        history["ee"].append(ee.copy())
        history["error"].append(error.copy())
        history["error_norm"].append(error_norm)
        history["singular"].append(singular)
        history["joint_limit_violation"].append(not inside_limits)

        if error_norm < tolerance:
            stopped_reason = "goal_reached"
            break

        if singular:
            stopped_reason = "singularity_detected"
            break

        if not inside_limits:
            stopped_reason = "joint_limit_violation"
            break

        v = Kp * error

        v_norm = np.linalg.norm(v)
        if v_norm > max_ee_velocity:
            v = (v / v_norm) * max_ee_velocity

        J = robot.jacobian(theta)

        theta_dot = np.linalg.inv(J) @ v

        history["theta_dot"].append(theta_dot.copy())

        theta = theta + theta_dot * dt
        theta = robot.clamp_to_joint_limits(theta)

    history["stopped_reason"] = stopped_reason
    history["final_theta"] = theta.copy()
    history["final_error"] = history["error_norm"][-1]

    return history