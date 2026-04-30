import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def setup_axis(ax, robot, title=None):
    reach = robot.l1 + robot.l2

    ax.set_xlim(-reach - 0.5, reach + 0.5)
    ax.set_ylim(-reach - 0.5, reach + 0.5)
    ax.set_aspect("equal")
    ax.grid(True)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if title is not None:
        ax.set_title(title)


def plot_robot_pose(ax, robot, theta, label=None, alpha=1.0):
    base, elbow, ee = robot.joint_positions(theta)

    ax.plot(
        [base[0], elbow[0]],
        [base[1], elbow[1]],
        linewidth=4,
        alpha=alpha,
    )

    ax.plot(
        [elbow[0], ee[0]],
        [elbow[1], ee[1]],
        linewidth=4,
        alpha=alpha,
    )

    ax.scatter(
        [base[0], elbow[0], ee[0]],
        [base[1], elbow[1], ee[1]],
        s=40,
        alpha=alpha,
    )

    if label is not None:
        ax.text(ee[0], ee[1], label)


def plot_snapshots(
    robot,
    history,
    goal=None,
    num_snapshots=6,
    save_path=None,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axis(ax, robot, title="Robot Motion Snapshots")

    theta_history = history["theta"]
    ee_positions = np.array(history["ee"])

    ax.plot(
        ee_positions[:, 0],
        ee_positions[:, 1],
        linestyle="--",
        linewidth=2,
        label="End-effector path",
    )

    snapshot_indices = np.linspace(
        0,
        len(theta_history) - 1,
        num_snapshots,
        dtype=int,
    )

    for count, idx in enumerate(snapshot_indices):
        alpha = 0.35 + 0.65 * (count / max(1, len(snapshot_indices) - 1))
        plot_robot_pose(
            ax,
            robot,
            theta_history[idx],
            label=f"t{idx}",
            alpha=alpha,
        )

    if goal is not None:
        goal = np.array(goal)
        ax.scatter(goal[0], goal[1], marker="*", s=180, label="Goal")

    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_error(history, dt=0.05, save_path=None):
    error_norm = np.array(history["error_norm"])
    time = np.arange(len(error_norm)) * dt

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time, error_norm, linewidth=2)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("End-effector error norm")
    ax.set_title("Tracking Error Over Time")
    ax.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def animate_robot_motion(
    robot,
    history,
    goal=None,
    dt=0.05,
    save_path=None,
):
    theta_history = history["theta"]

    fig, ax = plt.subplots(figsize=(7, 7))
    setup_axis(ax, robot, title="RR Robot Animation")

    link1_line, = ax.plot([], [], linewidth=4)
    link2_line, = ax.plot([], [], linewidth=4)
    joints_line, = ax.plot([], [], "o", markersize=7)
    trace_line, = ax.plot([], [], "--", linewidth=2)

    if goal is not None:
        goal = np.array(goal)
        ax.scatter(goal[0], goal[1], marker="*", s=180, label="Goal")
        ax.legend()

    ee_trace = []

    def init():
        link1_line.set_data([], [])
        link2_line.set_data([], [])
        joints_line.set_data([], [])
        trace_line.set_data([], [])
        return link1_line, link2_line, joints_line, trace_line

    def update(frame):
        theta = theta_history[frame]
        base, elbow, ee = robot.joint_positions(theta)

        link1_line.set_data(
            [base[0], elbow[0]],
            [base[1], elbow[1]],
        )

        link2_line.set_data(
            [elbow[0], ee[0]],
            [elbow[1], ee[1]],
        )

        joints_line.set_data(
            [base[0], elbow[0], ee[0]],
            [base[1], elbow[1], ee[1]],
        )

        ee_trace.append(ee.copy())
        trace = np.array(ee_trace)

        trace_line.set_data(trace[:, 0], trace[:, 1])

        return link1_line, link2_line, joints_line, trace_line

    animation = FuncAnimation(
        fig,
        update,
        frames=len(theta_history),
        init_func=init,
        interval=dt * 1000,
        blit=True,
    )

    if save_path is not None:
        animation.save(save_path, fps=int(1 / dt))

    plt.show()

    return animation