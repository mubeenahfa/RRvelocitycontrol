import numpy as np
import pandas as pd

from robot import PlanarRRRobot
from controller import simulate_controller
from tuner import tune_kp
from plotter import plot_snapshots, plot_error, animate_robot_motion


def main():
    robot = PlanarRRRobot(
        l1=3.0,
        l2=2.0,
        theta1_limits=(-90, 90),
        theta2_limits=(-170, 170),
    )

    # im giving initial angles joints so its basically start position
    #theta0 = np.deg2rad([30, 60])
    theta0 = np.deg2rad([20, 40])

    # this is my goal position 
    goal = np.array([3.0, 2.0])
    #goal = np.array([2.0, 3.0])

    dt = 0.05
    max_steps = 1000
    tolerance = 0.02
    max_ee_velocity = 2.0

    # Candidate Kp values that i will search over to tune
    kp_values = np.logspace(-2, 2, 100)  

    best_result, tuning_results = tune_kp(
        robot=robot,
        theta0=theta0,
        goal=goal,
        kp_values=kp_values,
        dt=dt,
        max_steps=max_steps,
        tolerance=tolerance,
        max_ee_velocity=max_ee_velocity,
    )

    # all my tuning results are saved as a csv
    results_df = pd.DataFrame(tuning_results)
    print("\nTuning results:")
    print(results_df)
    print("\nBest Kp result:")
    print(best_result)
    best_kp = best_result["Kp"]
    results_df.to_csv("tuning_results.csv", index=False)

    print("\n==============================")
    print(" BEST KP FOUND")
    print("==============================")
    print(f"Kp: {best_result['Kp']:.4f}")
    print("==============================\n")

    # below here i run the final simulation using the best found kp value
    history = simulate_controller(
        robot=robot,
        theta0=theta0,
        goal=goal,
        Kp=best_kp,
        dt=dt,
        max_steps=max_steps,
        tolerance=tolerance,
        max_ee_velocity=max_ee_velocity,
    )

    print("\nFinal simulation:")
    print("Stopped reason:", history["stopped_reason"])
    print("Final error:", history["final_error"])
    print("Final theta [deg]:", np.rad2deg(history["final_theta"]))

    # this plots a snapshot of the robot
    plot_snapshots(
        robot=robot,
        history=history,
        goal=goal,
        num_snapshots=6,
        save_path="robot_snapshots.png",
    )

    # this plots the error curve 
    plot_error(
        history=history,
        dt=dt,
        save_path="tracking_error.png",
    )

    # this saves the whole movement animation
    animate_robot_motion(
        robot=robot,
        history=history,
        goal=goal,
        dt=dt,
        save_path="robot_animation.mp4",
    )


if __name__ == "__main__":
    main()