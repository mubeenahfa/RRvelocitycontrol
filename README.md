# Planar RR Robot Controller Simulation

This project simulates a 2-link planar RR robot using inverse Jacobian velocity control.  
The system automatically tunes the proportional gain \(K_p\), runs the simulation, and generates plots and an animation of the robot motion.

---

## Project Structure

robot.py Robot model (kinematics, Jacobian, singularity checks)
controller.py Controller + simulation loop
tuner.py Kp tuning and evaluation logic
plotter.py Plotting and animation utilities
main.py Main script to run everything
requirements.txt Dependencies


---

## Installation

Install required Python packages:

```bash
pip install -r requirements.txt

sudo apt install ffmpeg

python main.py
```


## What the Program Does

When you run main.py, it:

Creates the RR robot model
Defines initial joint angles and goal position
Searches over a range of Kp values
Selects the best Kp using a cost function
Runs a final simulation with the optimal Kp
Generates plots and an animation

## Outputs
tuning_results.csv
robot_snapshots.png
tracking_error.png
robot_animation.mp4

