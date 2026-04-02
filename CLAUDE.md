# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Five-link leg-wheel robot simulation using MuJoCo. The robot (GBC486) has two legs with a five-bar linkage mechanism and two wheels. Control is implemented via Virtual Model Control (VMC) for leg kinematics and PID controllers for leg length, body posture, and position.

## Setup & Run

```bash
# Environment: Python 3.10
pip install -r requirements.txt   # mujoco, numpy, pynput

# Run simulation (launches MuJoCo viewer, ESC to exit)
python3 Simulation.py
```

## Architecture

**Simulation.py** — Main loop. Configures PID parameters, runs the control loop at different frequencies:
- Sensor read: every 1ms (`t1`)
- Control calc: every 4ms (`t2`)
- Print/keyboard: every 20ms (`t3`)

**environment.py** (`LegWheelRobot`) — MuJoCo wrapper. Loads MJCF model, reads sensors (IMU, joint encoders, wheel encoders), computes odometry and body state, sets actuator torques. Actuator order: joints [右前, 右后, 左前, 左后], wheels [右, 左].

**VMC.py** (`leg_VMC`) — Virtual Model Control for the five-bar linkage. Converts between joint space (phi1, phi4) and virtual leg space (L0 = leg length, theta = leg angle). Computes forward kinematics and Jacobian-based torque mapping: virtual forces (F0, Tp) → joint torques (T1, T4). Link lengths: L1=L4=0.1m, L2=L3=0.1m, L5=0.12m.

**Controller.py** (`PID`) — PID controller with integral and output limiting.

**caculation.py** — Utility: quaternion-to-Euler conversion.

**keyboard.py** (`KeyboardController`) — Keyboard input via pynput for runtime commands.

**MJCF/** — MuJoCo model files. `env.xml` is the entry point (includes `robot.xml` and `axis.xml`). STL meshes for each link. Model originally converted from URDF (`GBC486.SLDASM.urdf`).

## Control Flow

```
Sensor data → VMC forward kinematics (joint angles → L0, theta)
           → PID controllers compute virtual forces (F0, Tp)
           → VMC inverse (Jacobian transpose) maps to joint torques
           → Wheel PID (position + velocity) computes wheel torques
           → Apply all torques to actuators
```

## Conventions

- Right/Left legs use separate VMC instances with mirrored angle conventions (note sign flips on pitch/gyro for left leg).
- Joint angle offsets are hardcoded in `sensor_read_data()` to align simulation joint zeros with the kinematic model.
- Language: code comments and variable names mix Chinese and English.
