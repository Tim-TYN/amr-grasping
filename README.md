# AMR Grasping – Autonomous Object Detection and Manipulation Using the LIMO CoBot

This repository contains the full ROS-based implementation developed for the Master’s thesis  
**“Autonomous Object Detection and Manipulation Using a Mobile Cobot”**  
by **Tim Yago Nordhoff** (TH Köln, 2025).

## Overview
The project implements a **fully onboard autonomous pipeline** that enables a mobile cobot to:
- explore **unknown indoor environments**
- detect a **text-specified target object** (open-vocabulary)
- estimate its **3D position**
- navigate toward and **grasp the object autonomously**

All components are designed for real-time execution on embedded hardware.

## Repository Structure

```text
amr-grasping/
├── limo_explore/        # ROS package with exploration, coverage, and object finding nodes
│   ├── scripts/         # Core Python nodes (ObjectFinder, CamCoverage, planners, ...)
│   ├── launch/          # Launch files for the system
│   ├── CMakeLists.txt
│   └── package.xml
├── traj_imgs/           # Example and debug images of recorded trajectories
├── traj_logs/           # Logged robot trajectories (*.txt)
└── README.md
```

## Key Components
- **SLAM & Navigation**  
  EKF-based odometry fusion, `slam_gmapping`, `move_base`

- **Exploration**  
  Camera-based coverage mapping (`CamCoverage`) and exploration planners

- **Perception**  
  Open-vocabulary object detection using **NanoOWL**

- **Manipulation**  
  Deterministic grasp strategy (approach → align → close-in → grasp → lift)

## Platform
- **Robot**: LIMO Cobot (mobile base with robotic arm)
- **Sensors**: 2D LiDAR, RGB-D camera, IMU
- **Framework**: ROS (Noetic), fully onboard execution

