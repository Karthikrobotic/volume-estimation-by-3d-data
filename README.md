# 🚧 AutoPotholeFiller: End‑to‑End Pothole Detection & Robotic Filling

![Project Showcase](path/to/animated-demo.gif)

**AutoPotholeFiller** is an innovative system that uses deep learning, 3D vision, and robotic control to detect, measure, and autonomously fill potholes—just like a 3D‑printing robot for our roads!

---

## 🔍 Table of Contents

1. [🚀 Project Overview](#-project-overview)  
2. [🛠️ Key Features](#️-key-features)  
3. [📁 Repo Structure](#-repo-structure)  
4. [🎯 How It Works](#-how-it-works)  
5. [⚙️ Prerequisites & Installation](#️-prerequisites--installation)  
6. [▶️ Quick Start](#️-quick-start)  
7. [👨‍💻 Folder Details](#-folder-details)  
8. [🤝 Contributing & Contact](#-contributing--contact)  

---

## 🚀 Project Overview

Modern road maintenance is costly, dangerous, and time‑consuming. **AutoPotholeFiller** automates this process with a 2-bar robot arm and ZED2i stereo camera. The system:

1. **Detects** potholes in real time using YOLOv8  
2. **Estimates** their 3D volume from depth data  
3. **Projects** pothole contours into robot world coordinates  
4. **Executes** a precise fill routine with inverse kinematics  

---

## 🛠️ Key Features

- **Real‑Time Detection**: YOLOv8 for lightning‑fast live segmentation  
- **Accurate Volume Computation**: RGB‑D fusion with ZED2i  
- **Seamless Robot Integration**: 2‑bar manipulator IK for smooth trajectories  
- **Modular Codebase**: Clear folders for ML, vision, kinematics, and control  

---

## 📁 Repo Structure


---

## 🎯 How It Works

1. **Pothole Detection**  
   We train **YOLOv8** (for speed) and **Faster R‑CNN** (for accuracy) on annotated pothole images.  
   ➜ See detailed code & training logs in [`code/pothole_ml/`](code/pothole_detection/README.md)

2. **Volume Computation**  
   Capture synchronized RGB and depth frames with **ZED2i**. Extract 2D contours, lift them to 3D point‑clouds, and compute volume via mesh integration.  
   ➜ Full pipeline explained in [`code/pothole_volume_detection/README.md`](code/pothole_volume_detection/README.md)

3. **World‑Coordinate Projection**  
   Transform camera coordinates to the robot base frame using pre‑calibrated extrinsics. Generate a smooth “filling” path along the pothole contours—akin to 3D printing layers.  
   ➜ See math derivations & code in [`code/2bar_robot_link/README.md`](code/2bar_robot_link/README.md)

4. **Robotic Filling Execution**  
   Solve inverse kinematics for each waypoint using a custom IKFast‑inspired solver for our 2‑bar manipulator. Send joint trajectories to the UR5 controller to deposit “thar” fill material.  
   ➜ Detailed solver & control scripts in [`code/ik_solution/`](code/ik_solution/)

---

## ⚙️ Prerequisites & Installation

1. **Hardware**  
   - Design 2 door link robot
   - ZED2i stereo camera  
   - Pneumatic “thar” dispenser attachment  

2. **Software**  
   - Ubuntu 20.04 / ROS Noetic  
   - Python 3.8+ (+ OpenCV, PyTorch, NumPy)  
   - ZED SDK 3.7  
    

```bash
# Clone & set up workspace
git clone https://github.com/yourusername/AutoPotholeFiller.git
cd AutoPotholeFiller
./install_dependencies.sh
