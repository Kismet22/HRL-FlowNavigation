# HRL-FlowNavigation
## Overview
This repository is the official implementation of the article **"Navigation and obstacle avoidance in complex vortical flows via hierarchical reinforcement learning"** (DOI Link). The goal of this work is to provide a robust and scalable solution for autonomous robot control and navigation in highly unsteady fluid environments. By introducing a **goal-conditioned hierarchical reinforcement learning (HRL)** framework, we address the inherent difficulty of balancing long-horizon strategic planning with short-horizon reactive control under stochastic flow disturbances and strong fluid–structure interactions.
Through this hierarchical decomposition, we have successfully developed:
### A Robust Low-Level Controller
Capable of internalizing complex fluid dynamics to achieve precise force compensation. It ensures stable hovering and accurate path tracking by neutralizing instantaneous flow perturbations, providing a reliable physical foundation for the entire system.
### An Efficient High-Level Planner
Capable of generating adaptive sub-goals and explicit directional intents for safe transit. Even when relying solely on local sensory information, the planner effectively navigates long distances and bypasses obstacles, exhibiting a level of decisiveness and consistency that traditional end-to-end approaches fail to achieve.


### Core Contributions

1. **Hovering Task**  
   In this task, the controller maintains a stationary position in a fluid flow environment. The low-level controller is responsible for stable control, reacting to disturbances in real-time to ensure stability in turbulent conditions.  
   **Effectiveness & Advantages**: The low-level controller provides consistent stability even under unsteady conditions, highlighting its capability for reliable reactive control in dynamic environments.

2. **Path Tracking Task**  
   In this task, the controller follows a predefined path in a fluid flow, adjusting its movement based on real-time flow conditions. The low-level controller ensures precise trajectory control, compensating for disturbances in the flow and maintaining accurate path tracking.  
   **Effectiveness & Advantages**: The low-level controller efficiently handles path tracking in complex environments, demonstrating robust short-horizon control capabilities.

3. **Navigation & Obstacle Avoidance**  
   This task involves both the low-level controller and the high-level planner working together. The low-level controller manages fine-grained control tasks, such as obstacle avoidance and maintaining the robot's position, while the high-level planner provides strategic guidance for long-term navigation. The hierarchical structure effectively decouples control and planning tasks, optimizing performance.  
   **Effectiveness & Advantages**: The combined efforts of the low-level controller and high-level planner ensure efficient navigation and obstacle avoidance. The hierarchical framework allows for clear separation of reactive control and strategic planning, resulting in more flexible and robust performance in complex environments.


