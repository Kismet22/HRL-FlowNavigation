# HRL-FlowNavigation
## Overview
This repository is the official implementation of the article **"Navigation and obstacle avoidance in complex vortical flows via hierarchical reinforcement learning"** (DOI Link). The goal of this work is to provide a robust and scalable solution for autonomous robot control and navigation in highly unsteady fluid environments. By introducing a **goal-conditioned hierarchical reinforcement learning (HRL)** framework, we address the inherent difficulty of balancing long-horizon strategic planning with short-horizon reactive control under stochastic flow disturbances and strong fluid–structure interactions.
Through this hierarchical decomposition, we have successfully developed:
### A Robust Low-Level Controller
Capable of internalizing complex fluid dynamics to achieve precise force compensation. It ensures stable hovering and accurate path tracking by neutralizing instantaneous flow perturbations, providing a reliable physical foundation for the entire system.
### An Efficient High-Level Planner
Capable of generating adaptive sub-goals and explicit directional intents for safe transit. Even when relying solely on local sensory information, the planner effectively navigates long distances and bypasses obstacles, exhibiting a level of decisiveness and consistency that traditional end-to-end approaches fail to achieve.



