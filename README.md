# HRL-FlowNavigation

## Overview
This repository is the official implementation of the article **"Navigation and obstacle avoidance in complex vortical flows via hierarchical reinforcement learning"** (DOI Link). The goal of this work is to provide a robust and scalable solution for autonomous robot control and navigation in highly unsteady fluid environments. By introducing a **goal-conditioned hierarchical reinforcement learning (HRL)** framework, we address the inherent difficulty of balancing long-horizon strategic planning with short-horizon reactive control under stochastic flow disturbances and strong fluid–structure interactions.

Through this hierarchical decomposition, we have successfully developed:

### A Robust Low-Level Controller
Capable of internalizing complex fluid dynamics to achieve precise force compensation. It ensures stable hovering and accurate path tracking by neutralizing instantaneous flow perturbations, providing a reliable physical foundation for the entire system.

### An Efficient High-Level Planner
Capable of generating adaptive sub-goals and explicit directional intents for safe transit. Even when relying solely on local sensory information, the planner effectively navigates long distances and bypasses obstacles, exhibiting a level of decisiveness and consistency that traditional end-to-end approaches fail to achieve.

---

## Environment Configuration

The simulation environment relies on Lilypad CFD and requires several external dependencies.

---

### 1. Install XVFB (Virtual Display)

Lilypad requires a virtual display when running on headless servers.

```bash
sudo apt install xvfb
```

---

### 2. Install Processing 4.3

Lilypad CFD is built on the Processing framework. Please install **Processing 4.3**.

Download from the official website:

https://processing.org/download

After installation, set the environment variable:

```bash
export PROCESSING_PATH="/your/absolute/path/to/processing-4.3"
```

---

### 3. Configure Lilypad Conda Environment

Create the required Conda environment using the provided YAML file:

```bash
conda env create -f PATH/RL_Lilypad.yml -n NAME
```

Activate the environment:

```bash
conda activate NAME
```

---





