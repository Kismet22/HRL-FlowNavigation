# HRL-FlowNavigation

## Overview

Autonomous navigation in fluid environments is fundamentally challenged by unsteady flow disturbances, strong fluid–structure coupling, and limited sensory observability. These factors tightly couple rapid stabilization with long-horizon decision making, making conventional control and end-to-end reinforcement learning approaches difficult to generalize across dynamically complex scenarios.

This repository provides the official implementation of the paper:

**Navigation and obstacle avoidance in complex vortical flows via hierarchical reinforcement learning**  
*(DOI Link)*

Inspired by the hierarchical organization observed in biological locomotion, this work introduces a **goal-conditioned hierarchical reinforcement learning (HRL)** framework that explicitly separates long-horizon strategic planning from short-horizon dynamical control. The framework enables autonomous agents to maintain physical stability while performing complex navigation tasks in highly unsteady flow environments.

---

## Key Features

### Robust Low-Level Fluid-Aware Controller

The low-level controller learns fluid–structure interaction dynamics and generates dynamically feasible control forces. It enables:

- Stable hovering under strong vortex disturbances  
- Accurate trajectory tracking in time-varying flows  
- Real-time rejection of instantaneous flow perturbations  

---

### Adaptive High-Level Strategic Planner

The high-level policy generates flow-aware directional sub-goals using partial sensory observations. It enables:

- Long-distance autonomous navigation  
- Safe obstacle avoidance under limited perception  
- Consistent directional intent generation under stochastic disturbances  

Compared with end-to-end RL policies, the hierarchical strategy exhibits improved stability, interpretability, and generalization capability.

---

## Demonstration Videos

### Hovering Under Unsteady Flow
Demonstrates disturbance rejection and stable hovering in vortical flow fields.

(Video Link)

---

### Trajectory Tracking in Time-Varying Flow
Shows precise path-following performance while compensating for dynamic flow perturbations.

(Video Link)

---

### Autonomous Navigation with Obstacle Avoidance
Demonstrates long-range navigation with simultaneous obstacle avoidance in complex flow environments.

(Video Link)

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





