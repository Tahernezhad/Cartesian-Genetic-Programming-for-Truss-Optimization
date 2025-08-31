# Cartesian Genetic Programming for Truss Optimization

<p align="center">
  <img src="assets/Node_Edge_CGP_best_devo.gif" alt="Development Process of the Truss Structure">
</p>

This repository contains the Python implementation for the paper: **[EvoDevo: Bioinspired Generative Design via Evolutionary Graph-Based Development](https://doi.org/10.3390/a18080467)**. It specifically focuses on the Cartesian Genetic Programming (CGP) methods discussed in the paper for optimising the design of truss structures.

## 📜 About the Project

This work presents a bio-inspired generative design algorithm that uses the concept of evolutionary development (EvoDevo). Instead of directly optimising a design, this approach evolves a set of reusable developmental rules.

The core of this system is an artificial Gene Regulatory Network (GRN), which acts as a controller within simple entities called "cells". For a truss structure, these cells represent the **nodes** (vertices) and **edges** (members). Each cell's GRN senses its local environment (e.g., strain energy, volume) and outputs a "growth" command, such as moving a node or changing an edge's cross-sectional area.

This repository implements the **CGP-based GRN**, which offers more interpretable, "white-box" outputs compared to neural network alternatives like GNNs. The goal is to evolve a controller that can effectively optimise a truss structure over a series of developmental steps.

## ✨ Key Features

* **Evolutionary Development (EvoDevo)**: An indirect approach to design where the "designer" (the GRN) is evolved, not the design itself.
* **Cartesian Genetic Programming (CGP)**: A graph-based evolutionary algorithm used to create readable and efficient GRN controllers.
* **Cellular Representation**: The truss is broken down into node and edge cells, each with its own controller that makes local decisions to achieve a global objective.
* **Multiple Growth Mechanisms**: The system can optimise trusses by:
    * Adjusting the cross-sectional area of edges (`edge-only` method).
    * Moving the coordinates of nodes (`node-only` method).
    * Doing both simultaneously (`node-edge` method).
* **Fitness Function**: The evolutionary algorithm optimises controllers based on a fitness score that combines the total strain energy and total volume of the truss.

## 🔧 Getting Started

### Prerequisites

This project uses **Conda** to manage its environment and dependencies. You'll need to have Anaconda or Miniconda installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Tahernezhad/Cartesian-Genetic-Programming-for-Truss-Optimization.git](https://github.com/Tahernezhad/Cartesian-Genetic-Programming-for-Truss-Optimization.git)
    cd Cartesian-Genetic-Programming-for-Truss-Optimization
    ```

2.  **Create the Conda environment:**
    Use the provided `environment.yml` file to create the Conda environment. This will install all the necessary packages and dependencies.
    ```bash
    conda env create -f environment.yml
    ```

3.  **Activate the environment:**
    ```bash
    conda activate cgp
    ```

### Running the Experiment

To run the evolutionary algorithm, execute the `main.py` script:

```bash
python main.py
