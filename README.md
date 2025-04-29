# PSO-ANN-Training

## Overview
This project implements a Particle Swarm Optimization (PSO) algorithm for training an Artificial Neural Network (ANN) model. The PSO algorithm is used to optimize the weights of the ANN, allowing it to learn from the training data and make predictions on new data.

## Project Structure
The project is organized as follows:

```
PSO-ANN-Training
├── src
│   ├── Main.m               # Main entry point for the application
│   ├── psoTrainANN.m        # Implements the PSO algorithm for training the ANN
│   ├── predict.m            # Function for making predictions with the trained ANN
│   ├── nnCostFunction.m     # Defines the cost function for the ANN
│   ├── generatePopulation.m  # Generates initial population of particles for PSO
│   ├── utils
│   │   ├── mutation.m       # Applies mutation to particles in PSO
│   │   ├── crossover.m       # Implements crossover operation for particles
│   │   └── tournament_selection.m # Selects particles based on fitness scores
├── data
│   ├── iris_training.data    # Training dataset for the ANN
│   └── iris_testing.data     # Testing dataset for evaluating model performance
└── README.md                 # Documentation for the project
```

## Setup Instructions
1. Ensure you have Octave installed on your machine.
2. Download or clone the repository to your local machine.
3. Navigate to the project directory.

## Usage Guidelines
1. Place your training and testing data in the `data` directory.
2. Modify the parameters in `Main.m` as needed to suit your dataset and model requirements.
3. Run `Main.m` to start the training process using the PSO algorithm.

## Particle Swarm Optimization (PSO)
PSO is a computational method that optimizes a problem by iteratively improving candidate solutions with regard to a given measure of quality. In this project, PSO is used to optimize the weights of the ANN, allowing it to learn from the training data effectively.

## Acknowledgments
This project is inspired by various machine learning and optimization techniques. Special thanks to the contributors of the PSO algorithm and ANN literature.