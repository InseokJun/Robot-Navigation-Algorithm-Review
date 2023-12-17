# Particle Filter Algorithm

Welcome to the Particle Filter Algorithm directory within the Localization section of the Robot Navigation Algorithm repository! 🤖 Here, we delve into the principles and workings of the Particle Filter algorithm, a powerful tool for estimating the state of a dynamic system.

## Principle of the Particle Filter Algorithm

Particle Filter, often referred to as a Monte Carlo Localization method, describes the probability distribution of a non-linear system using entities known as 'Particles.' These particles represent the entire system's probability distribution and state.

### The system evolves through two main processes

**Prediction:**
1. Initialization: Start by initializing a predefined number of particles, denoted as N.
2. System Model: Utilize the system model to predict the next state of each particle, incorporating a non-Gaussian random variable.

**Estimation:**
1. Weight Update: Update the weight of each particle based on the difference between the measured value and the predicted value for each particle.
2. Normalization: Normalize the weights to ensure they fall between 0 and 1.

**Resampling:**
1. Resample particles based on their weights. Particles with higher weights are more likely to be replicated in the resampling process.
2. Normalize the weights of the resampled particles to maintain consistency.

**Estimation (Final):**
1. Compute the estimated state by considering the weighted sum of the resampled particles.

## Directory Structure:

- **`ParticleFilter` Directory:**
  - `README.md`: Details about the Particle Filter algorithm, its implementation, and considerations.
  - `particle_filter.py`: Original Particle Filter algorithm implementation by Atsushi Sakai.
  - `particle_filter_review.py`: Line-by-line code review and analysis of the Particle Filter algorithm.
  - `particle_filter_Exp.py`: Experimental code based on the Particle Filter algorithm for various scenarios.

## How to Use:

1. Navigate to the `ParticleFilter` directory.
2. Explore the README to understand the algorithm, its principles, and how it's implemented.
3. Examine the `particle_filter.py` script for the original code by Atsushi Sakai.
4. Check the `particle_filter_review.py` for a detailed review and analysis of the algorithm.
5. Experiment with different scenarios using the `particle_filter.py_Exp` script.

## Contribution and Feedback:

- 💡 Have ideas or suggestions for improvement? Open an issue to start a discussion.
- 🐞 Found a bug or an enhancement? Submit a pull request.
- 🤝 Open to collaborations and contributions from the community. Let's enhance our understanding of Particle Filter together!

---

**Note: If you find any errors or have suggestions for improvement, please feel free to contribute or provide feedback. This space is a collaborative effort to explore and advance Particle Filter algorithms in the field of robotics.**

**Original Code Author:**
Particle Filter localization sample by Atsushi Sakai (@Atsushi_twi)
