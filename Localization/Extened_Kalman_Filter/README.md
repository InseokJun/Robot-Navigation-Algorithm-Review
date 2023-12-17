# Extended Kalman Filter (EKF) Algorithm

Welcome to the Extended Kalman Filter (EKF) Algorithm directory within the Localization section of the Robot Navigation Algorithm repository!     
In this section, we explore the principles and application of the EKF, a variant of the Kalman Filter designed for non-linear systems.🌐

## Extended Kalman Filter (EKF)

The Extended Kalman Filter (EKF) is a powerful tool for state estimation in non-linear systems.    
Unlike the traditional Kalman Filter, which is suitable for linear dynamic systems, the EKF excels in scenarios where the underlying system dynamics are non-linear.   
It achieves this by employing linearization techniques to iteratively perform prediction and update steps.

### Key Steps in EKF

1. **Prediction:**
    - The EKF predicts the next state based on the current estimated state, incorporating non-linearities in the system model.
  
2. **Update:**
    - The update step adjusts the predicted state using measurement information, resulting in a more accurate estimation.
  
3. **Linearization:**
    - Linearization is a crucial aspect of EKF, where the non-linear system model is approximated with a linear model at each iteration.
  
4. **Applications:**
    - EKF finds applications in various fields, including control systems, robotics, and signal processing. It particularly shines in solving state estimation problems in non-linear dynamic systems.

## Contents of the Directory:

- **`ExtendedKalmanFilter` Directory:**
  - `README.md`: Detailed information about the Extended Kalman Filter algorithm, its principles, and implementation.
  - `extended_kalman_filter.py`: Original implementation of the Extended Kalman Filter by Atsushi Sakai.
  - `extended_kalman_filter_review.py`: Implementation with detailed comments providing insights into each line of the original code.
  - `extended_kalman_filter_Exp.py`: Implementation showcasing various experiments based on the Extended Kalman Filter.

## How to Use:

1. Navigate to the `ExtendedKalmanFilter` directory.
2. Explore the README to understand the Extended Kalman Filter algorithm, its principles, and implementation.
3. Examine the `extended_kalman_filter.py` script for the original code by Atsushi Sakai.
4. Check `extended_kalman_filter_review.py` for an implementation with detailed comments providing insights into each line of the original code.
5. Explore `extended_kalman_filter_Exp.py` for an implementation showcasing various experiments based on the Extended Kalman Filter.

## Contribution and Feedback:

- 💡 Have ideas or suggestions for improvement? Open an issue to start a discussion.
- 🐞 Found a bug or an enhancement? Submit a pull request.
- 🤝 Open to collaborations and contributions from the community. Let's enhance our understanding of Extended Kalman Filter together!

---

**Note**  
If you find any errors or have suggestions for improvement, please feel free to contribute or provide feedback.     
This space is a collaborative effort to explore and advance Extended Kalman Filter algorithms in the field of robotics.

**Original Code Author**
Extended Kalman Filter localization sample by Atsushi Sakai (@Atsushi_twi)

**Key Knowledge**
- Jacobian
- Non Linear to Linear

