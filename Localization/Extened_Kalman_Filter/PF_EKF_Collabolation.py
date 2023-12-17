"""
[Common Library & Module Import]
"""

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math

import matplotlib.pyplot as plt
import numpy as np

from utils.angle import rot_mat_2d
from utils.plot import plot_covariance_ellipse

np.random.seed(1)

"""
[EKF Variable Declare]
"""

Q_ekf = np.diag([
    0.1,  
    0.1,  
    np.deg2rad(1.0),  
    1.0  
]) ** 2  

R_ekf = np.diag([1.0, 1.0]) ** 2  

INPUT_NOISE_ekf = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE_ekf = np.diag([0.5, 0.5]) ** 2

"""
[PF Variable Declare]
"""

Q_pf = np.diag([0.2]) ** 2  
R_pf = np.diag([2.0, np.deg2rad(40.0)]) ** 2

Q_sim_pf = np.diag([0.2]) ** 2
R_sim_pf = np.diag([1.0, np.deg2rad(30.0)]) ** 2

MAX_RANGE_pf = 20.0 
NP_pf = 100  
NTh_pf = NP_pf/ 2.0  

"""
[Common Variable Declare]
"""

DT = 0.1  
SIM_TIME = 50.0  

show_animation = True

"""
[Common Function]
"""

def calc_input():
    v = 1.0  
    yawrate = 0.1  
    u = np.array([[v], [yawrate]])
    return u

"""
[EKF Function Definition]
"""

def observation_ekf(xTrue, xd_ekf, u):
    xTrue = motion_model_ekf(xTrue, u)

    z_ekf = observation_model_ekf(xTrue) + GPS_NOISE_ekf @ np.random.randn(2, 1)

    ud_ekf = u + INPUT_NOISE_ekf @ np.random.randn(2, 1)

    xd_ekf = motion_model_ekf(xd_ekf, ud_ekf)

    return xTrue, z_ekf, xd_ekf, ud_ekf


def motion_model_ekf(x, u):

    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model_ekf(x):
    
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    z = H @ x

    return z


def jacob_f_ekf(x, u):

    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h_ekf():

    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation_ekf(xEst, PEst, z, u):

    xPred = motion_model_ekf(xEst, u)
    jF = jacob_f_ekf(xEst, u)
    PPred = jF @ PEst @ jF.T + Q_ekf

    jH = jacob_h_ekf()
    zPred = observation_model_ekf(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R_ekf
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst

"""
[PF Function Definition]
"""

def observation_pf(x_true, xd_pf, u, rf_id_pf):
    x_true = motion_model_pf(x_true, u)

    z_pf = np.zeros((0, 3))

    for i in range(len(rf_id_pf[:, 0])):

        dx_pf = x_true[0, 0] - rf_id_pf[i, 0]
        dy_pf = x_true[1, 0] - rf_id_pf[i, 1]
        d_pf = math.hypot(dx_pf, dy_pf)
        if d_pf <= MAX_RANGE_pf:
            dn_pf = d_pf + np.random.randn() * Q_sim_pf[0, 0] ** 0.5  # add noise
            zi_pf = np.array([[dn_pf, rf_id_pf[i, 0], rf_id_pf[i, 1]]])
            z_pf = np.vstack((z_pf, zi_pf))

    ud1_pf = u[0, 0] + np.random.randn() * R_sim_pf[0, 0] ** 0.5
    ud2_pf = u[1, 0] + np.random.randn() * R_sim_pf[1, 1] ** 0.5
    ud_pf = np.array([[ud1_pf, ud2_pf]]).T

    xd_pf = motion_model_pf(xd_pf, ud_pf)

    return x_true, z_pf, xd_pf, ud_pf


def motion_model_pf(x, u):
    
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x_pf = F.dot(x) + B.dot(u)

    return x_pf


def gauss_likelihood_pf(x, sigma):
    p_pf = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p_pf


def calc_covariance_pf(x_est_pf, px_pf, pw_pf):

    cov_pf = np.zeros((3, 3))
    n_particle_pf = px_pf.shape[1]
    for i in range(n_particle_pf):
        dx_pf = (px_pf[:, i:i + 1] - x_est_pf)[0:3]
        cov_pf += pw_pf[0, i] * dx_pf @ dx_pf.T
    cov_pf *= 1.0 / (1.0 - pw_pf @ pw_pf.T)

    return cov_pf


def pf_localization_pf(px_pf, pw_pf, z_pf, u):

    for ip in range(NP_pf):
        x_pf = np.array([px_pf[:, ip]]).T
        w_pf = pw_pf[0, ip]

        ud1_pf = u[0, 0] + np.random.randn() * R_pf[0, 0] ** 0.5
        ud2_pf = u[1, 0] + np.random.randn() * R_pf[1, 1] ** 0.5
        ud_pf = np.array([[ud1_pf, ud2_pf]]).T
        x_pf = motion_model_pf(x_pf, ud_pf)

        for i in range(len(z_pf[:, 0])):
            dx_pf = x_pf[0, 0] - z_pf[i, 1]
            dy_pf = x_pf[1, 0] - z_pf[i, 2]
            pre_z_pf = math.hypot(dx_pf, dy_pf)
            dz_pf = pre_z_pf - z_pf[i, 0]
            w_pf = w_pf * gauss_likelihood_pf(dz_pf, math.sqrt(Q_pf[0, 0]))
       
        px_pf[:, ip] = x_pf[:, 0]
        pw_pf[0, ip] = w_pf

    pw_pf = pw_pf / pw_pf.sum()  

    x_est_pf = px_pf.dot(pw_pf.T)
    p_est_pf = calc_covariance_pf(x_est_pf, px_pf, pw_pf)

    N_eff_pf = 1.0 / (pw_pf.dot(pw_pf.T))[0, 0] 
    
    if N_eff_pf < NTh_pf:
        px_pf, pw_pf = re_sampling_pf(px_pf, pw_pf)
        
    return x_est_pf, p_est_pf, px_pf, pw_pf


def re_sampling_pf(px_pf, pw_pf):

    w_cum_pf = np.cumsum(pw_pf)
    base_pf = np.arange(0.0, 1.0, 1 / NP_pf)
    re_sample_id_pf = base_pf + np.random.uniform(0, 1 / NP_pf)
    
    indexes = []
    ind = 0
    
    for ip in range(NP_pf):
        while re_sample_id_pf[ip] > w_cum_pf[ind]:
            ind += 1
        indexes.append(ind)

    px_pf = px_pf[:, indexes]
    pw_pf = np.zeros((1, NP_pf)) + 1.0 / NP_pf

    return px_pf, pw_pf


def plot_covariance_ellipse_pf(x_est_pf, p_est_pf):
    p_xy_pf = p_est_pf[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(p_xy_pf)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    try:
        a = math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    
    fx = rot_mat_2d(angle) @ np.array([[x, y]])
    px = np.array(fx[:, 0] + x_est_pf[0, 0]).flatten()
    py = np.array(fx[:, 1] + x_est_pf[1, 0]).flatten()
    plt.plot(px, py, "--r")


"""
[Main]
"""

def main():
    print(__file__ + " start!!")

    # Common Variable
    time = 0.0
    xTrue = np.zeros((4, 1))
    
    # EKF Local Variable
    xEst_ekf = np.zeros((4, 1))
    PEst_ekf = np.eye(4)

    xDR_ekf = np.zeros((4, 1)) 

    hxEst_ekf = xEst_ekf
    hxTrue_ekf = xTrue
    hxDR_ekf = xTrue
    hz_ekf = np.zeros((2, 1))
    
    # PF Variable
    rf_id_pf = np.array([[10.0, 0.0],
                    [10.0, 10.0],
                    [0.0, 15.0],
                    [-5.0, 20.0]])
    x_est_pf = np.zeros((4, 1))
    px_pf = np.zeros((4, NP_pf))  
    pw_pf = np.zeros((1, NP_pf)) + 1.0 / NP_pf  
    x_dr_pf = np.zeros((4, 1))  

    h_x_est_pf = x_est_pf
    h_x_true_pf = xTrue
    h_x_dr_pf = xTrue

    # Simulation
    while SIM_TIME >= time:
        
        time += DT
        u = calc_input()
        
        
        # EKF 
        xTrue, z_ekf, xDR_ekf, ud_ekf = observation_ekf(xTrue, xDR_ekf, u)

        xEst_ekf, PEst_ekf = ekf_estimation_ekf(xEst_ekf, PEst_ekf, z_ekf, ud_ekf)

        hxEst_ekf = np.hstack((hxEst_ekf, xEst_ekf))
        hxDR_ekf = np.hstack((hxDR_ekf, xDR_ekf))
        hxTrue_ekf = np.hstack((hxTrue_ekf, xTrue))
        hz_ekf = np.hstack((hz_ekf, z_ekf))
        
        # PF
        xTrue, z_pf, x_dr_pf, ud_pf = observation_pf(xTrue, x_dr_pf, u, rf_id_pf)

        x_est_pf, PEst_pf, px_pf, pw_pf = pf_localization_pf(px_pf, pw_pf, z_pf, ud_pf)

        # store data history
        h_x_est_pf = np.hstack((h_x_est_pf, x_est_pf))
        h_x_dr_pf = np.hstack((h_x_dr_pf, x_dr_pf))
        h_x_true_pf = np.hstack((h_x_true_pf, xTrue))

        if show_animation:
            plt.cla()
            
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            
            plt.plot(hxTrue_ekf[0, :].flatten(), hxTrue_ekf[1, :].flatten(), "-b", label='True State')
            plt.plot(hz_ekf[0, :], hz_ekf[1, :], "o", color='lime', label='Observation (EKF)')
            plt.plot(hxDR_ekf[0, :].flatten(), hxDR_ekf[1, :].flatten(), "-k", label='Dead Reckoning (EKF)')
            plt.plot(hxEst_ekf[0, :].flatten(), hxEst_ekf[1, :].flatten(), "-c", label='Estimated (EKF)')
            plot_covariance_ellipse(xEst_ekf[0, 0], xEst_ekf[1, 0], PEst_ekf)
        
            plt.plot(rf_id_pf[:, 0], rf_id_pf[:, 1], "*", color='orange', label='Landmarks (PF)')
            plt.plot(px_pf[0, :], px_pf[1, :], ".", color='tomato', label='Particles (PF)')
            plt.plot(np.array(h_x_dr_pf[0, :]).flatten(), np.array(h_x_dr_pf[1, :]).flatten(), "-y", label='Dead Reckoning (PF)')
            plt.plot(np.array(h_x_est_pf[0, :]).flatten(), np.array(h_x_est_pf[1, :]).flatten(), "-m", label='Estimated (PF)')
            plot_covariance_ellipse_pf(x_est_pf, PEst_pf)
        
            plt.axis("equal")
            plt.grid(True)
            plt.legend()  
            plt.pause(0.001)



if __name__ == '__main__':
    main()
    
    plt.show()