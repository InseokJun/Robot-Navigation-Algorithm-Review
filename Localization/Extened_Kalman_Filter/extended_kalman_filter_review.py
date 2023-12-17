"""
[필요한 Module 및 Library Import]
"""

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math
import matplotlib.pyplot as plt
import numpy as np

from utils.plot import plot_covariance_ellipse

"""
[변수 정의]
"""

# Covariance for EKF simulation
# Prediction을 하는데사용되며, 시스템 모델에 따른 예측 오차의 크기를 나타낸다. 
# 행렬의 대각 요소들을 아래와 같이 설정하도록 한다. 
Q = np.diag([

		# x축 위치에 대한 분산을 의미한다. 
    0.1,  # variance of location on x-axis

		# y축 위치에 대한 분산을 의미한다.  
    0.1,  # variance of location on y-axis

		# pose에 대한 분산을 의미한다. 
    np.deg2rad(1.0),  # variance of yaw angle

		# 속도에 대한 분산을 의미한다. 
    1.0  # variance of velocity
]) ** 2  # predict state covariance
# 구성한 분산에 제곱을 하여 Q Matrix를 구성한다.

# Update 단계에서 사용되는 Covariance Matrix이다.
# 측정값과 예측값 간의 차이에 대한 노이즈의 크기를 나타낸다. 
# x축 위치에 대한 분산을 첫 번째 요소에 나타낸다. 
# y축 위치에 대한 분산을 두 번째 요소에 나타낸다.  
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

#  Simulation parameter
# 입력에 대한 노이즈를 나타내며 이는 속도와 각속도에 대한 노이즈를 나타낸다. 
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2

# 측정값인 위치 대한 노이즈를 나타낸다. 
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

# 시간 간격에 대한 설정을 한다. 
DT = 0.1  # time tick [s]

# 시뮬레이션을 진행하는 전체 시간을 설정한다.
SIM_TIME = 50.0  # simulation time [s]

# 결과의 시각화를 진행하도록 설정한다.
show_animation = True

"""
[calc_input 함수]
"""

# Input 명령을 생성하는 부분으로 각도 및 각속도에 대한 Input 명령 생성 함수이다. 
def calc_input():

		# 선속도에 대한 값을 설정한다. 
    v = 1.0  # [m/s]

		# 각속도에 대한 값을 설정한다. 
    yawrate = 0.1  # [rad/s]

		# 선속도와 각속도를 2x1의 matrix로 저장한다.
    u = np.array([[v], [yawrate]])

		# 선속도와 각속도 정보를 지닌 Input 명령을 반환한다. 
    return u

"""
[observation 함수]
"""

# 현재 상태 xTrue에서 Measurement를 생성하는 함수이다. 
# xTrue: Robot의 실제 State를 의미한다. 
# xd: 현재 State를 아는 상태에서 출발하여 속도와 방향 정보를 이용하여 이동한 후의 예상 State를 추정한 값을 의미한다. 
#     Measuremenet를 사용하지 않고 Control Input에 대해서 예측된 State를 의미한다. 
# u: Robot의 선속도와 가속도 정보를 지니는 Control Input을 의미한다. 
def observation(xTrue, xd, u):

		# 현재 Robot의 실제 State와 Control Input을 사용하여 Robot의 Robot의 다음 State를 예측한다. 
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
		# 실제 Robot의 State에서 Measurement를 생성한다. 
		# 생성된 Measurement에 Noise를 더하여 실제 Measurement의 불확실성을 Modeling 한다. 
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
		# Input Control에 대한 불확실성을 Modeling 한다. 
    ud = u + INPUT_NOISE @ np.random.randn(2, 1)
		
		# 불확실성이 내포된 Control Input과 이전에 구한 Predict State를 이용하여 Robot의 Predict State를 Update 한다. 
    xd = motion_model(xd, ud)
		
		# 현재 Robot 상태 xTrue, 생성된 Noise가 내포된 Measurement z, 업데이트된 Predicted State xd, 및 노이즈가 추가된 Input ud를 반환한다. 
    return xTrue, z, xd, ud

# 시뮬레이션에서 현실 세계의 불확실성을 모델링하기 위해 작성한 코드이다. 
# 실제 세계에서는 Sensor의 Measurement에는 불확실성이 존재하고 Input에 역시 불확실성이 존재한다. 
# 실제 환경에서 존재하는 각각의 불확실성을 고려하여 Modeling을 진행한다. 

"""
[motion_model 함수]
"""

# 현재의 State와 Input을 받아서 다음 State를 Predict 하는 함수이다.
def motion_model(x, u):

		# 현재 State를 다음 State로 변환하는 변환 행렬을 의미한다. 
		# x, y 위치와 yaw에 대한 부분을 고려하고 v에 대한 부분은 고려하지 않는다.  
		# 각각의 행은 State 변수들의 변화를 의미한다.
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])
		
		# Input에 대한 변환을 나타내는 행렬을 의미한다. 
		# 현재 입력 u를 이용하여 State에 어떤 변화가 있는지를 나타낸다.
		# cos과 sin에 대한 비선형 함수가 사용된다. 
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
		
		# 현재 State에 구성한 State 변환 행렬 F를 곱하여 다음 State 예측을 진행한다. 
		# 현재 Input 명령에 B를 곱하여 Input에 따른 State 변화 예측을 진행한다. 
		# 각각의 변화량을 더하여 다음 State를 예측할 수 있다.  
    x = F @ x + B @ u
		
		# 최종적으로 구해진 State를 반환한다. 
    return x

"""
[observation_model 함수]
"""

# 현재 State에서 Measurement를 생성하는 함수이다.
# x: 현재 State를 나타낸다. 
def observation_model(x):

		# Observation Model을 나타내는 Matrix로 State와 Measurement 간의 관계를 나타낸다.
		# 로봇의 x 좌표를 측정값으로 사용할 것임을 나타내며, 이 관측값에 대한 가중치는 1임을 나타낸다.
		# 로봇의 y 좌표를 측정값으로 사용할 것임을 나타내며, 이 관측값에 대한 가중치는 1임을 나타낸다. 
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
		
		# 구성한 H Matrix와 Robot의 State Matrix를 이용하여 Measurement를 구한다.
    z = H @ x
		
		# 구해진 Measurement를 반환한다. 
		# z는 State x를 기반으로 구해진 Measurement를 의미한다. 
    return z

"""
[jacob_f 함수]
"""


# 비선형에 관한 함수를 선형화 시키는 작업을 하는 함수이다. 
# State와 Input에 대한 Motion Model의 Jacobian Matrix를 구한다. 
# Jacobian Matrix는 비선형 함수를 선형 근사화하기 위한 Matrix를 의미한다. 
def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t + v * dt * cos(yaw)
    y_{t+1} = y_t + v * dt * sin(yaw)
    yaw_{t+1} = yaw_t + omega * dt
    v_{t+1} = v{t}

    so

    dx/dyaw = -v * dt * sin(yaw)
    dx/dv = dt * cos(yaw)
    dy/dyaw = v * dt * cos(yaw)
    dy/dv = dt * sin(yaw)
    """
		
		# State Matrix에서 해당하는 부분을 yaw에 저장한다. 
    yaw = x[2, 0]

		# Input Matrix에서 해당하는 부분을 v에 저장한다. 
    v = u[0, 0]

		# 각 State 변수에 대한 변화율을 고려하여 Jacobian Matrix를 구성한다. 
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])
		
		# Extended Kalman Filter의 예측 단계에서는 현재 State를 Motion Model에 적용하여 다음 State를 예측한다.
		# 그 후 이 jacobian matrix을 사용하여 예측 오차의 Covariance를 계산한다. 
		# 해당 과정에서 비선형 Motion Model을 선형으로 근사화하기 위한 Matrix이다.  
    return jF

"""
[jacob_h 함수]
"""

# 비선형에 대한 부분을 선형으로 변환하는 함수이다. 
# Observation Model에 대한 Jacobian Matrix를 생성한다. 
def jacob_h():

    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH

"""
[ekf_estimation]
"""

# EKF의 예측 및 업데이트 단계를 구현하는 함수이다. 
# Predict 단계에서는 Motion Model에 따라 State를 Prediction 한다.
# Update 단계에서는 관측된 값과 예측된 값 간의 오차를 최소화하며 State를 Update 한다. 
# xEst: State의 Estimation을 의미한다.
# PEst: 현재 State에 대한 Covariance를 의미한다. 
# z: Measurement를 의미한다. 
# u: Control Input을 의미한다. 
def ekf_estimation(xEst, PEst, z, u):

    # Predict
		# State와 Control Input을 이용하여 State를 Prediction 한다. 
    xPred = motion_model(xEst, u)
		
		# Motion Model을 선형화하기 위한 Jacobian을 계산한다. 
    jF = jacob_f(xEst, u)

		# 예측된 State에 대한 Covariance를 구하고 이에 불확실성을 도입한다.
    PPred = jF @ PEst @ jF.T + Q

    # Update
		# Observation Model에 대한 Jacobian Matrix를 생성한다. 
    jH = jacob_h()
		
		# Observation Model에 Prediction 한 State를 넣어 Measurement를 Predict한다.
    zPred = observation_model(xPred)

		# Measurement와 Predction 한 Measurement의 차이를 구한다. 
    y = z - zPred

		# Measurement 오차를 나타내는Covariance를 구하며 Sensor의 불확실성을 고려한다. 
    S = jH @ PPred @ jH.T + R
		
		# Kalman Filter의 Gain을 구한다.
    K = PPred @ jH.T @ np.linalg.inv(S)

		# Prediction State와 Kalman Gain과 Measurement의 차이를 고려하여 State를 Update 한다. 
    xEst = xPred + K @ y
		
		# Predict Covariance와 Kalman Gain을 사용하여 Covariance를 Update 한다.
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

		# Update 된 State와 Covariance를 반환한다. 
    return xEst, PEst


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

		# 측정값을 사용하지 않고 현재 상태에서의 속도 및 방향 정보만으로 위치를 추정한 것을 의미한다. 
    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        xTrue, z, xDR, ud = observation(xTrue, xDR, u)

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
    
    plt.show()