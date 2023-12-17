"""
[알고리즘에 사용되는 Module Import]
"""

# 경로 문제를 해결하기 위해 필요한 모듈 및 패키지의 경로를 추가하는 코드이다. 
import sys
import pathlib

# 현재 실행 중인 스크립트 파일의 상위 디렉토리로 경로를 설정하는 역할을 한다. 
# __file__ 는 현재 실행 중인 스크립트의 파일 경로를 나타내는 파이썬 내장 변수를 의미한다. 
# pathlib.Path를 이용하여 현재 실핼 중인 스크립트 파일 경로를 인자로 하여 파일 경로 객체를 생성한다. 
# 얻은 객체를 문자열로 반환한다. 
# 얻은 경로를 sys.path 라는 Python이 모듈이나 패키지를 찾을 때 검색하는 경로를 지닌 리스트에 얻은 경로를 추가한다. 

'''
project/
|-- utils/
|   |-- __init__.py
|   |-- my_module.py
|
|-- scripts/
|   |-- script.py
'''

# 위와 같은 구조에서 script.py에서 my_module.py를 사용하고자 하는 경우 sys.path.append()를 사용하여 utils 디렉토리의 경로를 sys.path에 추가함으로써 my_module을 불러올 수 있다. 

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

# 필요한 수학 및 시각화 관련 라이브러리를 Import 한다. 
import math

import matplotlib.pyplot as plt
import numpy as np

# rot_mat_2d 이란 함수는 2D 공간에서의 회전 변환 행렬(rotation matrix)을 생성하는 함수이다. 
# 해당 함수는 각도를 주고 해당 각도를 이용하여 이에 따라 회전을 시키는데 사용되는 행렬이다. 
from utils.angle import rot_mat_2d

"""
[알고리즘에 사용되는 변수들 선언부]
"""

# Estimation parameter of PF
# Q와 R은 PF의 측정 모델과 제어 모델의 오차를 나타내는 공분산 행렬을 의미한다. 
# 해당 매개 변수들은 Esimation 알고리즘이 시스템의 불확실성을 어떻게 다룰지를 결정하며, 성능과 안정성에 영향을 준다. 

# Q는 측정 범위 오차는 센서로부터 얻은 측정값의 불확실성을 의미한다. 
# 공분산 행렬은 각 차원의 오차의 크기와 상호 간의 관계를 의미한다. 
# 주 대각선 요소에 0.2를 갖는 대각 행렬을 생성하고, 해당 행렬을 제곱하여 양의 값을 지니는 Covariance Matrix를 구성한다. 
# 해당 값은 센서 측정의 오차 크기를 의미한다.
Q = np.diag([0.2]) ** 2  # range error

# R은 입력(제어) 오차로 외부에서 시스템에 가해지는 제어 입력의 불확실성을 의미한다. 
# 2차원의 입력(선속도와 각속도)에 대한 Covariance Matrix를 생성하고 해당 Matrix도 역시 양의 값을 지니도록 제곱을 하여 구성한다. 
# 첫 번째 차원인 2.0 이라는 값이 선속도에 대한 오차, 두 번째 차원인 degree 40은 각속도에 대한 오차를 나타낸다. 
R = np.diag([2.0, np.deg2rad(40.0)]) ** 2  # input error

# => 오차 공분산 행렬들은 추정 알고리즘이 예측 단계와 측정 업데이트 단계에서 시스템의 불확실성을 어떻게 다룰지를 결정하는 데 중요한 역할을 한다. 
#    해당하는 값이 클수록 시스템의 불확실성이 크게 고려되며 PF가 Measurement과 Model Prediction 사이의 불일치를 얼마나 크게 간주하는지를 의미한다. 

# Q는 Sensor의 오차를 의미하며 Sensor에서 측정한 값이 실제 위치에서 어느 정도 떨어져 있을 수 있는지를 의미한다. 
# Q의 값이 클수록 Sensor Measurement이 더 불확실함을 의미한다. 

# R은 주어진 명령에 따라 로봇이 이동을 하는데 해당 명령의 정확도에 대한 불확실성을 의미한다. 
# R의 값이 클수록 외부에서 하는 Control Input이 더 불확실함을 의미한다. 

# Covariance Matrix들은 PF가 State Update 시, 현재 추정된 상태와의 불일치를 어떻게 고려할지를 결정한다. 
# Q와 R이 클수록 해당 오차가 크게 고려되어, Prediction과 Estimation 사이의 불일치를 보다 유연하게 다루게 되고 반대로 작을수록 불일치에 민감하게 반응하게 된다.


#  Simulation parameter
# Q_sim은 PF의 시뮬레이션에서 사용되는 값으로, 실제 센서에서 나타나는 불확실성을 시뮬레이션에서 어떻게 모델링할지를 나타낸다. 
# 시뮬레이션 상에서는 정확한 측정값이 아니라 모델에서 생성된 가상의 노이즈를 나타낸다. 

# R_sim은 실제 외부에서 발생하는 불확실성을 시뮬레이션에서 어떻게 구현할지를 나타낸다. 
# 시뮬레이션 상에서는 정확한 제어 입력이 아니라 모델에서 생성된 가상의 노이즈를 나타낸다. 

Q_sim = np.diag([0.2]) ** 2
R_sim = np.diag([1.0, np.deg2rad(30.0)]) ** 2

# Q와 R은 실제 데이터에서 나오는 불확실성을 반영하며, Q_sim과 R_sim은 모델이나 시뮬레이션에서 사용되는 불확실성을 의미하는 것이다. 
# 실제 환경과 시뮬레이션 간의 차이를 반영하고자 하는 목적에서 생성하는 두 개의 다른 Convariance Matrix이다. 

# 시뮬레이션에 사용되는 시간 간격, 총 시뮬레이션 시간 및 최대 관측 범위를 설정한다. 
DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]
MAX_RANGE = 20.0  # maximum observation range -> 실제 Sensor가 가지는 측정 범위 제한을 나타내기 위한 값이다. 

# PF에서 사용되는 Particle의 수를 지정하고 Re Sampling 하는 경우에 대한 Particle의 수를 지정한다. 
# Particle filter parameter
NP = 100  # Number of Particle
NTh = NP / 2.0  # Number of particle for re-sampling -> Re Sampling을 위해 선택되어야 하는 최소한의 Particle 수를 의미한다. 
# 상위 50%의 파티클들만을 유지하고 나머지는 제거하도록 하는 기준으로 사용한다. 
# 효과적으로 다양한 상태를 표현하고, 중요한 파티클을 유지하면서 계산 비용을 효율적으로 관리하기 위한 역할을 한다. 

# Animation을 나타내도록 한다. 
show_animation = True

"""
[calc_input 함수]
"""

# Robot Input Control 신호를 만드는 함수이다. 
# v는 선속도를 의미하고 yaw_rate는 각속도를 의미한다. 
# Robot은 1m를 1초 동안 이동하도록 설정한다. 
# Robot은 1초 당 0.1 rad 회전하도록 설정한다. 
# 최종적으로 설정한 선속도와 각속도를 각각의 요소로 가지는 Column Vector를 생성한다. 
# 이는 Robot의 Control Input에 해당하여 이에 따라 Robot의 이동 속도와 회전 속도가 결정되어 Robot Modeling을 하게 된다.
def calc_input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u

"""
[observation 함수]
"""

# Robot의 State와 Measurement를 생성하는 함수이다. 
# motion_model 함수를 사용하여 로봇의 움직임을 모델링한 후 측정값과 입력에 노이즈를 추가하는 작업을 수행한다.
def observation(x_true, xd, u, rf_id):

		# 로봇의 실제 상태인 x_true를 update 한다. 
		# 아래의 motion_model 함수는 Robot의 현재 State와 Control Input을 사용하여 다음 시간 단계에서의 State를 Prediction 하기 위해 계산하는 역할을 한다. 
		# 실제 Robot의 State와 Control Input을 이용하여 실제 Robot의 State를 구한다. 
    x_true = motion_model(x_true, u)

    # add noise to gps x-y
		# Measurement(z)을 저장하기 위한 용도로 사용되는 2차원 배열을 생성한다. 
		# 행이 0개 이고 열이 3개인 형태인 배열이며 초기에는 빈 배열의 형태로 존재한다. 
    z = np.zeros((0, 3))
		
		# rf_id는 Robot이 관찰할 수 있는 특정 지점 혹은 특징을 나타내는 랜드마크를 정보를 지니고 있다. 
		# rf_id는 랜드마크의 좌표를 담고 있는 배열로 각 행은 하나의 랜드마크를 나타내며 여러 행이 있는 경우 이는 다른 랜드마크들을 의미하게 된다. 
		# 각 행 내부에는 2개의 열이 존재하며 이는 각각 해당 랜드마크의 X 좌표 정보와 Y 좌표 정보를 지니고 있다. 
		# 해당 for 반복문은 각각의 랜드마크의 개수 만큼 반복을 진행하는 형태이다. 
    for i in range(len(rf_id[:, 0])):        
				
				# dx는 Robot의 현재 위치의 x 좌표와 i번째 랜드마크의 x 좌표 간의 차이를 나타낸다. 
        dx = x_true[0, 0] - rf_id[i, 0]
				
				# dy는 Robot의 현재 위치의 y 좌표와 i번째 랜드마크의 y 좌표 간의 차이를 나타낸다.				
        dy = x_true[1, 0] - rf_id[i, 1]
				
				# 구한 각각의 x와 y의 좌표 차이를 이용하여 Euclidian Distance를 계산하여 d에 저장한다. 
				# d는 현재 Robot의 위치와 i번째 랜드마크 간의 거리 차이를 의미하게 된다. 
        d = math.hypot(dx, dy)

				# 아래의 if 조건문은 현재 Robot의 위치와 i번째 랜드마크 간의 거리가 Sensor의 측정 가능 범위 내에 있는 경우를 나타낸다.
        if d <= MAX_RANGE:
						
						# 구한 거리 d에 정규분포를 따르는 임의의 난수를 생성하여 더하고 이 값에 Noise를 추기하여 dn을 구성한다. 
						# dn은 구한 거리 d에 Noise가 추가된 현재 Robot의 위치와 i번째 랜드마크 간의 거리를 의미하게 된다. 
						# 마지막 값의 **0.5는 해당 행렬의 대각 성분에 sqrt를 적용한 것으로 이는 측정 오차의 표준편차를 의미한다.  
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise

						# 계산된 Noise가 존재하는 Robot과 랜드마크 사이의 거리 값과 i번째 랜드마크의 좌표를 원소로 하는 배열을 만든다. 
						# zi라는 변수는 i번째 랜드마크에 대한 Noise Distance와 좌표 정보를 지니는 배열을 의미한다. 
            zi = np.array([[dn, rf_id[i, 0], rf_id[i, 1]]])

						# 구성한 i번째 랜드마크의 정보들을 구성한 z라는 배열에 수직의 방향으로 쌓아 올리게 된다. 
						# 이전까지의 구성된 z라는 배열에 새롭게 구한 zi라는 배열을 쌓아 z를 새롭게 update 한다. 
            z = np.vstack((z, zi))

    # add noise to input
		# u는 Control Input Signal을 나타내며, 여기에 R_sim으로 정의된 Input 오차의 Noise를 추가하도록 한다. 
		# 선속도에 대한 Noise를 추가한다. 
		# R_sim[0, 0] ** 0.5는 R_sim 행렬의 대각 원소에 sqrt를 적용한 값으로, Input 오차의 표준 편차를 나타낸다. 
    ud1 = u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5
		
		# 각속도에 대한 Noise를 추가한다. 
    ud2 = u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5
		# Robot의 선속도와 각속도를 나타내는 Control Input에 Noise를 추가하여 나타낸 값이 된다. 

		# Noise를 고려하여 나타낸 값을 Column Vector로 구성하여 ud 라는 배열에 저장한다. 
		# ud는 현재 Robot의 State에 Control Input의 불확실성을 고려하여 나타낸 값을 지니고 있다. 
    ud = np.array([[ud1, ud2]]).T
		
		# 현재 Robot의 State를 나타내는 xd와 구한 Noise가 포함된 Control Input을 이용하여 다음 시간 단계에 대한 Robot의 State를 구한다. 
		# Control Input Signal u에 대한 오차를 고려하여 Noise를 추가하고 다음 Step에 대한 Robot의 예측된 State를 생성하는 작업을 수행한다.  
    xd = motion_model(xd, ud)
		
		# 해당 함수는 Robot의 실제 State(x_true), 랜드마크의 Measurement(z), 예측된 로봇의 State(xd), 및 노이즈가 추가된 Input(ud)을 반환한다.
    return x_true, z, xd, ud

# x_true = motion_model(x_true, u) 
# => Robot이 실제로 움직이는 경우의 State 전이 모델을 나타낸다. 

# xd = motion_model(xd, ud)
# => PF가 사용하는 예측된 로봇 State를 갱신하는 부분을 나타낸다. 

# PF가 예측된 State를 기반으로 측정값을 예측하고, 이를 실제 관측값과 비교하여 State Update를 수행하는 과정을 진행하기에 나타난다. 
# 1. 실제 Robot의 State를 Update한다. 
# 2. Update 된 위치에서 랜드마크들과의 관계를 이용하여 Measurement를 측정한다. -> 오차를 지니는 Measurement 
# 3. Control Input에도 오차를 고려하고 이전까지 예측된 Robot의 State에 Noise가 추가된 Control Input을 가해 새로운 Predicted Robot State를 구성한다. 
# 4. Real Robot의 State & Noise가 고려된 Measurement & Predicted Robot State & Noise가 고려된 Control Input을 반환한다. 

"""
[motion model 함수]
"""

# Robot의 움직임을 나타내는 Motion Model에 대한 함수이다. 
# Robot의 State와 Control Input을 매개변수로 받는다. 
def motion_model(x, u):

		# F는 간단한 직선 운동을 나타내는 State 전이 Matrix를 의미한다. 
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])
		
		# B는 Control Input과 State 변수 사이의 변환 Matrix를 의미한다.
    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
		
		# 구성한 각각의 State 전이 Matrix에 [x,y,θ,v].T를 나타내는 x와 [v,ω].T를 나타내는 u와 연산을 한다. 
		# 계산된 결과는 Robot의 다음 State를 나타낸다. 
    x = F.dot(x) + B.dot(u)
		
		# Robot이 직진하고 회전하는 Motion에 대한 Modeling을 하고 있으며 구해진 Robot의 State를 반환한다. 
    return x

"""
[gauss_likelihood 함수]
"""

# 가우시안 확률을 계산하는 함수이다. 
# 가우시안 분포에 대한 수식을 나타낸다. -> 평균을 0으로 나타내기에 e에 대한 지수승에서 -x만으로 표현된다. 
def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p

"""
[calc_covariance 함수]
"""

# Convariance Matrix를 계산하는 함수이다.
# x_est: PF의 Estimation State Vector로 모든 Particle들의 가중 평균을 의미한다. => PF는 State를 확률 분포로 나타내며 이때 확률 분포의 중심에 해당하는 것을 의미한다. => PF가 현재 로봇의 State에 대한 최상의 Estimation을 나타낸다.
# px: Particle들의 State Vector를 갖는 Matrix를 나타낸다.
# pw: Particle들의 가중치를 갖는 Vector를 나타낸다. 

def calc_covariance(x_est, px, pw):
    """
    calculate covariance matrix
    see ipynb doc
    """
		
		# 3x3의 크기를 가지는 공분산을 저장할 배열을 만든다. 
    cov = np.zeros((3, 3))
		
		# Particle의 개수를 저장한다. 
    n_particle = px.shape[1]

		# Particle의 수만큼 for 반복문을 반복한다. 
    for i in range(n_particle):

				# 현재 Particle의 State Vector에서 Estimation State Vector를 뺀 차이를 계산한다.
        dx = (px[:, i:i + 1] - x_est)[0:3]
				
				# Particle의 가중치와 State 차이 Vector의 외적을 이용하여 Covariance Matrix을 누적한다. 
        cov += pw[0, i] * dx @ dx.T
		
		# 모든 Particle에 대한 누적된 Convariance을 전체 가중치의 합으로 나누어 정규화한다. 
    cov *= 1.0 / (1.0 - pw @ pw.T)
		
		# 최종적으로 계산된 Convariance Matrix를 반환한다. 
		# Covariance Matrix는 Estimation State의 불확실성을 나타낸다.
    return cov

"""
[pf_localization 함수]
"""

# PF Localization을 수행하는 함수이다. 
# px: Particle의 State를 의미한다.
# pw: Particle의 가중치를 의미한다. 
# z: Measurement를 의미한다.
# u: Control Input을 의미한다. 
def pf_localization(px, pw, z, u):
    
    """
    Localization with Particle filter
    """
		
		# 설정한 Particle의 수 만큼 for 반복문을 반복한다.
    for ip in range(NP):

				# 현재 Particle의 State를 x에 할당한다.  
        x = np.array([px[:, ip]]).T
				
				# 현재 Particle의 Weight를 w에 할당한다. 
        w = pw[0, ip]

        # Predict with random input sampling
				# Control Input에 대한 노이즈를 고려한다. 
        ud1 = u[0, 0] + np.random.randn() * R[0, 0] ** 0.5
        ud2 = u[1, 0] + np.random.randn() * R[1, 1] ** 0.5
        ud = np.array([[ud1, ud2]]).T
				
				# 현재 Particle의 State와 Noise가 추가된 Input Control을 가지고 현재 Particle의 State를 Update 한다.
        x = motion_model(x, ud)

        # Calc Importance Weight
				# Measurement의 수 만큼 for 반복문을 반복한다. 
        for i in range(len(z[:, 0])):

						# Particle Measurement의 차이를 구한다. 
            dx = x[0, 0] - z[i, 1]
            dy = x[1, 0] - z[i, 2]

						# 구한 차이를 가지고 두 사이의 거리를 구한다. 
            pre_z = math.hypot(dx, dy)

						# 예측된 거리와 측정된 거리의 차이를 계산한다.
            dz = pre_z - z[i, 0]
					
						# 가중치를 Update 한다. 
            w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))
				
				# Prediction 된 Particle의 State를 Update 한다. 
        px[:, ip] = x[:, 0]

				# Update 된 가중치를 저장한다. 
        pw[0, ip] = w
		
		# 구해진 가중치를 정규화 한다. 
    pw = pw / pw.sum()  # normalize
		
		# Particle의 State와 구한 가중치를 Dot Production 한다.
    x_est = px.dot(pw.T)

		# Estimation State와 Particle의 State와 가중치를 이용하여 Covariance Matrix를 계산한다. 
    p_est = calc_covariance(x_est, px, pw)
		
		# Effective Particle Number를 구한다. 
    N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number

		# 설정한 Threshold보다 수가 적은 경우 Re Sampling을 수행한다.
    if N_eff < NTh:
        px, pw = re_sampling(px, pw)

		# Update 된 Estimation State와 Convariance, Particle State, weight를 반환한다. 
    return x_est, p_est, px, pw

"""
[re_sampling 함수]
"""

# Re Sampling을 진행하는 함수를 의미한다. 
def re_sampling(px, pw):
    """
    low variance re-sampling
    """
		
		# 인자로 받은 가중치 pw를 누적하여 w_cum 에 저장한다. 
    w_cum = np.cumsum(pw)

		# 일정한 간격의 배열을 생성한다. 
    base = np.arange(0.0, 1.0, 1 / NP)
	  
	# 주어진 범위에서의 균일 분포 난수를 생성하고 이를 앞서 생성한 배열에 더해 Sampling 위치를 나타낸다.   
    re_sample_id = base + np.random.uniform(0, 1 / NP)

		# Re Sampling Particle Index 저장 배열을 생성한다. 
    indexes = []

		# 누적 가중치 접근을 위한 Index 변수를 생성한다. 
    ind = 0

		# Particle의 수 만큼 for 반복문을 반복한다.  
    for ip in range(NP):
	
				# 높은 가중치를 가지는 Particle을 더 자주 선택되게 하고 낮은 가중치는 제거하도록 한다. 
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
				
				# 가중치 비교 연산에서 선택된 Particle Index들을 추가한다. 
        indexes.append(ind)
		
		# 해당하는 Particle들의 State만을 px로 update한다. 
    px = px[:, indexes]

		# 가중치를 초기화한다. 
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight
		
		# 최종적인 px와 pw를 반환한다. 
    return px, pw


"""
[plot covariance ellipse 함수]
"""

# 추정된 State x_est와 해당 State의 공분산 행렬 p_est를 사용하여 오차 타원을 Plotting 하는 함수이다. 
# 타원의 크기와 방향은 State의 불확실성을 의미한다.
def plot_covariance_ellipse(x_est, p_est):  # pragma: no cover
    p_xy = p_est[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(p_xy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
    # numbers extremely close to 0 (~10^-20), catch these cases and set the
    # respective variable to 0
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
    px = np.array(fx[:, 0] + x_est[0, 0]).flatten()
    py = np.array(fx[:, 1] + x_est[1, 0]).flatten()
    plt.plot(px, py, "--r")

"""
[main 함수]
"""

def main():
    print(__file__ + " start!!")

    time = 0.0

    # RF_ID positions [x, y]
    rf_id = np.array([[10.0, 0.0],
                      [10.0, 10.0],
                      [0.0, 15.0],
                      [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    x_est = np.zeros((4, 1))
    x_true = np.zeros((4, 1))

    px = np.zeros((4, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    x_dr = np.zeros((4, 1))  # Dead reckoning

    # history
    h_x_est = x_est
    h_x_true = x_true
    h_x_dr = x_true

    while SIM_TIME >= time:
        time += DT
        u = calc_input()

        x_true, z, x_dr, ud = observation(x_true, x_dr, u, rf_id)

        x_est, PEst, px, pw = pf_localization(px, pw, z, ud)

        # store data history
        h_x_est = np.hstack((h_x_est, x_est))
        h_x_dr = np.hstack((h_x_dr, x_dr))
        h_x_true = np.hstack((h_x_true, x_true))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            for i in range(len(z[:, 0])):
                plt.plot([x_true[0, 0], z[i, 1]], [x_true[1, 0], z[i, 2]], "-k")
            plt.plot(rf_id[:, 0], rf_id[:, 1], "*k")
            plt.plot(px[0, :], px[1, :], ".r")
            plt.plot(np.array(h_x_true[0, :]).flatten(),
                     np.array(h_x_true[1, :]).flatten(), "-b")
            plt.plot(np.array(h_x_dr[0, :]).flatten(),
                     np.array(h_x_dr[1, :]).flatten(), "-k")
            plt.plot(np.array(h_x_est[0, :]).flatten(),
                     np.array(h_x_est[1, :]).flatten(), "-r")
            plot_covariance_ellipse(x_est, PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()
    
    plt.show()