import numpy as np

a2 = 425
a3 = 392
d1 = 89.2
d4 = 109.3
d5 = 94.75
d6 = 82.5


Theta_1_deg_up = 0
theta5_deg = 90
theta6_deg = -180



TB0 = np.array([[1 ,0 ,0, 0],
                [0 ,1 ,0, 0],
                [0 ,0 ,1, d1],
                [0 ,0 ,0, 1]])

T01_up = np.array([
    [np.cos(Theta_1_deg_up), -np.sin(Theta_1_deg_up), 0, 0],
    [np.sin(Theta_1_deg_up), np.cos(Theta_1_deg_up), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


T45 = np.array([
    [np.cos(theta5_deg), -np.sin(theta5_deg), 0, 0],
    [0, 0, 1, d5],
    [-np.sin(theta5_deg), -np.cos(theta5_deg), 0, 0],
    [0, 0, 0, 1]])

T56 = np.array([
[np.cos(theta6_deg + np.pi), -np.sin(theta6_deg + np.pi), 0, 0],
    [0, 0, -1, 0],
    [np.sin(theta6_deg + np.pi), np.cos(theta6_deg + np.pi), 0, 0],
    [0, 0, 0, 1]
])

T6W = np.array([[1 ,0 ,0, 0],
                [0 ,1 ,0, 0],
                [0 ,0 ,1, d6],
                [0 ,0 ,0, 1]])



T_DBW = np.array([
[     0.000000,     0.000000,    -1.000000,  -474.500000],
[     -1.000000,     0.000000,    -0.000000,  -109.300000],
[     -0.000000,     1.000000,     0.000000,  -430.550000 ],
[      0.000000,     0.000000,     0.000000,     1.000000 ]
])

Td1_4 = np.linalg.inv(TB0 @ T01_up) @ T_DBW @ np.linalg.inv(T45 @ T56 @ T6W)

np.set_printoptions(precision=2, suppress=True)
print(Td1_4)