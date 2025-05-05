import numpy as np


a2 = 425
a3 = 392
d1 = 89.2
d4 = 109.3
d5 = 94.75
d6 = 82.5

theta1_snyt = 0
theta2_snyt = 90
theta3_snyt = -90
theta4_snyt = 0
theta5_snyt = 90
theta6_snyt = 0

t1 = np.deg2rad(theta1_snyt)
t2 = np.deg2rad(theta2_snyt)
t3 = np.deg2rad(theta3_snyt)
t4 = np.deg2rad(theta4_snyt)
t5 = np.deg2rad(theta5_snyt)
t6 = np.deg2rad(theta6_snyt)

TB0 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d1],
                [0, 0, 0, 1],])

T01 = np.array([
    [np.cos(t1), -np.sin(t1), 0, 0],
    [np.sin(t1), np.cos(t1), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T12 = np.array([
    [np.cos(t2 + np.pi), -np.sin(t2 + np.pi), 0, 0],
    [0, 0, -1, 0],
    [np.sin(t2 + np.pi), np.cos(t2 + np.pi), 0, 0],
    [0, 0, 0, 1]
])

T23 = np.array([
    [np.cos(t3), -np.sin(t3), 0, a2],
    [np.sin(t3), np.cos(t3), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T34 = np.array([
    [np.cos(t4), -np.sin(t4), 0, a3],
    [np.sin(t4), np.cos(t4), 0, 0],
    [0, 0, 1, d4],
    [0, 0, 0, 1]
])

T45 = np.array([
    [np.cos(t5), -np.sin(t5), 0, 0],
    [0, 0, 1, d5],
    [-np.sin(t5), -np.cos(t5), 0, 0],
    [0, 0, 0, 1]
])

T56 = np.array([
    [np.cos(t6 + np.pi), -np.sin(t6 + np.pi), 0, 0],
    [0, 0, -1, 0],
    [np.sin(t6 + np.pi), np.cos(t6 + np.pi), 0, 0],
    [0, 0, 0, 1]
])
T6W = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,d6],
               [0,0,0,1]])

TBW = TB0 @ T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T6W

T_DBW = np.array([
[     0.000000,     0.000000,    -1.000000,  -474.500000],
[-1.000000,     0.000000,    -0.000000,  -109.300000],
[-0.000000,     1.000000,     0.000000,  -430.550000],
[0.000000,     0.000000,     0.000000,     1.000000 ]
])

td1_4 = np.linalg.inv(TB0 @ T01)  @ T_DBW @ np.linalg.inv(T45 @ T56 @ T6W)


np.set_printoptions(precision=2, suppress=True)
print(td1_4)
