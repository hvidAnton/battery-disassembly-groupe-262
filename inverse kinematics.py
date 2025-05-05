
import numpy as np
a2 = 425
a3 = 392
d1 = 89.2
d4 = 109.3
d5 = 94.75
d6 = 82.5


np.set_printoptions(precision=2, suppress=True)

"""here you input your matrix from robodk"""
T_DBW = np.array([
[     0.000000,     0.000000,    -1.000000,  -474.500000],
[     -1.000000,     0.000000,    -0.000000,  -109.300000],
[     -0.000000,     1.000000,     0.000000,  -430.550000 ],
[      0.000000,     0.000000,     0.000000,     1.000000 ]
])
"""current matrix inside this program has the theta angles (0, 90, -90, 0, 90, 0) and my code tries to work
backwards to find the angles again by only knowing the matrix given from robodk """


""" we start by removing the matrices TB0 and T6W from T_DBW.
This is because the matrices only extend the robots reach without havin a 
 theta angle or rotation"""

TB0 = np.array([[1 ,0 ,0, 0],
                [0 ,1 ,0, 0],
                [0 ,0 ,1, d1],
                [0 ,0 ,0, 1]])

T6W = np.array([[1 ,0 ,0, 0],
                [0 ,1 ,0, 0],
                [0 ,0 ,1, d6],
                [0 ,0 ,0, 1]])

td0_6 = np.linalg.inv(TB0) @ T_DBW @ np.linalg.inv(T6W)

#print(td0_5)
"""Theta 1"""

x0_6 = td0_6[0,3]
y0_6 = td0_6[1,3]

D = np.arctan2(y0_6,x0_6)

arccos_theta_1 = np.arccos(d4/(np.sqrt(x0_6**2 + y0_6**2)))

Theta1_elbow_up = D + arccos_theta_1 + np.pi/2


Theta_1_deg_up = np.rad2deg(Theta1_elbow_up)


print(f"θ1 up: {Theta_1_deg_up:.2f}°")

"""theta 5 """



T01_up = np.array([
    [np.cos(Theta_1_deg_up), -np.sin(Theta_1_deg_up), 0, 0],
    [np.sin(Theta_1_deg_up), np.cos(Theta_1_deg_up), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])


theta5 = np.arccos(np.sin(Theta1_elbow_up)*td0_6[0,2]-np.cos(Theta1_elbow_up)*td0_6[1,2])

theta5_deg = np.rad2deg(theta5)

#print(td0_5[0,2])
#print(td0_5[1,2])

print(f"theta5: {theta5_deg:.2f}")

T45 = np.array([
    [np.cos(theta5_deg), -np.sin(theta5_deg), 0, 0],
    [0, 0, 1, d5],
    [-np.sin(theta5_deg), -np.cos(theta5_deg), 0, 0],
    [0, 0, 0, 1]])

"""Theta 6"""

# Compute θ₆ from orientation matrix
R06 = td0_6[:3, :3]

sin_theta5 = np.sin(theta5)
if abs(sin_theta5) < 1e-6:
    raise ValueError("Singularity: θ₅ is 0° or 180°; θ₆ is undefined.")

# Use correct formula
X6x, X6y = R06[0, 0], R06[1, 0]  # X₆ in base frame
Y6x, Y6y = R06[0, 1], R06[1, 1]  # Y₆ in base frame

numerator_x = (-X6y * np.sin(Theta1_elbow_up) + Y6y * np.cos(Theta1_elbow_up))
numerator_y = (X6x * np.sin(Theta1_elbow_up) - Y6x * np.cos(Theta1_elbow_up))

# θ₅ from earlier step
sin_theta5 = np.sin(theta5)
if abs(sin_theta5) < 1e-6:
    raise ValueError("θ₅ is near singularity.")

x = numerator_x / sin_theta5
y = numerator_y / sin_theta5

theta6 = np.arctan2(y, x)
theta6_deg = np.rad2deg(theta6)
print(f"theta6: {theta6_deg:.2f}")

T56 = np.array([
[np.cos(theta6_deg + np.pi), -np.sin(theta6_deg + np.pi), 0, 0],
    [0, 0, -1, 0],
    [np.sin(theta6_deg + np.pi), np.cos(theta6_deg + np.pi), 0, 0],
    [0, 0, 0, 1]
])

"""Theta 2"""

""" nu importerer jeg vores forward kinematic, ole sage det ver en god ide"""

"""nogle af de matricer jeg har lavet er ikke orthogonale ved ikke hvorfor"""

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

TB0_s = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, d1],
                [0, 0, 0, 1],])

T01_s = np.array([
    [np.cos(t1), -np.sin(t1), 0, 0],
    [np.sin(t1), np.cos(t1), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T12_s = np.array([
    [np.cos(t2 + np.pi), -np.sin(t2 + np.pi), 0, 0],
    [0, 0, -1, 0],
    [np.sin(t2 + np.pi), np.cos(t2 + np.pi), 0, 0],
    [0, 0, 0, 1]
])

T23_s = np.array([
    [np.cos(t3), -np.sin(t3), 0, a2],
    [np.sin(t3), np.cos(t3), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T34_s = np.array([
    [np.cos(t4), -np.sin(t4), 0, a3],
    [np.sin(t4), np.cos(t4), 0, 0],
    [0, 0, 1, d4],
    [0, 0, 0, 1]
])

T45_s = np.array([
    [np.cos(t5), -np.sin(t5), 0, 0],
    [0, 0, 1, d5],
    [-np.sin(t5), -np.cos(t5), 0, 0],
    [0, 0, 0, 1]
])

T56_s = np.array([
    [np.cos(t6 + np.pi), -np.sin(t6 + np.pi), 0, 0],
    [0, 0, -1, 0],
    [np.sin(t6 + np.pi), np.cos(t6 + np.pi), 0, 0],
    [0, 0, 0, 1]
])
T6W_s = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,d6],
               [0,0,0,1]])

#calculations for theta 2



td1_4 = np.linalg.inv(TB0_s @ T01_s) @ T_DBW @ np.linalg.inv(T45_s @ T56_s @ T6W_s)

#print(td1_4)

x1_4 = td1_4[0,3]
z1_4 = td1_4[2,3]


D14 = np.sqrt(x1_4**2 + z1_4**2)

# Check reachability
if D14 > (a2 + a3) or D14 < np.abs(a2 - a3):
    raise ValueError("Target unreachable!")

# Angle to target (from +X-axis)
phi = np.arctan2(z1_4,x1_4)


#phi2 = np.arctan2(a3,a2)
# Law of Cosines to find alpha
cos_phi2 = (-a2**2 - D14**2 + a3**2) / (2 * a2 * D14)
alpha = np.arccos(np.clip(cos_phi2, -1, 1))  # Avoid numerical errors

#print(np.rad2deg(alpha))
#print(np.rad2deg(phi))
# Two solutions (elbow up/down)
theta2_elbow_up = phi - alpha
theta2_elbow_down = phi + alpha

# Convert to degrees
theta2_up_deg = np.rad2deg(theta2_elbow_up)
theta2_down_deg = np.rad2deg(theta2_elbow_down)

#print(f"θ₂ (elbow up): {theta2_up_deg:.2f}°")
print(f"θ₂ (elbow down): {theta2_down_deg:.2f}°")


"""Theta 3"""

cos_phi3 = (-a2**2 - a3**2 + D14**2) / (2 * a2 * a3)
phi3 = np.arccos(np.clip(cos_phi3, -1, 1))
theta3 = -phi3
theta3_deg = np.rad2deg(theta3)
print(f"θ₃: {theta3_deg:.2f}°")

"""Theta 4"""

td3_4 = np.linalg.inv(TB0_s @ T01_s @ T12_s @ T23_s) @ T_DBW @ np.linalg.inv(T45_s @ T56_s @ T6W_s)

# calculations for theta 4

# To find theta4, we need T34 (Frame [4] relative to Frame [3])
# Since T03 = T01 @ T12 @ T23, we can compute T34 as:


# Extract the x-axis of Frame [4] in Frame [3]'s coordinates
x4_x = td3_4[0, 0]  # cos(theta4)
x4_y = td3_4[1, 0]  # sin(theta4)

# Compute theta4 using arctan2
theta4_computed = np.arctan2(x4_y, x4_x)
theta4_deg = np.rad2deg(theta4_computed)

print(f"θ₄: {theta4_deg:.2f}°")



"""
# calculations for theta 6

R06 = td0_5[:3, :3]

# Extract axes (corrected indices)
X6_x = R06[0, 0]  # 6X0x
X6_y = R06[1, 0]  # 6X0y
Y6_x = R06[0, 1]  # 6Y0x
Y6_y = R06[1, 1]  # 6Y0y

# Compute numerators (ensure correct signs)
numerator_x = -X6_y * np.sin(Theta1_elbow_up) + X6_y * np.cos(Theta1_elbow_up)
numerator_y = Y6_x * np.sin(Theta1_elbow_up) - Y6_y * np.cos(Theta1_elbow_up)

# Handle θ₅ = 90° case (sin(θ₅) = 1)
sin_theta5 = np.sin(theta5)
if abs(sin_theta5) < 1e-10:
    raise ValueError("Singularity: θ₅ is 0° or 180°.")

# Compute θ₆ (flip signs if needed)
x = numerator_x / sin_theta5
y = numerator_y / sin_theta5
theta6 = np.arctan2(y, x)

# Adjust for quadrant (if θ₆ is 180° but should be 0°)
if np.isclose(abs(theta6), np.pi, atol=1e-4):  # ≈ 180°
    theta6 = 0.0  # Force to 0° if expected


theta6_deg = np.rad2deg(theta6)

print(f"θ₆: {theta6_deg:.2f}°")  # Should now output 0.00°




#calculations for theta 2

T45 = sp.Matrix([
    [sp.cos(theta5), -sp.sin(theta5), 0, 0],
    [0, 0, 1, d5],
    [-sp.sin(theta5), -sp.cos(theta5), 0, 0],
    [0, 0, 0, 1]
])

td1_4 = np.linalg.inv(Theta1_elbow_up) @ td0_5 @ np.linalg.inv(T45)

x1_4 = td1_4[0,3]
z1_4 = td1_4[2,3]

# Distance to target
D = np.sqrt(x1_4**2 + z1_4**2)

# Check reachability
if D > (a2 + a3) or D < np.abs(a2 - a3):
    raise ValueError("Target unreachable!")

# Angle to target (from +X-axis)
phi = np.arctan2(z1_4, x1_4)

# Law of Cosines to find alpha
cos_phi2 = (a2**2 + D**2 - a3**2) / (2 * a2 * D)
alpha = np.arccos(np.clip(cos_phi2, -1, 1))  # Avoid numerical errors

# Two solutions (elbow up/down)
theta2_elbow_up = -phi - alpha
theta2_elbow_down = -phi + alpha

# Convert to degrees
theta2_up_deg = np.rad2deg(theta2_elbow_up)
theta2_down_deg = np.rad2deg(theta2_elbow_down)

#print(f"θ₂ (elbow up): {theta2_up_deg:.2f}°")
print(f"θ₂ (elbow down): {theta2_down_deg:.2f}°")

#print(Theta2)

# calculations for theta 3

cos_phi3 = (a2**2 + a3**2 - D**2) / (2 * a2 * a3)
phi3 = np.arccos(np.clip(cos_phi3, -1, 1))
theta3 = np.pi - phi3
theta3_deg = np.rad2deg(theta3)
print(f"θ₃: {theta3_deg:.2f}°")

# calculations for theta 4

# To find theta4, we need T34 (Frame [4] relative to Frame [3])
# Since T03 = T01 @ T12 @ T23, we can compute T34 as:
T03 = T01 @ T12 @ T23
T34_computed = np.linalg.inv(T03) @ T04  # T34 = T03^{-1} @ T04

# Extract the x-axis of Frame [4] in Frame [3]'s coordinates
x4_x = T34_computed[0, 0]  # cos(theta4)
x4_y = T34_computed[1, 0]  # sin(theta4)

# Compute theta4 using arctan2
theta4_computed = np.arctan2(x4_y, x4_x)
theta4_deg = np.rad2deg(theta4_computed)

print(f"θ₄: {theta4_deg:.2f}°")

#compute theta5

Px6_0 = td0_6[0,3]
Py6_0 = td0_6[1,3]

# Compute numerator
numerator = -Px6_0 * np.sin(theta1) + Py6_0 * np.cos(theta1) + d4
# Avoid division by zero and numerical errors
if abs(d6) < 1e-10:
    raise ValueError("d6 is too small, cannot divide by zero.")

# Ensure numerator is within [-1, 1] for arccos
cos_theta5 = numerator / d6
cos_theta5 = np.clip(cos_theta5, -1, 1)  # Clip to valid range

# Two solutions (elbow up/down)
theta5_up = np.arccos(cos_theta5)
theta5_down = -np.arccos(cos_theta5)

# Convert to degrees
theta5_up_deg = np.rad2deg(theta5_up)
theta5_down_deg = np.rad2deg(theta5_down)

#print(f"θ₅ (elbow up): {theta5_up_deg:.2f}°")
print(f"θ₅ (elbow down): {theta5_down_deg:.2f}°")

# calculations for theta 6

R06 = T06[:3, :3]

# Extract axes (corrected indices)
X6_x = R06[0, 0]  # 6X0x
X6_y = R06[1, 0]  # 6X0y
Y6_x = R06[0, 1]  # 6Y0x
Y6_y = R06[1, 1]  # 6Y0y

# Compute numerators (ensure correct signs)
numerator_x = -X6_x * np.sin(theta1) + Y6_x * np.cos(theta1)
numerator_y = -X6_y * np.sin(theta1) + Y6_y * np.cos(theta1)

# Handle θ₅ = 90° case (sin(θ₅) = 1)
sin_theta5 = np.sin(theta5)
if abs(sin_theta5) < 1e-10:
    raise ValueError("Singularity: θ₅ is 0° or 180°.")

# Compute θ₆ (flip signs if needed)
x = numerator_x / sin_theta5
y = numerator_y / sin_theta5
theta6 = np.arctan2(y, x)

# Adjust for quadrant (if θ₆ is 180° but should be 0°)
if np.isclose(abs(theta6), np.pi, atol=1e-4):  # ≈ 180°
    theta6 = 0.0  # Force to 0° if expected


theta6_deg = np.rad2deg(theta6)

print(f"θ₆: {theta6_deg:.2f}°")  # Should now output 0.00°
"""