import numpy as np
import sympy as sp
a2,a3,d1,d4,d5,d6,theta1,theta2,theta3,theta4,theta5,theta6 = sp.symbols('a2 a3 d1 d4 d5 d6 theta1 theta2 theta3 theta4 theta5 theta6')

TB0 = sp.Matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, d1],
                  [0, 0, 0, 1],])

T01 = sp.Matrix([
    [sp.cos(theta1), -sp.sin(theta1), 0, 0],
    [sp.sin(theta1), sp.cos(theta1), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T12 = sp.Matrix([
    [sp.cos(theta2+sp.pi), -sp.sin(theta2+sp.pi), 0, 0],
    [0, 0, -1, 0],
    [sp.sin(theta2+sp.pi), sp.cos(theta2+sp.pi), 0, 0],
    [0, 0, 0, 1]
])

T23 = sp.Matrix([
    [sp.cos(theta3), -sp.sin(theta3), 0, a2],
    [sp.sin(theta3), sp.cos(theta3), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T34 = sp.Matrix([
    [sp.cos(theta4), -sp.sin(theta4), 0, a3],
    [sp.sin(theta4), sp.cos(theta4), 0, 0],
    [0, 0, 1, d4],
    [0, 0, 0, 1]
])

T45 = sp.Matrix([
    [sp.cos(theta5), -sp.sin(theta5), 0, 0],
    [0, 0, 1, d5],
    [-sp.sin(theta5), -sp.cos(theta5), 0, 0],
    [0, 0, 0, 1]
])

T56 = sp.Matrix([
    [sp.cos(theta6+sp.pi), -sp.sin(theta6+sp.pi), 0, 0],
    [0, 0, -1, 0],
    [sp.sin(theta6+sp.pi), sp.cos(theta6+sp.pi), 0, 0],
    [0, 0, 0, 1]
])
T6W = sp.Matrix([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, d6],
                 [0, 0, 0, 1],])



T05 = T01 * T12 * T23 * T34 * T45

T03 = T01 * T12 * T23

T06 = T01 * T12 * T23 * T34 * T45 * T56

T36 = T34 * T45 * T56



sp.pprint(sp.simplify(), num_columns = 250, wrap_line = False)


"""now find the inverse of T06"""



#sp.init_printing()
#sp.pprint(sp.simplify(T06), num_columns = 250, wrap_line = False)

#TB3 = TB1 * T12 * T23
#sp.init_printing()
#sp.pprint(sp.simplify(TB3), num_columns = 250, wrap_line = False)

#T36 = T34 * T45 * T56
#sp.init_printing()
#sp.pprint(sp.simplify(T36), num_columns = 250, wrap_line = False)

