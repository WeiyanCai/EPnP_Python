# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 16:11:15 2017

@author: Weiyan Cai
"""

import numpy as np


def gauss_newton(Kernel, Cw, Beta0):
    L = compute_L6_10(Kernel)
    rho = compute_rho(Cw)
    
    current_betas = Beta0
    
    n_iterations = 10
    
    for k in range(n_iterations):
        A, b = compute_A_and_b_Gauss_Newton(current_betas, rho, L)
        dbeta = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
        current_betas = current_betas + dbeta.T[0]
        error = np.matmul(b.T, b)
    
    Beta_opt = current_betas
    
    return Beta_opt, error

def compute_L6_10(K):
    L = np.zeros((6, 10))

    # extract vectors
    v1 = K[:, 0]
    v2 = K[:, 1]
    v3 = K[:, 2]
    v4 = K[:, 3]
    
    d1 = []
    for i in range(3):
        for j in range(i + 1, 4):
            d1.append(v1[3 * i : 3 * (i + 1)] - v1[3 * j : 3 * (j + 1)])
    
    d12 = d1[0]
    dx112, dy112, dz112 = d12[0], d12[1], d12[2]
    d13 = d1[1]
    dx113, dy113, dz113 = d13[0], d13[1], d13[2]
    d14 = d1[2]
    dx114, dy114, dz114 = d14[0], d14[1], d14[2]
    d23 = d1[3]
    dx123, dy123, dz123 = d23[0], d23[1], d23[2]
    d24 = d1[4]
    dx124, dy124, dz124 = d24[0], d24[1], d24[2]
    d34 = d1[5]
    dx134, dy134, dz134 = d34[0], d34[1], d34[2]
       
    d2 = []
    for i in range(3):
        for j in range(i + 1, 4):
            d2.append(v2[3 * i : 3 * (i + 1)] - v2[3 * j : 3 * (j + 1)])
    
    d12 = d2[0]
    dx212, dy212, dz212 = d12[0], d12[1], d12[2]
    d13 = d2[1]
    dx213, dy213, dz213 = d13[0], d13[1], d13[2]
    d14 = d2[2]
    dx214, dy214, dz214 = d14[0], d14[1], d14[2]
    d23 = d2[3]
    dx223, dy223, dz223 = d23[0], d23[1], d23[2]
    d24 = d2[4]
    dx224, dy224, dz224 = d24[0], d24[1], d24[2]
    d34 = d2[5]
    dx234, dy234, dz234 = d34[0], d34[1], d34[2]

    d3 = []
    for i in range(3):
        for j in range(i + 1, 4):
            d3.append(v3[3 * i : 3 * (i + 1)] - v3[3 * j : 3 * (j + 1)])
    
    d12 = d3[0]
    dx312, dy312, dz312 = d12[0], d12[1], d12[2]
    d13 = d3[1]
    dx313, dy313, dz313 = d13[0], d13[1], d13[2]
    d14 = d3[2]
    dx314, dy314, dz314 = d14[0], d14[1], d14[2]
    d23 = d3[3]
    dx323, dy323, dz323 = d23[0], d23[1], d23[2]
    d24 = d3[4]
    dx324, dy324, dz324 = d24[0], d24[1], d24[2]
    d34 = d3[5]
    dx334, dy334, dz334 = d34[0], d34[1], d34[2]
    
    d4 = []
    for i in range(3):
        for j in range(i + 1, 4):
            d4.append(v4[3 * i : 3 * (i + 1)] - v4[3 * j : 3 * (j + 1)])
    
    d12 = d4[0]
    dx412, dy412, dz412 = d12[0], d12[1], d12[2]
    d13 = d4[1]
    dx413, dy413, dz413 = d13[0], d13[1], d13[2]
    d14 = d4[2]
    dx414, dy414, dz414 = d14[0], d14[1], d14[2]
    d23 = d4[3]
    dx423, dy423, dz423 = d23[0], d23[1], d23[2]
    d24 = d4[4]
    dx424, dy424, dz424 = d24[0], d24[1], d24[2]
    d34 = d4[5]
    dx434, dy434, dz434 = d34[0], d34[1], d34[2]
    
    L[0, 0] =        dx112 * dx112 + dy112 * dy112 + dz112 * dz112
    L[0, 1] = 2.0 *  (dx112 * dx212 + dy112 * dy212 + dz112 * dz212)
    L[0, 2] =        dx212 * dx212 + dy212 * dy212 + dz212 * dz212
    L[0, 3] = 2.0 *  (dx112 * dx312 + dy112 * dy312 + dz112 * dz312)
    L[0, 4] = 2.0 *  (dx212 * dx312 + dy212 * dy312 + dz212 * dz312)
    L[0, 5] =        dx312 * dx312 + dy312 * dy312 + dz312 * dz312
    L[0, 6] = 2.0 *  (dx112 * dx412 + dy112 * dy412 + dz112 * dz412)
    L[0, 7] = 2.0 *  (dx212 * dx412 + dy212 * dy412 + dz212 * dz412)
    L[0, 8] = 2.0 *  (dx312 * dx412 + dy312 * dy412 + dz312 * dz412)
    L[0, 9] =       dx412 * dx412 + dy412 * dy412 + dz412 * dz412
    
    
    L[1, 0] =        dx113 * dx113 + dy113 * dy113 + dz113 * dz113
    L[1, 1] = 2.0 *  (dx113 * dx213 + dy113 * dy213 + dz113 * dz213)
    L[1, 2] =        dx213 * dx213 + dy213 * dy213 + dz213 * dz213
    L[1, 3] = 2.0 *  (dx113 * dx313 + dy113 * dy313 + dz113 * dz313)
    L[1, 4] = 2.0 *  (dx213 * dx313 + dy213 * dy313 + dz213 * dz313)
    L[1, 5] =        dx313 * dx313 + dy313 * dy313 + dz313 * dz313
    L[1, 6] = 2.0 *  (dx113 * dx413 + dy113 * dy413 + dz113 * dz413)
    L[1, 7] = 2.0 *  (dx213 * dx413 + dy213 * dy413 + dz213 * dz413)
    L[1, 8] = 2.0 *  (dx313 * dx413 + dy313 * dy413 + dz313 * dz413)
    L[1, 9] =       dx413 * dx413 + dy413 * dy413 + dz413 * dz413
    
    
    L[2, 0] =        dx114 * dx114 + dy114 * dy114 + dz114 * dz114
    L[2, 1] = 2.0 *  (dx114 * dx214 + dy114 * dy214 + dz114 * dz214)
    L[2, 2] =        dx214 * dx214 + dy214 * dy214 + dz214 * dz214
    L[2, 3] = 2.0 *  (dx114 * dx314 + dy114 * dy314 + dz114 * dz314)
    L[2, 4] = 2.0 *  (dx214 * dx314 + dy214 * dy314 + dz214 * dz314)
    L[2, 5] =        dx314 * dx314 + dy314 * dy314 + dz314 * dz314
    L[2, 6] = 2.0 *  (dx114 * dx414 + dy114 * dy414 + dz114 * dz414)
    L[2, 7] = 2.0 *  (dx214 * dx414 + dy214 * dy414 + dz214 * dz414)
    L[2, 8] = 2.0 *  (dx314 * dx414 + dy314 * dy414 + dz314 * dz414)
    L[2, 9] =       dx414 * dx414 + dy414 * dy414 + dz414 * dz414
    
    
    L[3, 0] =        dx123 * dx123 + dy123 * dy123 + dz123 * dz123
    L[3, 1] = 2.0 *  (dx123 * dx223 + dy123 * dy223 + dz123 * dz223)
    L[3, 2] =        dx223 * dx223 + dy223 * dy223 + dz223 * dz223
    L[3, 3] = 2.0 *  (dx123 * dx323 + dy123 * dy323 + dz123 * dz323)
    L[3, 4] = 2.0 *  (dx223 * dx323 + dy223 * dy323 + dz223 * dz323)
    L[3, 5] =        dx323 * dx323 + dy323 * dy323 + dz323 * dz323
    L[3, 6] = 2.0 *  (dx123 * dx423 + dy123 * dy423 + dz123 * dz423)
    L[3, 7] = 2.0 *  (dx223 * dx423 + dy223 * dy423 + dz223 * dz423)
    L[3, 8] = 2.0 *  (dx323 * dx423 + dy323 * dy423 + dz323 * dz423)
    L[3, 9] =       dx423 * dx423 + dy423 * dy423 + dz423 * dz423
    
    
    L[4, 0] =        dx124 * dx124 + dy124 * dy124 + dz124 * dz124
    L[4, 1] = 2.0 *  (dx124 * dx224 + dy124 * dy224 + dz124 * dz224)
    L[4, 2] =        dx224 * dx224 + dy224 * dy224 + dz224 * dz224
    L[4, 3] = 2.0 * ( dx124 * dx324 + dy124 * dy324 + dz124 * dz324)
    L[4, 4] = 2.0 * (dx224 * dx324 + dy224 * dy324 + dz224 * dz324)
    L[4, 5] =        dx324 * dx324 + dy324 * dy324 + dz324 * dz324
    L[4, 6] = 2.0 * ( dx124 * dx424 + dy124 * dy424 + dz124 * dz424)
    L[4, 7] = 2.0 * ( dx224 * dx424 + dy224 * dy424 + dz224 * dz424)
    L[4, 8] = 2.0 * ( dx324 * dx424 + dy324 * dy424 + dz324 * dz424)
    L[4, 9] =       dx424 * dx424 + dy424 * dy424 + dz424 * dz424
    
    
    L[5, 0] =        dx134 * dx134 + dy134 * dy134 + dz134 * dz134
    L[5, 1] = 2.0 * ( dx134 * dx234 + dy134 * dy234 + dz134 * dz234)
    L[5, 2] =        dx234 * dx234 + dy234 * dy234 + dz234 * dz234
    L[5, 3] = 2.0 * ( dx134 * dx334 + dy134 * dy334 + dz134 * dz334)
    L[5, 4] = 2.0 * ( dx234 * dx334 + dy234 * dy334 + dz234 * dz334)
    L[5, 5] =        dx334 * dx334 + dy334 * dy334 + dz334 * dz334
    L[5, 6] = 2.0 *  (dx134 * dx434 + dy134 * dy434 + dz134 * dz434)
    L[5, 7] = 2.0 *  (dx234 * dx434 + dy234 * dy434 + dz234 * dz434)
    L[5, 8] = 2.0 *  (dx334 * dx434 + dy334 * dy434 + dz334 * dz434)
    L[5, 9] =       dx434 * dx434 + dy434 * dy434 + dz434 * dz434
   
    return L
    
def compute_rho(Cw):
    rho = []
    for i in range(3):
        for j in range(i + 1, 4):
            rho.append(dist2(Cw[i, :], Cw[j, :]))
             
    return rho

def compute_A_and_b_Gauss_Newton(cb, rho, L):
    A = np.zeros((6, 4))
    b = np.zeros((6, 1))
    
    B=[cb[0] * cb[0],
       cb[0] * cb[1],
       cb[1] * cb[1],
       cb[0] * cb[2],
       cb[1] * cb[2],
       cb[2] * cb[2],
       cb[0] * cb[3],
       cb[1] * cb[3],
       cb[2] * cb[3],
       cb[3] * cb[3]]
    
    for i in range(6):
        A[i, 0] = 2*cb[0] * L[i, 0] + cb[1] * L[i, 1] + cb[2] * L[i, 3] + cb[3] * L[i, 6]
        A[i, 1] = cb[0] * L[i, 1] + 2 * cb[1] * L[i, 2] + cb[2] * L[i, 4] + cb[3] * L[i, 7]
        A[i, 2] = cb[0] * L[i, 2] + cb[1] * L[i, 4] + 2 * cb[2] * L[i, 5] + cb[3] * L[i, 8]
        A[i, 3] = cb[0] * L[i, 3] + cb[1] * L[i, 7] + cb[2] * L[i, 8] + 2 * cb[3] * L[i, 9]
        
        b[i] = rho[i] - np.matmul(L[i, :], B)         

    return A, b

def dist2(v1, v2):
    return np.sum((v1 - v2) ** 2, axis=0)

def sign_determinant(C):
    M = []
    for i in range(3):
        M.append(C[i, :].T - C[-1, :].T)

    return np.sign(np.linalg.det(M))
    
    