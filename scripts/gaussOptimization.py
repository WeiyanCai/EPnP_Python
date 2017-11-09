# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 16:11:15 2017

@author: Weiyan Cai
"""

import numpy as np
import constraintMatrix as cM


def gauss_newton(Kernel, Cw, Beta0):
    L = cM.compute_L6_10(Kernel)
    rho = cM.compute_rho(Cw)
    
    current_betas = Beta0
    
    n_iterations = 10
    
    for k in range(n_iterations):
        A, b = compute_A_and_b_Gauss_Newton(current_betas, rho, L)
        dbeta = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
        current_betas = current_betas + dbeta.T[0]
        error = np.matmul(b.T, b)
    
    Beta_opt = current_betas
    
    return Beta_opt, error

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
        A[i, 0] = 2 * cb[0] * L[i, 0] + cb[1] * L[i, 1] + cb[2] * L[i, 3] + cb[3] * L[i, 6]
        A[i, 1] = cb[0] * L[i, 1] + 2 * cb[1] * L[i, 2] + cb[2] * L[i, 4] + cb[3] * L[i, 7]
        A[i, 2] = cb[0] * L[i, 2] + cb[1] * L[i, 4] + 2 * cb[2] * L[i, 5] + cb[3] * L[i, 8]
        A[i, 3] = cb[0] * L[i, 3] + cb[1] * L[i, 7] + cb[2] * L[i, 8] + 2 * cb[3] * L[i, 9]
        
        b[i] = rho[i] - np.matmul(L[i, :], B)         

    return A, b

def sign_determinant(C):
    M = []
    for i in range(3):
        M.append(C[i, :].T - C[-1, :].T)

    return np.sign(np.linalg.det(M))
