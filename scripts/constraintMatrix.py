# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 22:24:16 2017

@author: Weiyan Cai
"""

import numpy as np


def dist2(v1, v2):
	return np.sum((v1 - v2) ** 2, axis=0)

def compute_rho(Cw):
    rho = []
    for i in range(3):
        for j in range(i + 1, 4):
            rho.append(dist2(Cw[i, :], Cw[j, :]))
    return rho

def compute_L6_10(K):
    L = np.zeros((6, 10))

    v = []
    for i in range(4):
        v.append(K[:, i])

    dv = []

    for r in range(4):
        dv.append([])
        for i in range(3):
            for j in range(i+1, 4):
                dv[r].append(v[r][3*i:3*(i+1)]-v[r][3*j:3*(j+1)])

    index = [
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3)
        ]

    for i in range(6):
        j = 0
        for a, b in index:
            L[i, j] = np.matmul(dv[a][i], dv[b][i].T)
            if a != b:
                L[i, j] *= 2
            j += 1

    return L

# (2, 2) (2, 3) (3, 3)
def compute_L6_3(L6_10):
	return L6_10[:, (5, 8, 9)]

# (1, 1) (1, 2) (1, 3) (2, 2) (2, 3) (3, 3)
def compute_L6_6(L6_10):
	return L6_10[:, (2, 4, 7, 5, 8, 9)]
