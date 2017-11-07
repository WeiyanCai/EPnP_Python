# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 19:57:55 2017

@author: weiya
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import EPnP


class EPnPTest(object):
    def __init__(self):
        self.load_test_data()
        self.epnp = EPnP.EPnP()

    def load_test_data(self):        
        data = sio.loadmat('input_data_noise.mat')

        self.A = data['A']
        self.Rt = data['Rt']

        points = data['point']
        self.n = len(points[0])
        self.Xcam, self.Ximg_true, self.Ximg_pix_true, self.Ximg, self.Ximg_pix, self.Xworld= [], [], [], [], [], []
        for p in points[0]:
            self.Xcam.append(p[0])
            self.Ximg_true.append(p[1])
            self.Ximg_pix_true.append(p[2])
            self.Ximg.append(p[3])
            self.Ximg_pix.append(p[4])
            self.Xworld.append(p[5])
                    
    def draw_input_noisy_data(self):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        for p in self.Ximg_pix_true:
            plt.plot(p[0], p[1], '.r')
        for p in self.Ximg_pix:
            plt.plot(p[0], p[1], 'xg')
        axes.set_title('Noise in Image Plane', fontsize=18)
        plt.grid()
        
        fig.savefig('Noise_in_Image_Plane.png', dpi=100)
        plt.show()
        
    def apply_EPnP(self):
        error, Rt, Cc, Xc = self.epnp.efficient_pnp(np.array(self.Xworld), np.array(self.Ximg_pix), self.A)
        self.plot_3d_reconstruction("EPnP (Old)", Xc)
        print "Error of EPnP: ", error
        
    def apply_EPnP_Gauss(self):
        self.epnp.efficient_pnp_gauss(np.array(self.Xworld), np.array(self.Ximg_pix), self.A)
        #self.plot_3d_reconstruction("EPnP (Gauss)", Xc)
        #print "Error of EPnP - Gauss Optimization: ", error
        
    def plot_3d_reconstruction(self, method, Xc):
        fig = plt.figure()
        fig.set_size_inches(18.5, 13)
        axes = fig.add_subplot(1, 1, 1)
        for p in self.Xcam:
            plt.plot(p[0], p[1], '.r')
        for p in Xc:
            plt.plot(p[0], p[1], 'xg')
        axes.set_title(method + ' - Reprojection Error', fontsize=18)
        plt.grid()
        
        fig.savefig(method + '_Reprojection_Error.png', dpi=100)
        plt.show()
    

if __name__ == "__main__":
    ET = EPnPTest()
    ET.load_test_data()
    #ET.draw_input_noisy_data()
    #ET.apply_EPnP()
    ET.apply_EPnP_Gauss()