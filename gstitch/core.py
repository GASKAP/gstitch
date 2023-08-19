# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

class stitch(object):
    def __init__(self, hdr=None, path_data=None):
        super(stitch, self).__init__()
        self.hdr = hdr if hdr is not None else None
        self.path_data = path_data if path_data is not None else "./"
        print("path data =", self.path_data)
        if self.hdr is not None : 
            dv = self.hdr["CDELT3"] / 1000.
            crval = self.hdr["CRVAL3"] / 1000.
            ctype = self.hdr["CTYPE3"]
            crpix = self.hdr["CRPIX3"] - 1
            
            naxis = self.hdr["NAXIS3"]
            
            x = np.arange(naxis)
            if ctype == 'FELO-LSR':
                clight = 2.99792458e5 # km/s 
                restfreq = self.hdr['RESTFREQ']
                crval2 = restfreq/(crval/clight + 1)
                df = -dv*crval2**2/(clight*restfreq)                
                f = (x-crpix)*df+crval2                
                self.v = clight*(restfreq - f)/f #optical definition
            else:
                self.v = (x-crpix)*dv+crval

    def run(self):
        print("Read mosaic coverage")
        print("Start stitching")
                
        
if __name__ == '__main__':    
    print("Test gstitch")
    #Load data
    path="/home/amarchal/public_html/gaskap/downloads/"
    release = "release-v2.0/"
    field = "LMC-5/"
    filename = "LMC-5_askap_parkes_PBC_K_full.fits"
    hdu = fits.open(path + release + field + filename)
    hdr = hdu[0].header

    # #Call ROHSApy
    core = stitch(hdr=hdr, path_data=path)
    core.run()
