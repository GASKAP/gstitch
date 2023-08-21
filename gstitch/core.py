# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from reproject import reproject_from_healpix, reproject_interp
from astropy import units as u
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from scipy import interpolate
from tqdm import tqdm

def set_wcs(sizex,sizey, projx, projy, cdelt, GLON, GLAT):
    w           = wcs.WCS(naxis=2)
    w.wcs.crpix = [sizex/2,sizey/2]
    w.wcs.crval = [GLON, GLAT]
    w.wcs.cdelt = np.array([-cdelt,cdelt])
    w.wcs.ctype = [projx, projy]
    return w

def wcs2D(hdr):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cdelt = np.array([hdr['CDELT1'], hdr['CDELT2']])
    w.wcs.crval = [hdr['CRVAL1'], hdr['CRVAL2']]
    w.wcs.ctype = [hdr['CTYPE1'], hdr['CTYPE2']]
    # w.wcs.Wcsprm = [hdr['LONPOLE'], hdr['LATPOLE']]
    return w

class stitch(object):
    def __init__(self, hdr=None, path=None):
        super(stitch, self).__init__()
        self.hdr = hdr if hdr is not None else None
        self.path = path if path is not None else "./"
        print("path data =", self.path)
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
        return 0

    def get_vrange(self):
        #Read fitsnames
        my_file = open("./files.txt", "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        vmin = []; vmax = []
        for i in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[i].split("/")[2]
            print(fitsname)
            hdu = fits.open(self.path + fitsnames[i])
            hdr = hdu[0].header

            dv = hdr["CDELT3"] / 1000.
            crval = hdr["CRVAL3"] / 1000.
            ctype = hdr["CTYPE3"]
            crpix = hdr["CRPIX3"] - 1
            
            naxis = hdr["NAXIS3"]
            
            x = np.arange(naxis)
            if ctype == 'FELO-LSR':
                clight = 2.99792458e5 # km/s 
                restfreq = hdr['RESTFREQ']
                crval2 = restfreq/(crval/clight + 1)
                df = -dv*crval2**2/(clight*restfreq)                
                f = (x-crpix)*df+crval2                
                v = clight*(restfreq - f)/f #optical definition
            else:
                v = (x-crpix)*dv+crval
                
            print(np.min(v),np.max(v))
            vmin.append(np.min(v))
            vmax.append(np.max(v))
        return np.min(vmin), np.max(vmax)

    def write_rms_maps(self):
        #Read fitsnames
        my_file = open("./files.txt", "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        for i in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[i].split("/")[2]
            print(fitsname)
            hdu = fits.open(self.path + fitsnames[i])
            hdr = hdu[0].header
            cube = hdu[0].data
            
            #Calculate rms map
            rms = np.nanstd(cube[::-1][:10],0)
            #Write RMS map
            pathout=self.path+"tmp/RMS/"
            hdu0 = fits.PrimaryHDU(rms, header=hdr)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(pathout + fitsname[:-5] + "_RMS.fits", overwrite=True)

    def write_Tbmax_maps(self):
        #Read fitsnames
        my_file = open("./files.txt", "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        for i in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[i].split("/")[2]
            print(fitsname)
            hdu = fits.open(self.path + fitsnames[i])
            hdr = hdu[0].header
            cube = hdu[0].data
            
            #Calculate rms map
            Tbmax = np.nanmax(cube,0)
            #Write RMS map
            pathout=self.path+"tmp/Tbmax/"
            hdu0 = fits.PrimaryHDU(Tbmax, header=hdr)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(pathout + fitsname[:-5] + "_Tbmax.fits", overwrite=True)
        
    def reproj_rms_maps(self):
        #Read fitsnames
        my_file = open("./files.txt", "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        for i in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[i].split("/")[2]
            print(fitsname)
            hdu = fits.open(self.path + "tmp/RMS/" + fitsname[:-5]+"_RMS.fits")
            hdr = hdu[0].header
            w = wcs2D(hdr)
            rms = hdu[0].data
            #reproject
            #Header large mosaic
            glon=52.940314716173; glat=-72.14763039399
            reso = 0.005
            sizey = int(10000); sizex = int(8000)
            target_wcs = set_wcs(sizex,sizey,'RA---TAN','DEC--TAN', reso, glon, glat)
            target_hdr = target_wcs.to_header()
            target_hdr["CRPIX1"] -= 460
            target_hdr["CRPIX2"] -= 3230
            sizex-=3230+1350; sizey-=460+2970
            
            reproj_rms, footprint = reproject_interp((rms,w.to_header()), target_hdr, shape_out=(sizex,sizey))
            #write on disk
            pathout=self.path+"tmp/RMS/reproj/"
            hdu0 = fits.PrimaryHDU(reproj_rms, header=target_hdr)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(pathout + fitsname[:-5] + "_RMS_large.fits", overwrite=True)


    def regrid(self, beam=None):
        #Read fitsnames
        my_file = open("./files.txt", "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        for k in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[k].split("/")[2]
            print(fitsname)
            hdu = fits.open(self.path + fitsnames[k])
            hdr = hdu[0].header
            w = wcs2D(hdr)
            dv = np.abs(hdr["CDELT3"]*1.e-3)
            shape = hdu[0].data.shape
            cube = hdu[0].data
        
            #Convolved and regrid one channel map                                                                     
            fwhm_input = 30*u.arcsec #native beam size
            cdelt = hdr["CDELT2"]*u.deg
            fwhm_output = beam #NEW beam size                                                         
            fwhm = np.sqrt(fwhm_output**2. - fwhm_input**2.)
            std = fwhm.to(u.rad) / np.sqrt(8 * np.log(2))
            stdpix = (std / (np.radians(cdelt))).decompose()
            kernel = Gaussian2DKernel(stdpix.value,stdpix.value)
            
            xc = int(hdr["NAXIS1"]/2)
            yc = int(hdr["NAXIS2"]/2)
            sky = w.pixel_to_world(xc,yc)
            xc_world = sky.ra.value
            yc_world = sky.dec.value
            
            reso_nquist = fwhm_output / 2.5
            reso = reso_nquist.to(u.deg).value
            ratio = (reso/cdelt).value
            sizey = int(hdr["NAXIS2"]/ratio); sizex = int(hdr["NAXIS1"]/ratio)
            target_wcs = set_wcs(sizex,sizey,hdr["CTYPE1"],hdr["CTYPE2"], reso, xc_world, yc_world)
            target_header = target_wcs.to_header()

            print("Original shape = ", cube.shape)
            print("New shape = ", (cube.shape[0],sizey,sizex))
            
            reproj_cube = np.zeros((cube.shape[0],sizey,sizex))
            print("Start spatial convolution")
            for i in np.arange(cube.shape[0]):
            # for i in np.arange(2):
                if fwhm_input == fwhm_output:
                    # print("Keep native " + str(beam.value))
                    conv = cube[i]
                else:
                    # print("Convolve to " + str(beam.value))
                    conv = convolve_fft(cube[i], kernel, allow_huge=True)
                reproj, footprint = reproject_interp((conv,w.to_header()), target_header, shape_out=(sizey,sizex))
                reproj_cube[i] = reproj
                
            #Update original header                                                                           
            hdr["CRPIX1"] = target_header["CRPIX1"]
            hdr["CRPIX2"] = target_header["CRPIX2"]
            # hdr["CRVAL3"] = xnew[0]
            hdr["CRVAL1"] = target_header["CRVAL1"]
            hdr["CRVAL2"] = target_header["CRVAL2"]
            hdr["CDELT1"] = target_header["CDELT1"]
            hdr["CDELT2"] = target_header["CDELT2"]
            # hdr["CDELT3"] = dvnew*1.e3
            hdr["NAXIS1"] = reproj_cube.shape[2]
            hdr["NAXIS2"] = reproj_cube.shape[1]
            hdr["NAXIS3"] = reproj_cube.shape[0]

            #Write outpout                                                                                     
            print("Write output " + fitsname + " file on disk")
            hdu0 = fits.PrimaryHDU(reproj_cube, header=hdr)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(path + "tmp/PPV/" + str(int(beam.value)) +  "arcsec/" + fitsname[:-5] + "_" + str(int(beam.value)) + ".fits", overwrite=True)

                    
if __name__ == '__main__':    
    print("Test gstitch")

    # #Call ROHSApy
    path="/priv/myrtle1/gaskap/downloads/"
    core = stitch(hdr=None, path=path)
    core.run()
    #Regrid the data to 30' but Nyquist sampling
    vmin, vmax = core.get_vrange()
    core.regrid(beam=30*u.arcsec)
    stop    
    
    # core.write_rms_maps()
    # core.write_Tbmax_maps()
    # core.reproj_rms_maps()

    print("Start stitching")

