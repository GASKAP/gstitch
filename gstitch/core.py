# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from reproject import reproject_from_healpix, reproject_interp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft
from scipy import interpolate
from tqdm import tqdm

#tmp
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter


def set_wcs(sizex,sizey, projx, projy, cdelt, GLON, GLAT):
    w           = wcs.WCS(naxis=2)
    w.wcs.crpix = [sizey/2,sizex/2]
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


    def get_v(self, hdr=None):
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

        return v


    def get_vrange(self, filename=None):
        #Read fitsnames
        my_file = open(filename, "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        vmin = []; vmax = []
        for i in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[i].split("/")[2]
            # print(fitsname)
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

            vmin.append(np.min(v))
            vmax.append(np.max(v))
        return np.min(vmin), np.max(vmax)


    def write_rms_maps(self, filename):
        #Read fitsnames
        my_file = open(filename, "r")        
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


    # def write_Tbmax_maps(self, filename):
    #     #Read fitsnames
    #     my_file = open(filename, "r")        
    #     # reading the file
    #     data = my_file.read()
    #     fitsnames = data.split("\n")[:-1]
    #     my_file.close()
        
    #     for i in np.arange(len(fitsnames)):
    #         #Load data
    #         fitsname = fitsnames[i].split("/")[2]
    #         print(fitsname)
    #         hdu = fits.open(self.path + fitsnames[i])
    #         hdr = hdu[0].header
    #         cube = hdu[0].data
            
    #         #Calculate rms map
    #         Tbmax = np.nanmax(cube,0)
    #         #Write RMS map
    #         pathout=self.path+"tmp/Tbmax/"
    #         hdu0 = fits.PrimaryHDU(Tbmax, header=hdr)
    #         hdulist = fits.HDUList([hdu0])
    #         hdulist.writeto(pathout + fitsname[:-5] + "_Tbmax.fits", overwrite=True)
        
    # def reproj_rms_maps(self, filename, target_header=None, size=None):
    #     #Read fitsnames
    #     my_file = open(filename, "r")        
    #     # reading the file
    #     data = my_file.read()
    #     fitsnames = data.split("\n")[:-1]
    #     my_file.close()
        
    #     for i in np.arange(len(fitsnames)):
    #         #Load data
    #         fitsname = fitsnames[i].split("/")[2]
    #         print(fitsname)
    #         hdu = fits.open(self.path + "tmp/RMS/" + fitsname[:-5]+"_RMS.fits")
    #         hdr = hdu[0].header
    #         w = wcs2D(hdr)
    #         rms = hdu[0].data
    #         #reproject                        
    #         reproj_rms, footprint = reproject_interp((rms,w.to_header()), target_header, shape_out=size)

    #         #write on disk
    #         pathout=self.path+"tmp/RMS/reproj/"
    #         hdu0 = fits.PrimaryHDU(reproj_rms, header=target_header)
    #         hdulist = fits.HDUList([hdu0])
    #         hdulist.writeto(pathout + fitsname[:-5] + "_RMS_large.fits", overwrite=True)

    # def stack_reproj_rms_maps(self, filename, target_header=None):
    #     #Read fitsnames
    #     my_file = open(filename, "r")        
    #     # reading the file
    #     data = my_file.read()
    #     fitsnames = data.split("\n")[:-1]
    #     my_file.close()
        
    #     rms = []
    #     for i in np.arange(len(fitsnames)):
    #         #Load data
    #         fitsname = fitsnames[i].split("/")[2]
    #         print(fitsname)
    #         hdu = fits.open(self.path + "tmp/RMS/reproj/" + fitsname[:-5] + "_RMS_large.fits")
    #         hdr = hdu[0].header
    #         w = wcs2D(hdr)
    #         rms.append(hdu[0].data)
    #     stack = np.nansum(rms,0)

    #     #write on disk
    #     pathout=self.path+"tmp/RMS/reproj/"
    #     hdu0 = fits.PrimaryHDU(stack, header=target_header)
    #     hdulist = fits.HDUList([hdu0])
    #     hdulist.writeto(pathout + "STACK" + "_RMS_large.fits", overwrite=True)

    def reproj_avePB_maps(self, filename, target_header=None, size=None):
        #Read fitsnames
        my_file = open(filename, "r")        
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
            w = wcs2D(hdr)
            avePB = hdu[0].data
            #reproject                        
            reproj_rms, footprint = reproject_interp((avePB,w.to_header()), target_header, shape_out=size)

            #write on disk
            pathout=self.path+"tmp/avePB/reproj/"
            hdu0 = fits.PrimaryHDU(reproj_rms, header=target_header)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(pathout + fitsname[:-5] + "_avePB_large.fits", overwrite=True)


    def stack_reproj_avePB_maps(self, filename, target_header=None):
        #Read fitsnames
        my_file = open(filename, "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        avePB = []
        for i in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[i].split("/")[2]
            hdu = fits.open(self.path + "tmp/avePB/reproj/" + fitsname[:-5] + "_avePB_large.fits")
            hdr = hdu[0].header
            w = wcs2D(hdr)
            avePB.append(hdu[0].data)

        fields = [avePB[i] * avePB[i] for i in np.arange(len(avePB))]
        stack = np.nansum(np.array(fields),0) / np.nansum(np.array(avePB),0)

        #write on disk
        pathout=self.path+"tmp/avePB/reproj/"
        hdu0 = fits.PrimaryHDU(stack, header=target_header)
        hdulist = fits.HDUList([hdu0])
        hdulist.writeto(pathout + filename[:-4] + "_large.fits", overwrite=True)


    def regrid(self, filename, beam=None, conv=True, verbose=False, check=False):
        #Read fitsnames
        my_file = open(filename, "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        for k in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[k].split("/")[2]
            if verbose == True: print(fitsname)
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
            target_wcs = set_wcs(sizey,sizex,hdr["CTYPE1"],hdr["CTYPE2"], reso, xc_world, yc_world)
            target_header = target_wcs.to_header()

            if verbose == True: print("Original shape = ", cube.shape)
            if verbose == True: print("New shape = ", (cube.shape[0],sizey,sizex))
            
            reproj_cube = np.zeros((cube.shape[0],sizey,sizex))
            if check == False:
                if verbose == True: print("Start spatial convolution")
                for i in tqdm(np.arange(cube.shape[0])):
                    if fwhm_input == fwhm_output:
                        # print("Keep native " + str(beam.value))
                        fconv = cube[i]
                    else:
                        # print("Convolve to " + str(beam.value))
                        if conv == True:
                            fconv = convolve_fft(cube[i], kernel, allow_huge=True)
                        else:
                            fconv = cube[i]
                    reproj, footprint = reproject_interp((fconv,w.to_header()), target_header, shape_out=(sizey,sizex))
                    reproj_cube[i] = reproj
                
            #Update original header                                                                           
            hdr["CRPIX1"] = target_header["CRPIX1"]
            hdr["CRPIX2"] = target_header["CRPIX2"]
            hdr["CRVAL1"] = target_header["CRVAL1"]
            hdr["CRVAL2"] = target_header["CRVAL2"]
            hdr["CDELT1"] = target_header["CDELT1"]
            hdr["CDELT2"] = target_header["CDELT2"]
            hdr["NAXIS1"] = reproj_cube.shape[2]
            hdr["NAXIS2"] = reproj_cube.shape[1]
            hdr["NAXIS3"] = reproj_cube.shape[0]

            #Write outpout                                                                                     
            if verbose == True: print("Write output " + fitsname + " file on disk")
            hdu0 = fits.PrimaryHDU(reproj_cube, header=hdr)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(self.path + "tmp/PPV/" + str(int(beam.value)) +  "arcsec/" + fitsname[:-5] + "_" + str(int(beam.value)) + ".fits", overwrite=True)


    def regrid_v(self, filename, target_dv=None, vmin=None, vmax=None, beam=None, check=False):
        #Read fitsnames
        my_file = open(filename, "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()
        
        for k in np.arange(len(fitsnames)):
            #Load data
            fitsname = fitsnames[k].split("/")[2]
            print(fitsname)
            hdu = fits.open(self.path + "tmp/PPV/" + str(int(beam.value)) +  "arcsec/" + fitsname[:-5] + "_" + str(int(beam.value)) + ".fits")
            hdr = hdu[0].header
            w = wcs2D(hdr)
            dv = np.abs(hdr["CDELT3"]*1.e-3)
            shape = hdu[0].data.shape
            cube = hdu[0].data

            #Regrid velocity axis to the SMC pilot 1 resolution                                                                   
            fwhm_factor = np.sqrt(8*np.log(2)) #FIXMEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
            current_resolution = dv * u.km/u.s
            target_resolution = target_dv #ALL TO THIS V RES
            pixel_scale = dv * u.km/u.s
            gaussian_width = ((target_resolution**2 - current_resolution**2)**0.5 /
                              pixel_scale / fwhm_factor)
            if target_dv.value > dv:
                kernel = Gaussian1DKernel(gaussian_width.value)
            else:
                kernel = None

            #Get v array cube
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
            
            #Convolution v and interpolation
            print("interp vmin = ", vmin, "interp vmax = ", vmax, "vmin = ", np.min(v), "vmax = ", np.max(v))
            uu = np.arange(vmin,vmax+target_resolution.value, target_resolution.value) #FIXME IF BUG
            #get position vmin and vmax                                                                              
            idx_min = np.where(uu > np.min(v))[0][0]
            idx_max = np.where(uu < np.max(v))[0][::-1][0]

            print("idmin", uu[idx_min], "idmax", uu[idx_max])
            print("cube shape = ", cube.shape)

            #Regrid v  
            regrid = np.zeros((len(uu),cube.shape[1],cube.shape[2]))
            print("cube regrid shape = ", regrid.shape)

            if check == False:
                for i in tqdm(np.arange(regrid.shape[1])):
                    for j in np.arange(regrid.shape[2]):
                        if cube[0,i,j] != cube[0,i,j]: continue
                        if cube[0,i,j] == 0.: continue #FIXME IF BUG
                        if target_dv.value < np.abs(dv):
                            y = cube[:,i,j]
                        else:
                            y = convolve_fft(cube[:,i,j], kernel, allow_huge=True)
                        f = interpolate.interp1d(v, y)
                        regrid[idx_min:idx_max,i,j] = f(uu[idx_min:idx_max])            

            #Update original header                                                                                           
            hdr["CRVAL3"] = uu[0] * 1.e3
            hdr["CDELT3"] = target_dv.value*1.e3
            hdr["NAXIS1"] = regrid.shape[2]
            hdr["NAXIS2"] = regrid.shape[1]
            hdr["NAXIS3"] = regrid.shape[0]

            #Write outpout                                                                                                  
            print("Write output " + fitsname + " file on disk")
            hdu0 = fits.PrimaryHDU(regrid, header=hdr)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(self.path + "tmp/PPV/" + str(int(beam.value)) +  "arcsec/1kms/" 
                            + fitsname[:-5] + "_" + str(round(target_dv.value,2)) + ".fits", 
                            overwrite=True)


    def stich_v(self, filename=None, filename_avePB=None, target_header=None, size=None, ID=None, 
                disk=False, verbose=False, target_dv=None, beam=None):
        #Read fitsnames avePB
        my_file = open(filename_avePB, "r")        
        # reading the file
        data = my_file.read()
        fitsnames_avePB = data.split("\n")[:-1]
        my_file.close()

        #Read fitsnames avePB
        my_file = open(filename, "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()

        weights = []
        for i in np.arange(len(fitsnames_avePB)):
            fitsname = fitsnames_avePB[i].split("/")[2]
            #Open reprojected map
            fnamen = self.path + "tmp/avePB/reproj/" + fitsname[:-5] + "_avePB_large.fits"
            if verbose == True: print("Opening ", fnamen)
            hdu = fits.open(fnamen)
            field_avePB = hdu[0].data
            field_avePB[field_avePB == 0.] = np.nan
            weights.append(field_avePB)

        rfields = []
        for i in np.arange(len(fitsnames)):
            fitsname = fitsnames[i].split("/")[2]
            #Load data
            fname = self.path + "tmp/PPV/" + str(int(beam.value)) + "arcsec/1kms/" + fitsname[:-5] + "_" + str(round(target_dv.value,2)) + ".fits"
            if verbose == True: print("Opening ", fname)
            hdu = fits.open(fname)
            hdr = hdu[0].header
            v = self.get_v(hdr)
            w = wcs2D(hdr)
            if verbose == True: print("shape = ", hdu[0].data.shape)
            if verbose == True: print("Stitching v = ", v[ID])
            field = hdu[0].data[ID]
            field[field == 0.] = np.nan

            #Reproject                        
            rfield, footprint = reproject_interp((field,w.to_header()), target_header, shape_out=size)
            rfields.append(rfield)            

        wfields = np.array([weights[i]*rfields[i] for i in np.arange(len(fitsnames))])
        cfield = np.nansum(wfields,0) / np.nansum(weights,0)
        cfield[cfield==0.] = np.nan
            
        if disk == True:
            #Write on disk
            pathout=self.path+"tmp/PPV/45arcsec/1kms/combined/"
            hdu0 = fits.PrimaryHDU(cfield, header=target_header)
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(pathout + "LMC_comined_v0.fits", overwrite=True)
        
        return cfield


    def stich_all(self, filename, target_header=None, size=None, ID_start=None, ID_end=None, 
                  disk=False):
        #Read fitsnames
        my_file = open(filename, "r")        
        # reading the file
        data = my_file.read()
        fitsnames = data.split("\n")[:-1]
        my_file.close()

        weights = []
        for i in np.arange(len(fitsnames)):
            fitsname = fitsnames[i].split("/")[2]
            #Open reprojected map
            fnamen = self.path + "tmp/avePB/reproj/" + fitsname[:-5] + "_avePB_large.fits"
            print("Opening ", fnamen)
            hdu = fits.open(fnamen)
            field_avePB = hdu[0].data
            field_avePB[field_avePB == 0.] = np.nan
            weights.append(field_avePB)

        nv = ID_end - ID_start
        cube = np.zeros((nv,size[0],size[1]))
        for k in tqdm(np.arange(nv)):
            cube[k] = self.stich_v(filename, target_header, size, ID=ID_start+k, disk=False, verbose=False)

        if disk == True:
            #Write on disk
            pathout=self.path+"tmp/PPV/45arcsec/1kms/combined/"
            hdu0 = fits.PrimaryHDU(cube, header=target_header) #FIXME 3D header
            hdulist = fits.HDUList([hdu0])
            hdulist.writeto(pathout + "Tb_LMC_comined.fits", overwrite=True)            
            
        return cube
                    
if __name__ == '__main__':    
    print("gstitch work in progress")

    # #Header large mosaic
    # glon = 288; glat = -40
    # c = SkyCoord(glon*u.deg, glat*u.deg, frame='galactic')
    # reso = 0.00333333
    # sizex = 5400; sizey = 10200
    # target_wcs = set_wcs(sizex, sizey, 'RA---TAN', 'DEC--TAN', reso, c.icrs.ra.value, c.icrs.dec.value)
    # target_header = target_wcs.to_header()
    # size = (sizex,sizey)

    #Init gstitch
    path="/priv/myrtle1/gaskap/downloads/"
    core = stitch(hdr=None, path=path)
    #Regrid the data to 30' but Nyquist sampling
    filename="./files_LMC-fg.txt"
    filename_avePB="./files_LMC-fg_avePB.txt"
    vmin, vmax = core.get_vrange(filename)
    print("vmin = ", vmin, "vmax = ", vmax)
    # core.write_rms_maps(filename)
    # core.reproj_rms_maps(filename, target_header, size)
    # core.stack_reproj_rms_maps(filename, target_header)

    #START HERE

    #LMC only for GASKAP                                                                           
    c = SkyCoord(75.895*u.deg, -69.676*u.deg, frame="icrs")
    reso = 0.00333333 * 2
    sizex = int(5400 / 2); sizey = int(4500 / 2)
    target_wcs = set_wcs(sizex, sizey, 'RA---TAN', 'DEC--TAN', reso, c.icrs.ra.value, c.icrs.dec.value)
    target_header = target_wcs.to_header()
    size = (sizex,sizey)

    beam = 60*u.arcsec
    target_dv = 1*u.km/u.s
    conv = True
    verbose = False
    check = False

    core.reproj_avePB_maps(filename_avePB, target_header, size)
    core.stack_reproj_avePB_maps(filename_avePB, target_header)
    core.regrid(filename, beam=beam, conv=conv, verbose=verbose, check=check)
    core.regrid_v(filename, target_dv=target_dv, vmin=vmin, vmax=vmax, beam=beam, check=check)
    # field = core.stich_v(filename, filename_avePB, target_header, size, ID=50, 
    #                      disk=False, verbose=verbose, target_dv=target_dv, beam=beam)
    # cube = core.stich_all(filename, target_header, size, ID_start=100, ID_end=110, disk=True)

    stop

    field_norm = np.arcsinh(field/np.nanmax(field) * 50)

    stop

    #Plot asinh Tb map
    fig = plt.figure(figsize=(8.5, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=target_wcs)
    ax.set_xlabel(r"RA", fontsize=18.)
    ax.set_ylabel(r"DEC", fontsize=18.)
    cm_inf = plt.get_cmap('cividis')
    cm_inf.set_bad(color='white')
    cm_inf.set_under(color='black')
    imkw_inf = dict(origin='lower', interpolation='none', cmap=cm_inf)
    img = ax.imshow(field_norm, vmin=0., vmax=4.5, **imkw_inf)
    colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
    cbar = fig.colorbar(img, cax=colorbar_ax)
    cbar.ax.tick_params(labelsize=14.) 
    cbar.set_label(r"$\rm{a}\sinh(T_b/T_b^{\rm max}\times 50)$", fontsize=18.) 
    plt.savefig("plot/" + 'Tb_example_v_285.85.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
    
    stop
