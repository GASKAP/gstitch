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
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel
from astropy.convolution import convolve, convolve_fft

from gstitch import stitch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

plt.ion()

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

#Init gstitch
path="/priv/myrtle1/gaskap/downloads/"
core = stitch(hdr=None, path=path)
#Regrid the data to 30' but Nyquist sampling
# filename="./files_LMC-fg.txt"
filename="./files_all_fg_tmp.txt"
filename_all="./files_all_fg_tmp.txt"
# filename_avePB="./files_LMC-fg_avePB.txt"
vmin, vmax = core.get_vrange(filename_all)
print("vmin = ", vmin, "vmax = ", vmax)

#Header large mosaic
glon = 287.7; glat = -38.7
c = SkyCoord(glon*u.deg, glat*u.deg, frame='galactic')
reso = 0.00333333 * 2
sizex = int(5800 / 2); sizey = int(10200 / 2)
target_wcs = set_wcs(sizex, sizey, 'RA---TAN', 'DEC--TAN', reso, c.icrs.ra.value, c.icrs.dec.value)
target_header = target_wcs.to_header()
size = (sizex,sizey)

beam = 60*u.arcsec
target_dv = 1*u.km/u.s
conv = True
verbose = False
check = False
disk = True
fileout = path + "tmp/PPV/60arcsec/1kms/combined/Tb_combined_all_large.fits" 

v = np.arange(vmin,vmax+target_dv.value, target_dv.value)
ID_start = 40#0
ID_end = 92#len(v)

#Open output cube
path_sd = "/home/amarchal/public_html/gaskap/downloads/tmp/PPV/60arcsec/1kms/combined/"
fitsname_sd = "GASS_HI_LMC_foreground_cube_1.0.fits"
hdu0 = fits.open(path_sd+fitsname_sd)
hdr0 = hdu0[0].header
w0 = wcs2D(hdr0)
cube_gass = hdu0[0].data

#Open output cube
hdu = fits.open(fileout)
hdr = hdu[0].header
w = wcs2D(hdr)
cube_askap = hdu[0].data

cube = np.where(cube_askap != cube_askap, cube_gass, cube_askap)

#Image rgb                                                                                                                     
img = np.zeros((cube.shape[0], cube.shape[1], cube.shape[2], 3), dtype=float)

for i in tqdm(np.arange(cube.shape[0]-2)):
    field1 = cube[i]
    field2 = cube[i+1]
    field3 = cube[i+2]
        
    img[i,:,:,0] += field1 / np.nanmax(field1) * 255 / 100
    img[i,:,:,1] += field1 / np.nanmax(field1) * 0 / 100
    img[i,:,:,2] += field1 / np.nanmax(field1) * 0 / 100
    
    img[i,:,:,0] += field2 / np.nanmax(field2) * 0 / 100
    img[i,:,:,1] += field2 / np.nanmax(field2) * 255 / 100
    img[i,:,:,2] += field2 / np.nanmax(field2) * 0 / 100

    img[i,:,:,0] += field3 / np.nanmax(field3) * 0 / 100
    img[i,:,:,1] += field3 / np.nanmax(field3) * 0 / 100
    img[i,:,:,2] += field3 / np.nanmax(field3) * 255 / 100
        
img[img != img] = 255

# kernel = Gaussian1DKernel(3)
smooth = np.zeros(img.shape)
for i in tqdm(np.arange(cube.shape[0]-1)):
    if i == 0: continue
    smooth[i] = 0.25 * img[i-1] + 0.5 * img[i] + 0.25 * img[i+1]

    #Plot integrated column density field TOT                                                                                     
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=w)
    ax.set_xlabel(r"RA", fontsize=18.)
    ax.set_ylabel(r"DEC", fontsize=18.)
    im = ax.imshow(smooth[i], origin="lower")
    plt.savefig("plot/RGB_large/" + 'RGB_{:02d}.png'.format(i), format='png', bbox_inches='tight', pad_inches=0.02, dpi=200)

stop

#Plot linear Tb map
fig = plt.figure(figsize=(8.5, 10))
ax = fig.add_axes([0.1,0.1,0.78,0.8], projection=target_wcs)
ax.set_xlabel(r"RA", fontsize=18.)
ax.set_ylabel(r"DEC", fontsize=18.)
cm_inf = plt.get_cmap('afmhot')
cm_inf.set_bad(color='white')
cm_inf.set_under(color='black')
imkw_inf = dict(origin='lower', interpolation='none', cmap=cm_inf)
img = ax.imshow(field, vmin=0., vmax=20, **imkw_inf)
colorbar_ax = fig.add_axes([0.89, 0.11, 0.02, 0.78])
cbar = fig.colorbar(img, cax=colorbar_ax)
cbar.ax.tick_params(labelsize=14.) 
cbar.set_label(r"$Tb$ (K)", fontsize=18.) 
plt.savefig("plot/" + 'Tb_example_v_LVC.png', format='png', bbox_inches='tight', pad_inches=0.02, dpi=400)
