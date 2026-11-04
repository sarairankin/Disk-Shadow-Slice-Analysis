# Full Pipeline - OFFICIAL
# File Management Imports
import os

# Figure Imports
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import matplotlib.pylab as plt
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import seaborn as sns
cp = sns.color_palette()
from astropy.io import ascii

# Shadow Data Imports
import numpy as np
from astropy.nddata.utils import Cutout2D
from scipy.ndimage import gaussian_filter

# Shadow Model Imports
from scipy.integrate import simpson as simps

# Analysis Imports
import pickle
import corner
import warnings
warnings.filterwarnings("ignore")

# Retrieval Imports
from astropy.modeling.functional_models import Gaussian1D
from scipy.signal import fftconvolve
import emcee
from multiprocessing import Pool
from scipy.optimize import minimize, LinearConstraint

AU = 1.496e13  #cm

def shadow(h_r, offset, pl=-0.63978, plh=0.21125, sigmadust0=1e-5,
           rdisk=25*AU,rinner=0.1*AU,hr0=0.34, g2d=100, Cext=6000.): #Cext corresponds to WD5.5

    nr = 50    
    
    rr = np.logspace(np.log10(rinner), np.log10(rdisk), num=nr)
#    tt = np.arccos(h_r+offset)
    tt = np.pi/2.0 - np.arctan(h_r+offset)
    nt = tt.shape[0]

    hr = hr0 * (rr/rdisk)**plh
    sigdust = sigmadust0 * (rr/rdisk)**pl

    rhodust = np.zeros((nr,nt))
    for ir in np.arange(nr):
        rhodust[ir,:] = 3.98942280401e-1 * sigdust[ir] * \
                        np.exp(-0.5*((np.pi/2.0-tt)/hr[ir])**2 ) / (hr[ir]*rr[ir])
                        
    tau = np.zeros(nt)
    for it in np.arange(nt):
        tau[it] = simps(rhodust[:,it]*Cext,rr)
        
    mtot = simps(sigdust*2.0*np.pi*rr,rr)*g2d

    return tau

class ShadowDataHST():
    ''' 
    Inputs
        ri              Pixel Distance from Host Star Center 
        boxwidth        Slice Width (dr)
        image_file      Cleaned Image
        loc             Location of Host Star (x,y coord)
        max_angle       Shadow Angle Maximum
        delta_angle     Shadow Angle Thickness
        xs              Scale Height Pixel Range
        ys              Scale Height Pixel Range
    Returns
        Extraction of Shadow Info at Given Slice
    '''
    def __init__(self,ri=40,boxwidth=15,image_file='shadow_data.py.fits',loc=(7084,4374), upper_h_r = 0.4, lower_h_r = -0.4, xs=600,ys=400):
        
        image = fits.getdata(image_file,0)
        hdr = fits.getheader(image_file,0)
        exptime = hdr['EXPTIME']

        image_cut = Cutout2D(image,loc,(np.ceil(ys),xs))
        image_slice = np.median(image_cut.data[:,int(xs/2+ri-boxwidth/2):int(xs/2+ri+boxwidth/2)],1) 
        image_slice = image_slice/np.nanmax(image_slice) #normalize so we don't get a dependence on filter choice. 
        image_slice = gaussian_filter(image_slice, sigma=1.0)
        
        rr = np.arange(int(xs/2))
        hh = np.arange(ys)-ys/2
        h_r = hh/ri

        # Fixed Edge Range
        gsubs = np.where((h_r>lower_h_r) & (h_r<upper_h_r) & np.isfinite(image_slice))
        h_r_tr = h_r[gsubs]
        image_slice_tr = image_slice[gsubs] 

        locont = np.nanmax(image_slice[np.where(np.round(h_r) > lower_h_r)])
        hicont = np.nanmax(image_slice[np.where(np.round(h_r) < upper_h_r)])
        cont = (hicont-locont)/(upper_h_r-lower_h_r)*(h_r_tr-lower_h_r)+locont
        
        # Edge Search Boundaries
        try:
            losubs =  np.where((h_r>lower_h_r) & (h_r<lower_h_r+0.25))
            hisubs =  np.where((h_r<upper_h_r) & (h_r>upper_h_r-0.25))
            lobord = np.nanmax(image_slice[losubs])
            hibord = np.nanmax(image_slice[hisubs])
            hiarg = np.where(image_slice==hibord)
            loarg = np.where(image_slice==lobord)
            hiarg = np.where(image_slice==hibord)
            loarg = np.where(image_slice==lobord)
            hiang = h_r[hiarg]
            loang = h_r[loarg]
            self.bounds = [loang, hiang]      # Shadow Edges (H/R)
            self.bounds_slice = [loarg, hiarg]  # Shadow Edges Intensity IDX for plotting 
        except ValueError:
            print('Undefined Edge at Slice')
            self.bounds = [0, 0]
            self.bounds_slice = [0, 0]
        
            
    #        plt.plot(h_r, image_slice)
    #        plt.plot(h_r_tr, image_slice_tr)
    #        plt.plot(h_r_tr, cont)
    #        plt.show()

        
        self.full_h_r = h_r
        self.full_slice = image_slice
        
        self.cont = cont
        self.h_r = h_r_tr
        self.slice = image_slice_tr

        #Calculating covariance array, assuming shot noise only
        self.c_uncorr = np.diag(np.sqrt(np.abs(image_slice_tr)*exptime))/exptime/np.sqrt(boxwidth)
        self.npix = h_r_tr.size
        
class ShadowDataJWST():
    def __init__(self,ri=40,boxwidth=40,image_file='serpens_f140m_i2d_joel.fits',loc=(7084,4374), upper_h_r = 0.4, lower_h_r = -0.4, xs=1500,ys=1000):
        image = fits.getdata(image_file,0)
        hdr = fits.getheader(image_file,0)
        if'DURATION' in hdr:
            exptime = hdr['DURATION'] # s
        elif 'XPOSURE' in hdr:
            exptime = hdr['XPOSURE'] # s
            
        image_cut = Cutout2D(image,loc,(np.ceil(ys),xs))
        image_slice = np.median(image_cut.data[:,int(xs/2+ri-boxwidth/2):int(xs/2+ri+boxwidth/2)],1) 
        # Center (x/2) move to px (+ ri) extend to centered boxwidth
        image_slice = image_slice/np.nanmax(image_slice) #normalize so we don't get a dependence on filter choice.     
        image_slice = gaussian_filter(image_slice, sigma=1.0)
        
        rr = np.arange(int(xs/2))
        hh = np.arange(ys)-ys/2                                             # Slice From Top to Bottom of Cutout, Centers x (h_r) coordinates on 0
        h_r = hh/ri                                                         # Scale x axis (Height) in terms of Height Ratio
        
        # Fixed Edge Range
        gsubs = np.where((h_r>lower_h_r) & (h_r<upper_h_r) & np.isfinite(image_slice)) # Set data extraction bounds between two fixed scale heights on slice
        h_r_tr = h_r[gsubs]                                                            # (Truncated) H_r Values within bounds
        image_slice_tr = image_slice[gsubs]                                            # Slice Intensity Values within bounts Truncated

        locont = np.nanmax(image_slice[np.where(np.round(h_r) > lower_h_r)])
        hicont = np.nanmax(image_slice[np.where(np.round(h_r) < upper_h_r)])
        cont = (hicont-locont)/(upper_h_r-lower_h_r)*(h_r_tr-lower_h_r)+locont
        
        # Edge Search Boundaries
        try:
            losubs =  np.where((h_r>lower_h_r) & (h_r<lower_h_r+0.25))
            hisubs =  np.where((h_r<upper_h_r) & (h_r>upper_h_r-0.25))
            lobord = np.nanmax(image_slice[losubs])
            hibord = np.nanmax(image_slice[hisubs])
            hiarg = np.where(image_slice==hibord)
            loarg = np.where(image_slice==lobord)
            hiarg = np.where(image_slice==hibord)
            loarg = np.where(image_slice==lobord)
            hiang = h_r[hiarg]
            loang = h_r[loarg]
            self.bounds = [loang, hiang]        # Shadow Edges (H/R)
            self.bounds_slice = [loarg, hiarg]  # Shadow Edges Intensity IDX for plotting
        except ValueError:
            print('Undefined Edge at Slice')
            self.bounds = [0, 0]

        # Slope to Shadow Floor (bg is Shadow Plateau Level)
    #        plt.plot(h_r, image_slice)
    #        plt.plot(h_r_tr, image_slice_tr)
    #        plt.plot(h_r_tr, cont)
    #        plt.show()
        
        self.full_h_r = h_r
        self.full_slice = image_slice
        
        self.cont = cont
        self.h_r = h_r_tr
        self.slice = image_slice_tr

        #Calculating covariance array, assuming shot noise only
        self.c_uncorr = np.diag(np.sqrt(np.abs(image_slice_tr)*exptime))/exptime/np.sqrt(boxwidth)
        self.npix = h_r_tr.size

class Retrieval():
    '''
    Class for handling the Bayesian retrieval using the emcee package.
    "Fixed" Parameters: PL. PLH, logSigmaDust, Router, Rinner
    Varying Parameters: PA, HR0, BG, BGslope
    '''
    def __init__(self, Data, disk_params):
 
        self.AU = 1.496e13 #cm
        self.fwhm = 0.151/0.08
        
        pix_array = np.linspace(-5,5,11)
        kernel_model = Gaussian1D(amplitude=1,mean=0,stddev=self.fwhm/2.355)
        self.kernel = kernel_model(pix_array)
        self.kernel /= np.sum(self.kernel)
 
        self.Data = Data
        self.disk_params = disk_params
        self.nthreads = 10
        
    def run(self, theta0=np.array([0.05, 0.3, 0.15, 0.02])):
        nwalkers = 15
        ndim = 4
        pos = [theta0 + theta0*1e-2*np.random.randn(ndim) for i in range(nwalkers)]

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob, pool=pool)
            sampler.run_mcmc(pos, 2500, progress=True)
        '''
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lnprob)
        sampler.run_mcmc(pos, 2000, progress=True)
        '''
        samples = sampler.chain[:, 100:, :].reshape((-1, ndim))     
        sample_chain = sampler.get_chain(discard=100, flat=False, thin=50)
        return samples, sample_chain
    
    def _residual(self, theta):
        # Subtracts shadow Model from datapoints extracted from image
        theta_int = theta['int']
        theta_ext = theta['ext']
                        
        tau = shadow(self.Data.h_r, offset=theta_int['offset'], pl=self.disk_params['pl'],
                      plh=self.disk_params['plh'], sigmadust0=self.disk_params['sigmadust0'],
                      rdisk=self.disk_params['rdisk'], rinner=self.disk_params['rinner'],
                      hr0=theta_int['hr0'])
        bg = theta_ext['bg']+theta_ext['bg_slope']*self.Data.h_r

        model = (self.Data.cont-bg)*np.exp(-tau)+bg
        model_conv = np.copy(model)
        model_conv[3:-3] = fftconvolve(model,self.kernel,mode='same')[3:-3]

        return (self.Data.slice - model)
        
    def _lnlike(self, theta):
        
        c = self.Data.c_uncorr
        r = self._residual(theta)

        lnlike = -0.5*(np.matmul(np.transpose(r),np.matmul(np.linalg.inv(c),r)) + np.log(np.linalg.det(c) + self.Data.npix*np.log(2.0*np.pi)))

        if np.isinf(r).any() == True:
            lnlike = 0

        return lnlike
     
    def _lnprior(self, theta):
        prior = -np.inf
        # Postion Angle Offset
        if -0.4>theta['int']['offset'] or theta['int']['offset']>0.4:
            return -np.inf
        # Model Scale Height (at a given radii)
        elif 0.05>theta['int']['hr0'] or theta['int']['hr0']>0.65:
            return -np.inf  
        # Background Intensity
        elif 0.05>theta['ext']['bg'] or theta['ext']['bg']>0.62:
            return -np.inf       
        # Background Intensity Slope    
        elif -0.4>theta['ext']['bg_slope'] or theta['ext']['bg_slope']>0.4:
            return -np.inf           
        else:
            return 0
        
    def _lnprob(self, theta_vec):

        theta_int = {'offset':theta_vec[0], 'hr0':theta_vec[1]}
        theta_ext = {'bg':theta_vec[2],'bg_slope':theta_vec[3]}           
        theta = {'int':theta_int,'ext':theta_ext}
        lp = self._lnprior(theta)

        if not np.isfinite(lp):
            return -np.inf
         
        return lp + self._lnlike(theta)

class ShadowSlicer():
    '''
    Using a PSF masked, North Up fits file to extract a disk shadow of 6-46 lightdays away
    '''
    def __init__(self, target_name, image_file, target_coordinates, pixel_size,
                 light_days = np.linspace(6, 46, 21), 
                 easternlobe = False, westernlobe = False,
                 distance = 436.0, angle0=47,
                 disk_params={'pl':-0.63978, 'plh':0.21125, 'sigmadust0':1e-5,
                              'rdisk':25*AU,'rinner':0.1*AU}):
        # Constants (CGS)
        self.AU = 1.496e13         # cm
        self.CC = 2.998e10         # cm/s
        self.distance = distance   # pc
        self.disk_params=disk_params

        # Lobe
        self.westernlobe = westernlobe
        self.easternlobe = easternlobe
        
        # Image Selection + Save Paths
        self.image_file = image_file
        self.target_name = target_name
        self.filter_name = image_file.split('/')[-1][:5]
        self.save_path = self.target_name + '_' + self.filter_name + '/'
        os.makedirs(self.save_path, exist_ok=True)
        self.labels = ['offset', 'hr0', 'bg','bg_slope']

        # Light Day + Pixel Slice Indices
        self.loc = target_coordinates
        self.pixsize = pixel_size
        self.light_days = light_days
        self.ris = np.ceil(light_days * 3600 * 24 * self.CC / self.AU / self.distance / self.pixsize) # Radius Indices
        self.boxwidth = 2 * 3600 * 24 * self.CC / self.AU / self.distance / self.pixsize              # Boxwidth: 2 Light Days Across
        self.xs = 2 * self.ris[-1] + (8 * 3600 * 24 * self.CC / self.AU / self.distance / self.pixsize)
        self.ys = 1.5*self.ris[-1] + (10 * 3600 * 24 * self.CC / self.AU / self.distance / self.pixsize)

        self.lower_bound_west = { '5': -0.75,  '6': -0.70,  '8': -0.65, '10': -0.65, '12': -0.65, 
                                  '14': -0.75, '16': -0.60, '18': -0.50, '20': -0.60, '22': -0.50,
                                  '24': -0.7,  '26': -0.60, '28': -0.55, '30': -0.50, '32': -0.55,
                                  '34': -0.50, '36': -0.50, '38': -0.45, '40': -0.55, '42': -0.55, 
                                  '44': -0.60, '46': -0.50}
        self.upper_bound_west = { '5':  0.95,  '6': 0.80,   '8': 0.60,  '10': 0.55, '12': 0.45, 
                                 '14':  0.40, '16': 0.30,  '18': 0.30,  '20': 0.30, '22': 0.35, 
                                 '24':  0.60, '26': 0.35,  '28': 0.35,  '30': 0.35, '32': 0.35, 
                                 '34':  0.35, '36': 0.30,  '38': 0.3,  '40': 0.30, '42': 0.30,
                                 '44':  0.30, '46': 0.30}
        self.lower_bound_east = { '5': -0.65,  '6': -0.60,  '8': -0.70, '10': -0.60, '12': -0.60,
                                 '14': -0.50, '16': -0.45, '18': -0.50, '20': -0.50, '22': -0.40, 
                                 '24': -0.80,  '26': -0.70, '28': -0.65, '30': -0.60, '32': -0.65,
                                '34': -0.60, '36': -0.60, '38': -0.55, '40': -0.65, '42': -0.65, 
                                '44': -0.70, '46': -0.60}
        self.upper_bound_east = { '5':  0.65,  '6':  0.75,  '8': 0.60,  '10': 0.60,  '12': 0.70, 
                                 '14':  0.60, '16':  0.60, '18': 0.35,  '20': 0.30,  '22': 0.30,
                                 '24':  0.70, '26': 0.45,  '28': 0.45,  '30': 0.45, '32': 0.45, 
                                 '34':  0.45, '36': 0.40,  '38': 0.40,  '40': 0.40, '42': 0.40,
                                 '44':  0.40, '46': 0.40}
        
    def image_cutout(self,):
        os.makedirs(self.save_path + 'slice_selections/', exist_ok=True)
        # Save Image of Shadow Selection
        if self.pixsize >= 0.07:
            ShadowData = ShadowDataHST
        elif self.pixsize <= 0.07:
            ShadowData = ShadowDataJWST
        image_data = fits.getdata(self.image_file, 0)
        
        # Full Cutout
        fig = plt.figure(figsize=(10, 5))
        image_selection = Cutout2D(image_data, self.loc, (self.ys, self.xs), mode='partial')
        normalized_selection = image_selection.data/np.nanmean(image_selection.data)
        plt.scatter(self.xs/2, self.ys/2, marker='*', lw=2, color = 'orange')
        plt.imshow(normalized_selection, cmap="gray", origin='lower', vmin=0, vmax=3)
        for i in np.arange(len(self.ris)):
            # Western Slice Markers
            ri = self.ris[i]
            Data = ShadowData(ri=ri,image_file=self.image_file, loc=self.loc, 
                                            upper_h_r = self.upper_bound_west[str(self.light_days[i])], lower_h_r = self.lower_bound_west[str(self.light_days[i])],
                                            xs=self.xs, ys=self.ys)
            plt.scatter(self.xs/2 + ri, self.ys/2 + Data.bounds[1]*ri, marker='v', lw=1, color = 'orange', zorder=3)
            plt.scatter(self.xs/2 + ri, self.ys/2 + Data.bounds[0]*ri, marker='^', lw=1, color = 'orange', zorder=3)
            plt.vlines(self.xs/2 + ri,                 # Data Extraction Range
                       self.ys/2 + self.lower_bound_west[str(self.light_days[i])]*ri,
                       self.ys/2 + self.upper_bound_west[str(self.light_days[i])]*ri, 
                       ls = '--', lw=2, color='magenta')

            # Eastern Markers
            Data = ShadowData(ri=-ri,image_file=self.image_file, loc=self.loc, 
                                                        upper_h_r = self.upper_bound_east[str(self.light_days[i])], lower_h_r = self.lower_bound_east[str(self.light_days[i])],
                                                        xs=self.xs, ys=self.ys)
            plt.scatter(self.xs/2 - ri, self.ys/2 + Data.bounds[1]*ri, marker='v', lw=1, color = 'orange', zorder=3)
            plt.scatter(self.xs/2 - ri, self.ys/2 + Data.bounds[0]*ri, marker='^', lw=1, color = 'orange', zorder=3)
            plt.vlines(self.xs/2 - ri, 
                       self.ys/2 + self.lower_bound_east[str(self.light_days[i])]*ri,
                       self.ys/2 + self.upper_bound_east[str(self.light_days[i])]*ri, 
                       ls = '--', lw=2, color='magenta') 
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.title(self.target_name + ', ' + self.filter_name)
        fig.savefig(self.save_path + self.filter_name + '_selection.pdf', bbox_inches='tight')
        
        # Image Slices
        for i in np.arange(len(self.ris)):
        # Western Slices
            if self.westernlobe == True:
                ri = self.ris[i]
                # Pull Shadow Data at Slice 
                Data = ShadowData(ri=ri,image_file=self.image_file, loc=self.loc, 
                                upper_h_r = self.upper_bound_west[str(self.light_days[i])], lower_h_r = self.lower_bound_west[str(self.light_days[i])],
                                xs=self.xs, ys=self.ys)
            # Slice Cutout
                fig = plt.figure(figsize=(7, 5))
                ax1 = fig.add_subplot(121) # row col pos
                ax1.imshow(normalized_selection[:, int(self.xs/2+ri-self.boxwidth/2) : int(self.xs/2+ri+self.boxwidth/2)],
                        origin='lower', cmap="gray", vmin=0, vmax=2)  
                ax1.scatter(self.boxwidth/2, self.ys/2 + Data.bounds[1]*ri, marker='v', lw=2, color = 'orange', zorder=3, label='Detected Edge')
                ax1.scatter(self.boxwidth/2, self.ys/2 + Data.bounds[0]*ri, marker='^', lw=2, color = 'orange', zorder=3)
                ax1.scatter(self.boxwidth/2, self.ys/2 + self.upper_bound_west[str(self.light_days[i])]*ri, marker='v',
                            lw=1, color = 'magenta', zorder=3, label='Data Edge')
                ax1.scatter(self.boxwidth/2, self.ys/2 + self.lower_bound_west[str(self.light_days[i])]*ri, marker='^',
                            lw=1, color = 'magenta', zorder=3)
                ax1.set_ylabel('Slice @ ' + str(np.round(self.light_days[i], 1)) + str(' Light Travel Days'))   
                ax1.grid(False)
                ax1.set_xlabel('pixel')
                ax1.legend(bbox_to_anchor=(3.5, 1), fontsize=6)
            # Slice Data 2D
                ax2 = fig.add_subplot(122) # row col pos
                ax2.plot(Data.full_slice, Data.full_h_r, color='black', ls='--', label='Full Slice')
                ax2.plot(Data.slice, Data.h_r, color='magenta', marker = '.', ls='--', label='Truncated Shadow')
                ax2.scatter(Data.full_slice[Data.bounds_slice[1]], Data.bounds[1], marker='v', lw=2, color = 'orange', zorder=3)
                ax2.scatter(Data.full_slice[Data.bounds_slice[0]], Data.bounds[0], marker='^', lw=2, color = 'orange', zorder=3)                          
                ax2.set_title('Slice Data')
                ax2.set_ylabel('Scale Height H/R')
                ax2.set_xlabel('Intensity')
                ax2.legend()
                fig.savefig(self.save_path + 'slice_selections/slice_' + str(int(self.light_days[i])) + '.pdf', bbox_inches='tight')
            # Eastern Slices
            if self.easternlobe == True:
                ri = -self.ris[i]
                # Pull Shadow Data at Slice
                Data = ShadowData(ri=ri,image_file=self.image_file, loc=self.loc, 
                                upper_h_r = self.upper_bound_east[str(self.light_days[i])], lower_h_r = self.lower_bound_east[str(self.light_days[i])],
                                xs=self.xs, ys=self.ys)
            # Slice Cutout
                fig = plt.figure(figsize=(7, 5))
                ax1 = fig.add_subplot(121) # row col pos
                ax1.imshow(normalized_selection[:, int(self.xs/2+ri-self.boxwidth/2) : int(self.xs/2+ri+self.boxwidth/2)],
                        origin='lower', cmap="gray", vmin=0, vmax=2)  
                ax1.scatter(self.boxwidth/2, self.ys/2 - Data.bounds[1]*ri, marker='v',
                            lw=2, color = 'orange', zorder=3, label='Detected Edge')
                ax1.scatter(self.boxwidth/2, self.ys/2 - Data.bounds[0]*ri, marker='^',
                            lw=2, color = 'orange', zorder=3)
                ax1.scatter(self.boxwidth/2, self.ys/2 - self.upper_bound_east[str(self.light_days[i])]*ri, marker='v',
                            lw=1, color = 'magenta', zorder=3, label='Data Edge')
                ax1.scatter(self.boxwidth/2, self.ys/2 - self.lower_bound_east[str(self.light_days[i])]*ri, marker='^',
                            lw=1, color = 'magenta', zorder=3)
                ax1.set_ylabel('Slice @ ' + str(-np.round(self.light_days[i], 1)) + str(' Light Travel Days'))   
                ax1.grid(False)
                ax1.set_xlabel('pixel')
                ax1.legend(bbox_to_anchor=(3.5, 1), fontsize=6)
            # Slice Data 2D
                ax2 = fig.add_subplot(122) # row col pos
                ax2.plot(Data.full_slice, Data.full_h_r, color='black', ls='--', label='Full Slice')
                ax2.plot(Data.slice, Data.h_r, color='magenta', marker = '.', ls='--', label='Truncated Shadow')
                ax2.scatter(Data.full_slice[Data.bounds_slice[1]], Data.bounds[1], marker='v', lw=2, color = 'orange', zorder=3)
                ax2.scatter(Data.full_slice[Data.bounds_slice[0]], Data.bounds[0], marker='^', lw=2, color = 'orange', zorder=3)
                ax2.set_title('Slice Data')
                ax2.set_ylabel('Scale Height H/R')
                ax2.set_xlabel('Intensity')
                ax2.legend()
                fig.savefig(self.save_path + '/slice_selections/slice_' + str(-int(self.light_days[i])) + '.pdf', bbox_inches='tight')
                
    def MCMCRunner(self, upper_h_r = 0.3, lower_h_r = -0.5, save_chainplot=False):
        '''
        Run MCMC on lobes of Shadow for given light days
        Save Best Fit Parameters to a Dictionary
        '''
        lobes = []
        ndim = 4
        if self.pixsize >= 0.07:
            ShadowData = ShadowDataHST
        elif self.pixsize <= 0.07:
            ShadowData = ShadowDataJWST
        
        if self.westernlobe == True:
            lobes.append('west')
            print('Fitting Western Lobe...')
            for i in np.arange(len(self.ris)):
                ri = self.ris[i]
                lday = int(self.light_days[i])
                print(str(ri) + ' px, ' + str(int(lday)) + ' days')

                os.makedirs(self.save_path + 'west/', exist_ok=True)
                outname = self.save_path+ 'west/'+'samples_'+str(lday)+'.pkl'
                # Pull Shadow Data at Slice
                Data = ShadowData(ri=ri,image_file=self.image_file, loc=self.loc, 
                                upper_h_r = self.upper_bound_west[str(self.light_days[i])], lower_h_r = self.lower_bound_west[str(self.light_days[i])],
                                xs=self.xs, ys=self.ys)

                # MCMC Likelihood of parameters to fit Data Slice
                BestFit = Retrieval(Data, self.disk_params)
                samples, sample_chain = BestFit.run() 

                pickle_file = open(outname,'wb')
                pickle.dump({'samples':samples,'data':Data,'chains':sample_chain}, pickle_file)

                if save_chainplot == True:
                    self.chainplotfigure(sample_chain=sample_chain, lobe='west/', ri=ri, lday=lday)
                pickle_file.close()
                
            print('Done with Western Lobe!')
            print()

        if self.easternlobe == True:
            lobes.append('east')
            print('Fitting Eastern Lobe...')
            for i in np.arange(len(self.ris)):
                ri = -self.ris[i]
                lday = int(-self.light_days[i])
                print(str(ri) + ' px, ' + str(lday) + ' days')

                os.makedirs(self.save_path + 'east/', exist_ok=True)
                outname = self.save_path+ 'east/'+'samples_'+str(lday)+'.pkl'
                # Pull Shadow Data at Slice
                Data = ShadowData(ri=ri,image_file=self.image_file, loc=self.loc, 
                                upper_h_r = self.upper_bound_east[str(self.light_days[i])], lower_h_r = self.lower_bound_east[str(self.light_days[i])],
                                xs=self.xs, ys=self.ys)

                # MCMC Likelihood of parameters to fit Data Slice
                BestFit = Retrieval(Data, self.disk_params)
                samples, sample_chain = BestFit.run() 

                pickle_file = open(outname,'wb')
                pickle.dump({'samples':samples,'data':Data,'chains':sample_chain}, pickle_file)

                # Chain Plot
                if save_chainplot == True:
                    self.chainplotfigure(sample_chain=sample_chain, lobe='east/', ri=ri, lday=lday)
                pickle_file.close()
            print('Done with Eastern Lobe!')
            print()

        self.savebestfit(lobes)
        print('Figures Saved!')

    def chainplotfigure(self, sample_chain, lobe, ri, lday):
        # Chain Plot
        fig, axes = plt.subplots(len(self.labels), 1, figsize=(15, 12), sharex=True)
        for i in range(len(self.labels)):
            ax = axes[i]
            ax.plot(sample_chain[:, : , i], color='black', alpha=0.3)
            ax.set_ylabel(self.labels[i])
        fig.suptitle(self.target_name + ', ' + self.filter_name + ', Light Travel Day: ' + str(lday), y=0.91)
        fig.savefig(self.save_path + lobe + 'chainplot_' + str(lday) + '.pdf', bbox_inches='tight')
        print('Chain Plot Saved!')

    def savebestfit(self, lobes):
        print('Saving Figures...')
        for lobe in lobes:
            dict = {'PA':[],      'PA_16':[],      'PA_84':[], 
                    'HR0':[],     'HR0_16':[],     'HR0_84':[], 
                    'BG':[],      'BG_16':[],      'BG_84':[], 
                    'BGSLOPE':[], 'BGSLOPE_16':[], 'BGSLOPE_84':[]}
            offsets = []          # PA offsets
            hr0s = []             # Scale Height of Shadow
            bgs = []              # Background Intensity
            bg_slope = []         # Background Slope

            for r in np.arange(len(self.ris)):
                if lobe == 'west':
                    ri = self.ris[r]
                    lday = int(self.light_days[r])
                    dict_days = {'light_days': self.light_days}
                elif lobe == 'east':
                    ri = -self.ris[r]
                    lday = int(-self.light_days[r])
                    dict_days = {'light_days': -self.light_days}
                pickle_name = self.save_path+lobe+'/samples_'+str(lday)+'.pkl'
                pickle_file = open(pickle_name,'rb')
                pickle_data = pickle.load(pickle_file)
                samples = pickle_data['samples']
                data = pickle_data['data']
                
                # Corner Plot     
                fig = corner.corner(samples, labels=['PA offset [H/R]', r'$h_R(R_{outer})$', r'$A_{BG}$', r'$B_{BG}$'], 
                                    label_kwargs={'fontsize':'x-large'})
                fig.suptitle(self.target_name + ', ' + self.filter_name + '\n' + str(lday) + ' Light Days from Host', fontsize=22, x=0.75, y=0.91, ha='right')
                fig.savefig(self.save_path+lobe+'/corner_'+str(lday)+'.png')
                
                # Extracting Best Fit Values from Each pickles data to 1 std
                bestfit = {} 
                for i in range(len(self.labels)):
                    mcmc = np.percentile(samples[:, i], [16, 50, 84]) # 1 std of mean
                    q = np.diff(mcmc)
                    
                    if self.labels[i] == 'offset':
                        offsets.append((mcmc[1], q[0], q[1]))
                        dict['PA'].append(mcmc[1])                # PA in radians
                        dict['PA_16'].append(180*q[0]/np.pi)      # PA error in degrees
                        dict['PA_84'].append(180*q[1]/np.pi)      # "
                    if self.labels[i] == 'hr0':
                        hr0s.append((mcmc[1], q[0], q[1]))
                        dict['HR0'].append(mcmc[1])
                        dict['HR0_16'].append(q[0])
                        dict['HR0_84'].append(q[1])
                    if self.labels[i] == 'bg':
                        bgs.append((mcmc[1], q[0], q[1]))
                        dict['BG'].append(mcmc[1])
                        dict['BG_16'].append(q[0])
                        dict['BG_84'].append(q[1])
                    if self.labels[i] == 'bg_slope':
                        bg_slope.append((mcmc[1], q[0], q[1]))
                        dict['BGSLOPE'].append(mcmc[1])
                        dict['BGSLOPE_16'].append(q[0])
                        dict['BGSLOPE_84'].append(q[1])

                    bestfit[self.labels[i]] = mcmc[1]

                # Model Shadow Using Best Fit Parameters
                fig = plt.figure()
                ax = fig.add_subplot(111)
                            
                inds = np.random.randint(len(samples), size=100)
                for ii in inds:
                    pars = {}
                    for jj in range(len(self.labels)):
                        pars[self.labels[jj]] = samples[ii,jj]
                        
                    tau = shadow(data.h_r,offset=pars['offset'], hr0=pars['hr0'])
                    bg = pars['bg']+pars['bg_slope']*data.h_r
                    model = (data.cont-bg)*np.exp(-tau)+bg

                    plt.plot(data.h_r, model, color=cp[0], alpha=0.1)

                ax.plot(data.h_r,data.slice,marker='o',color='black')
                
                tau = shadow(data.h_r,offset=bestfit['offset'], hr0=bestfit['hr0'])
                bg = bestfit['bg']+bestfit['bg_slope']*data.h_r
                model = (data.cont-bg)*np.exp(-tau)+bg

                #ax.set_title('Pixel: ' + str(ri) + ', LDay: ' + str(lday))
                ax.set_xlabel('H/R', fontsize=15)
                ax.set_ylabel('Intensity [ADU]', fontsize=15)
                
                # Vertical Line at Projected PA vs Modeled PA
                ylim = ax.get_ylim()
                ax.plot([0,0],ylim,linestyle='--',lw=2)
                ax.plot([-bestfit['offset'],-bestfit['offset']],ylim,linestyle='--',lw=2)
                ax.arrow(-bestfit['offset'],ylim[0]+ylim[1]/2.0,bestfit['offset'],0,width=0.01,length_includes_head=True)
                pickle_file.close()
                ax.tick_params(axis='both', labelsize=13)

                fig.savefig(self.save_path+lobe+'/fit_quality_'+str(lday)+'.pdf', bbox_inches='tight')
            print('Figures Saved!')
            ascii.write(dict_days | dict, self.save_path + self.filter_name + '_dict_' +lobe+ '.dat', overwrite=True)
            print('Dictionary Saved as ' + self.save_path + self.filter_name + '_dict_' +lobe+ '.dat')
