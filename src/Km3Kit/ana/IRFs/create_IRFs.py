# %% [markdown]
# Only generate aeff.fits, psf.fits, edisp.fits, bkg_nu.fits, and bkg_mu.fits

import numpy as np
import km3io
from astropy.io import fits
import astropy.units as u
from gammapy.irf import EnergyDispersion2D
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import pandas as pd
import uproot
from collections import defaultdict
from irf_utils import calc_theta, aeff_2D, psf_3D, edisp_3D
import flux as km3_flux

# Input files
filename_nu = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.1.km3_numuCC.ALL.dst.bdt.root'
filename_nubar = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.1.km3_anumuCC.ALL.dst.bdt.root'
filename_mu10 = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.2.mupage_10T.sirene_mupage.ALL.bdt.root'  
filename_mu50 = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.2.mupage_50T.sirene_mupage.ALL.bdt.root'

# function 1: Read 
# 
# Read neutrino files with km3io
f_nu_km3io = km3io.OfflineReader(filename_nu)
f_nubar_km3io = km3io.OfflineReader(filename_nubar)

# Read neutrino files with uproot
f_nu_uproot = uproot.open(filename_nu)
f_nubar_uproot = uproot.open(filename_nubar)

data_uproot = dict()
for l,f in zip(['nu', 'nubar'], [f_nu_uproot, f_nubar_uproot]):
    E = f['E;1']
    T = f['T;1']

    data_uproot[l] = dict()
    data_uproot[l]['E'] = E['Evt/trks/trks.E'].array().to_numpy()[:,0]
    data_uproot[l]['dir_z'] = E['Evt/trks/trks.dir.z'].array().to_numpy()[:,0]
    data_uproot[l]['energy_mc'] = E['Evt/mc_trks/mc_trks.E'].array().to_numpy()[:,0]
    data_uproot[l]['dir_x_mc'] = E['Evt/mc_trks/mc_trks.dir.x'].array().to_numpy()[:,0]
    data_uproot[l]['dir_y_mc'] = E['Evt/mc_trks/mc_trks.dir.y'].array().to_numpy()[:,0]
    data_uproot[l]['dir_z_mc'] = E['Evt/mc_trks/mc_trks.dir.z'].array().to_numpy()[:,0]
    data_uproot[l]['weight_w2'] = E['Evt/w'].array().to_numpy()[:,1]
    bdt = T['bdt'].array().to_numpy()
    data_uproot[l]['bdt0'] = bdt[:,0]
    data_uproot[l]['bdt1'] = bdt[:,1]

df_nu = pd.DataFrame(data_uproot['nu'])
df_nubar = pd.DataFrame(data_uproot['nubar'])

# Atmospheric muons
files_atm_mu = [filename_mu10, filename_mu50]
live_times_mu = []
for fname in files_atm_mu:
    f = km3io.OfflineReader(fname)
    t = f.header.livetime.numberOfSeconds
    live_times_mu.append(t)

data_mu = defaultdict(list)
for i,fname in enumerate(files_atm_mu):
    f_mu = uproot.open(fname)
    E = f_mu['E;1']
    T = f_mu['T;1']

    data_mu['E'].append(E['Evt/trks/trks.E'].array().to_numpy()[:,0])
    data_mu['dir_x'].append(E['Evt/trks/trks.dir.x'].array().to_numpy()[:,0])
    data_mu['dir_y'].append(E['Evt/trks/trks.dir.y'].array().to_numpy()[:,0])
    data_mu['dir_z'].append(E['Evt/trks/trks.dir.z'].array().to_numpy()[:,0])
    data_mu['energy_mc'].append(E['Evt/mc_trks/mc_trks.E'].array().to_numpy()[:,0])
    data_mu['dir_x_mc'].append(E['Evt/mc_trks/mc_trks.dir.x'].array().to_numpy()[:,0])
    data_mu['dir_y_mc'].append(E['Evt/mc_trks/mc_trks.dir.y'].array().to_numpy()[:,0])
    data_mu['dir_z_mc'].append(E['Evt/mc_trks/mc_trks.dir.z'].array().to_numpy()[:,0])
    bdt = T['bdt'].array().to_numpy()
    data_mu['bdt0'].append(bdt[:,0])
    data_mu['bdt1'].append(bdt[:,1])

    w = np.full(len(data_mu['E'][-1]), 1. / live_times_mu[i])
    data_mu['weight'].append(w)

for k in data_mu:
    data_mu[k] = np.concatenate(data_mu[k])
df_mu = pd.DataFrame(data_mu)

# function 2: Selection cuts

def get_q_mask(bdt0, bdt1, dir_z):
    mask_down = bdt0 >= 11
    clear_signal = (bdt0 == 12)
    loose_up = (np.arccos(dir_z)*180/np.pi < 80) & (bdt1 > 0.)
    strong_horizontal = (np.arccos(dir_z)*180/np.pi > 80) & (bdt1 > 0.7)
    return mask_down & (clear_signal | loose_up | strong_horizontal)


q_mask_nu = get_q_mask(df_nu.bdt0, df_nu.bdt1, df_nu.dir_z)
df_nu_q = df_nu[q_mask_nu].copy()

q_mask_nubar = get_q_mask(df_nubar.bdt0, df_nubar.bdt1, df_nubar.dir_z)
df_nubar_q = df_nubar[q_mask_nubar].copy()

df_nu_all_q = pd.concat([df_nu_q, df_nubar_q], ignore_index=True)


# function 3:  apply weight
weights = dict()
weight_factor = -2.5
weights['nu'] = (df_nu_q.energy_mc**(weight_factor - f_nu_km3io.header.spectrum.alpha)).to_numpy()
weights['nu'] *= len(df_nu_q) / weights['nu'].sum()

weights['nubar'] = (df_nubar_q.energy_mc**(weight_factor - f_nubar_km3io.header.spectrum.alpha)).to_numpy()
weights['nubar'] *= len(df_nubar_q) / weights['nubar'].sum()

weights_all = np.concatenate([weights['nu'], weights['nubar']])

q_mask_mu = get_q_mask(df_mu.bdt0, df_mu.bdt1, df_mu.dir_z)
df_mu_q = df_mu[q_mask_mu].copy()

# Binning
cos_bins_fine = np.linspace(1, -1, 13)
t_bins_fine = np.arccos(cos_bins_fine)
e_bins_fine = np.logspace(2, 8, 49)

cos_bins_coarse = np.linspace(1, -1, 7)
t_bins_coarse = np.arccos(cos_bins_coarse)
e_bins_coarse = np.logspace(2, 8, 25)

migra_bins = np.logspace(-5, 2, 57)
rad_bins = np.concatenate((np.linspace(0, 1, 21),
                           np.linspace(1, 5, 41)[1:],
                           np.linspace(5, 30, 51)[1:],
                           [180.]))

e_binc_fine = np.sqrt(e_bins_fine[:-1] * e_bins_fine[1:])
e_binc_coarse = np.sqrt(e_bins_coarse[:-1] * e_bins_coarse[1:])

# Extended E bins
e_bins_fine_ext = np.logspace(0, 8, 65)

# Compute IRFs
aeff = aeff_2D(e_bins_fine, t_bins_fine, df_nu_all_q, gamma=(-weight_factor),
               nevents = f_nu_km3io.header.genvol.numberOfEvents + f_nubar_km3io.header.genvol.numberOfEvents) * 2

psf = psf_3D(e_bins_coarse, rad_bins, t_bins_coarse, df_nu_all_q, weights_all)
edisp = edisp_3D(e_bins_coarse, migra_bins, t_bins_coarse, df_nu_all_q, weights_all)

sizes_rad_bins = np.diff(rad_bins**2)
psf_weighted = psf / (sizes_rad_bins[:,None,None] * (np.pi/180)**2 * np.pi)
norm_psf = psf.sum(axis=0, keepdims=True)
psf_normed = np.nan_to_num(psf_weighted / norm_psf)

sizes_migra_bins = np.diff(migra_bins)
edisp /= sizes_migra_bins[:,np.newaxis]
m_normed = edisp * sizes_migra_bins[:,np.newaxis]
norm_edisp = m_normed.sum(axis=0, keepdims=True)
edisp_normed = np.nan_to_num(edisp / norm_edisp)

# Smooth EDISP
edisp_smoothed = np.zeros_like(edisp)
for i in range(edisp.shape[-1]):
    for j in range(edisp.shape[0]):
        kernel_size = 2 - 0.25*max(0, np.log10(edisp[j,:,i].sum()))
        edisp_smoothed[j,:,i] = gaussian_filter1d(edisp[j,:,i]*sizes_migra_bins, kernel_size, axis=0, mode='nearest')
edisp_smoothed /= sizes_migra_bins[:,None]
m_normed = edisp_smoothed * sizes_migra_bins[:,np.newaxis]
norm_edisp_sm = m_normed.sum(axis=1, keepdims=True)
edisp_smoothed_normed = np.nan_to_num(edisp_smoothed / norm_edisp_sm)

# Smooth PSF
s1 = gaussian_filter1d(psf_weighted, 0.5, axis=0, mode='nearest')
s2 = gaussian_filter1d(psf_weighted, 2,   axis=0, mode='nearest')
s3 = gaussian_filter1d(psf_weighted, 4,   axis=0, mode='nearest')
s4 = gaussian_filter1d(psf_weighted, 6,   axis=0, mode='constant')
psf_smoothed = np.concatenate((s1[:10], s2[10:20], s3[20:60], s4[60:-1], [psf_weighted[-1]]), axis=0)
psf_smoothed[10:-1] = gaussian_filter1d(psf_smoothed[10:-1], 1, axis=0, mode='nearest')
norm_psf_sm = (psf_smoothed * sizes_rad_bins[:,None,None] * (np.pi/180)**2 * np.pi).sum(axis=0, keepdims=True)
psf_smoothed_normed = np.nan_to_num(psf_smoothed / norm_psf_sm)

# Atmospheric muon bkg
df_mu_q['theta'] = calc_theta(df_mu_q, mc=False)
atm_mu_bkg = np.histogram2d(df_mu_q.E, df_mu_q.theta,
                            bins=(e_bins_fine_ext, t_bins_fine), weights=df_mu_q.weight)[0]
atm_mu_bkg *= 2
atm_mu_rate = atm_mu_bkg / (np.pi * 4 / len(t_bins_fine) * np.diff(e_bins_fine_ext)[:,None])

# Smooth muon background at horizon
mu_horizon_raw = atm_mu_rate[:,5]
mu_horizon_smoothed = gaussian_filter(mu_horizon_raw, [2.0], mode='nearest')
atm_mu_rate_out = np.zeros_like(atm_mu_rate)
atm_mu_rate_out[:,5] = mu_horizon_smoothed




# Knee corrections
honda_knee_e, honda_knee_f = np.loadtxt('honda_knee_correction_gaisser_H3a.dat', unpack=True)
enberg_knee_e, enberg_knee_f = np.loadtxt('enberg_knee_correction_gaisser_H3a.dat', unpack=True)
honda_knee_correction = lambda e:np.interp(np.log10(e), np.log10(honda_knee_e), honda_knee_f)
enberg_knee_correction = lambda e:np.interp(np.log10(e), np.log10(enberg_knee_e), enberg_knee_f)

e_binc_fine_ext = np.sqrt(e_bins_fine_ext[:-1]*e_bins_fine_ext[1:])
t_binc_fine = np.arccos((cos_bins_fine[:-1]+cos_bins_fine[1:])*0.5)

# Compute separate aeff for nu and nubar again (needed for bkg)
aeff_nu = aeff_2D(e_bins_fine, t_bins_fine, df_nu_q, gamma=(-weight_factor),
                  nevents=f_nu_km3io.header.genvol.numberOfEvents)*2
aeff_nubar = aeff_2D(e_bins_fine, t_bins_fine, df_nubar_q, gamma=(-weight_factor),
                     nevents=f_nubar_km3io.header.genvol.numberOfEvents)*2

atm_conv_flux = dict()
atm_prompt_flux = dict()
atm_conv_flux['nu'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))
atm_conv_flux['nubar'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))
atm_prompt_flux['nu'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))
atm_prompt_flux['nubar'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))

for l,pt in zip(['nu', 'nubar'], [14, -14]):
    for i in range(len(t_binc_fine)):
        conv_flux = np.array([km3_flux.atmospheric_conventional(mc_type=pt, zenith=t_binc_fine[i], energy=e) for e in e_binc_fine])
        prompt_flux = np.array([km3_flux.atmospheric_prompt(mc_type=pt, zenith=t_binc_fine[i], energy=e) for e in e_binc_fine])
        conv_flux *= honda_knee_correction(e_binc_fine)
        prompt_flux *= enberg_knee_correction(e_binc_fine)
        atm_conv_flux[l][:,i] = conv_flux
        atm_prompt_flux[l][:,i] = prompt_flux

atm_conv_rate_etrue = atm_conv_flux['nu']*aeff_nu + atm_conv_flux['nubar']*aeff_nubar
atm_prompt_rate_etrue = atm_prompt_flux['nu']*aeff_nu + atm_prompt_flux['nubar']*aeff_nubar

# Apply energy dispersion
edisp_cls = EnergyDispersion2D(e_bins_coarse[:-1]*u.GeV, e_bins_coarse[1:]*u.GeV,
                               migra_bins[:-1], migra_bins[1:],
                               t_bins_coarse[:-1]*u.deg, t_bins_coarse[1:]*u.deg,
                               edisp_smoothed_normed.T * sizes_migra_bins[:,None])

atm_conv_rate = np.zeros((len(e_bins_fine_ext[:-1]), len(t_binc_fine)))
atm_prompt_rate = np.zeros((len(e_bins_fine_ext[:-1]), len(t_binc_fine)))
for i in range(len(t_binc_fine)):
    edisp_matrix = edisp_cls.to_energy_dispersion(t_binc_fine[i]*u.deg,
                                                  e_true = e_bins_fine*u.GeV,
                                                  e_reco = e_bins_fine_ext*u.GeV)
    conv_etrue = atm_conv_rate_etrue[:,i]*np.diff(e_bins_fine)
    prompt_etrue = atm_prompt_rate_etrue[:,i]*np.diff(e_bins_fine)
    conv = np.dot(conv_etrue, edisp_matrix.pdf_matrix)
    prompt = np.dot(prompt_etrue, edisp_matrix.pdf_matrix)
    conv /= np.diff(e_bins_fine_ext)
    prompt /= np.diff(e_bins_fine_ext)
    atm_conv_rate[:,i] = conv
    atm_prompt_rate[:,i] = prompt

# Write IRFs to FITS
hdu = fits.PrimaryHDU()

# AEFF
col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(e_bins_fine)), unit='GeV', array=[e_bins_fine[:-1]])
col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(e_bins_fine)), unit='GeV', array=[e_bins_fine[1:]])
col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(t_bins_fine)), unit='rad', array=[t_bins_fine[:-1]])
col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(t_bins_fine)), unit='rad', array=[t_bins_fine[1:]])
col5 = fits.Column(name='EFFAREA', format='{}D'.format(len(e_bins_fine)*len(t_bins_fine)),
                   dim='({},{})'.format(len(e_bins_fine), len(t_bins_fine)), unit='m2', array=[aeff.T])
cols = fits.ColDefs([col1, col2, col3, col4, col5])
hdu2 = fits.BinTableHDU.from_columns(cols)
hdu2.header['EXTNAME'] = 'EFFECTIVE AREA'
hdu2.header['HDUCLASS'] = 'GADF'
hdu2.header['HDUCLAS1'] = 'RESPONSE'
hdu2.header['HDUCLAS2'] = 'EFF_AREA'
hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
hdu2.header['HDUCLAS4'] = 'AEFF_2D'
aeff_fits = fits.HDUList([hdu, hdu2])
aeff_fits.writeto('aeff.fits', overwrite=True)

# PSF
col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(e_bins_coarse)), unit='GeV', array=[e_bins_coarse[:-1]])
col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(e_bins_coarse)), unit='GeV', array=[e_bins_coarse[1:]])
col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(t_bins_coarse)), unit='rad', array=[t_bins_coarse[:-1]])
col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(t_bins_coarse)), unit='rad', array=[t_bins_coarse[1:]])
col5 = fits.Column(name='RAD_LO', format='{}E'.format(len(rad_bins)), unit='deg', array=[rad_bins[:-1]])
col6 = fits.Column(name='RAD_HI', format='{}E'.format(len(rad_bins)), unit='deg', array=[rad_bins[1:]])
col7 = fits.Column(name='RPSF', format='{}D'.format(len(e_bins_coarse)*len(t_bins_coarse)*len(rad_bins)),
                   dim='({},{},{})'.format(len(e_bins_coarse), len(t_bins_coarse), len(rad_bins)), unit='sr-1',
                   array=[psf_smoothed_normed])
cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7])
hdu2 = fits.BinTableHDU.from_columns(cols)
hdu2.header['EXTNAME'] = 'PSF_2D_TABLE'
hdu2.header['HDUCLASS'] = 'GADF'
hdu2.header['HDUCLAS1'] = 'RESPONSE'
hdu2.header['HDUCLAS2'] = 'RPSF'
hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
hdu2.header['HDUCLAS4'] = 'PSF_TABLE'
psf_fits = fits.HDUList([hdu, hdu2])
psf_fits.writeto('psf.fits', overwrite=True)

# EDISP
col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(e_bins_coarse)), unit='GeV', array=[e_bins_coarse[:-1]])
col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(e_bins_coarse)), unit='GeV', array=[e_bins_coarse[1:]])
col3 = fits.Column(name='MIGRA_LO', format='{}E'.format(len(migra_bins)), array=[migra_bins[:-1]])
col4 = fits.Column(name='MIGRA_HI', format='{}E'.format(len(migra_bins)), array=[migra_bins[1:]])
col5 = fits.Column(name='THETA_LO', format='{}E'.format(len(t_bins_coarse)), unit='rad', array=[t_bins_coarse[:-1]])
col6 = fits.Column(name='THETA_HI', format='{}E'.format(len(t_bins_coarse)), unit='rad', array=[t_bins_coarse[1:]])
col7 = fits.Column(name='MATRIX', format='{}D'.format(len(e_bins_coarse)*len(migra_bins)*len(t_bins_coarse)),
                   dim='({},{},{})'.format(len(e_bins_coarse), len(migra_bins), len(t_bins_coarse)),
                   array=[edisp_smoothed_normed * sizes_migra_bins[:,None]])
cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7])
hdu2 = fits.BinTableHDU.from_columns(cols)
hdu2.header['EXTNAME'] = 'EDISP_2D'
hdu2.header['HDUCLASS'] = 'GADF'
hdu2.header['HDUCLAS1'] = 'RESPONSE'
hdu2.header['HDUCLAS2'] = 'EDISP'
hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
hdu2.header['HDUCLAS4'] = 'EDISP_2D'
edisp_fits = fits.HDUList([hdu, hdu2])
edisp_fits.writeto('edisp.fits', overwrite=True)

# BKG NU
col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(e_bins_fine_ext)), unit='GeV', array=[e_bins_fine_ext[:-1]])
col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(e_bins_fine_ext)), unit='GeV', array=[e_bins_fine_ext[1:]])
col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(t_bins_fine)), unit='rad', array=[t_bins_fine[:-1]])
col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(t_bins_fine)), unit='rad', array=[t_bins_fine[1:]])
col5 = fits.Column(name='BKG', format='{}D'.format(len(e_bins_fine_ext)*len(t_bins_fine)),
                   dim='({},{})'.format(len(t_bins_fine), len(e_bins_fine_ext)), unit='MeV-1 sr-1 s-1',
                   array=[(atm_conv_rate + atm_prompt_rate)*1e-3])
cols = fits.ColDefs([col1, col2, col3, col4, col5])
hdu2 = fits.BinTableHDU.from_columns(cols)
hdu2.header['EXTNAME'] = 'BACKGROUND'
hdu2.header['HDUCLASS'] = 'GADF'
hdu2.header['HDUCLAS1'] = 'RESPONSE'
hdu2.header['HDUCLAS2'] = 'BKG'
hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
hdu2.header['HDUCLAS4'] = 'BKG_2D'
bkg_fits = fits.HDUList([hdu, hdu2])
bkg_fits.writeto('bkg_nu.fits', overwrite=True)

# BKG MU
t_bins_mu_out = np.insert(np.arccos(cos_bins_fine[1:] + 1./len(t_bins_fine)), 0, 0.0)
col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(e_bins_fine_ext)), unit='GeV', array=[e_bins_fine_ext[:-1]])
col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(e_bins_fine_ext)), unit='GeV', array=[e_bins_fine_ext[1:]])
col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(t_bins_fine)), unit='rad', array=[t_bins_mu_out[:-1]])
col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(t_bins_fine)), unit='rad', array=[t_bins_mu_out[1:]])
col5 = fits.Column(name='BKG', format='{}D'.format(len(e_bins_fine_ext)*len(t_bins_fine)),
                   dim='({},{})'.format(len(t_bins_fine), len(e_bins_fine_ext)), unit='MeV-1 sr-1 s-1',
                   array=[atm_mu_rate_out*1e-3])
cols = fits.ColDefs([col1, col2, col3, col4, col5])
hdu2 = fits.BinTableHDU.from_columns(cols)
hdu2.header['EXTNAME'] = 'BACKGROUND'
hdu2.header['HDUCLASS'] = 'GADF'
hdu2.header['HDUCLAS1'] = 'RESPONSE'
hdu2.header['HDUCLAS2'] = 'BKG'
hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
hdu2.header['HDUCLAS4'] = 'BKG_2D'
bkg_fits_mu = fits.HDUList([hdu, hdu2])
bkg_fits_mu.writeto('bkg_mu.fits', overwrite=True)