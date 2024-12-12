import numpy as np
import km3io
from astropy.io import fits
import astropy.units as u
from gammapy.irf import EnergyDispersion2D
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
import pandas as pd
import uproot
from collections import defaultdict

import numba as nb
from numba import jit, prange 
from km3pipe.math import azimuth, zenith
import astropy.coordinates as ac
from astropy.time import Time

from Km3Kit.ana.flux import flux as km3_flux


def calc_theta(table, mc=True):
    if not mc:
        dir_x = table['dir_x'].to_numpy()
        dir_y = table['dir_y'].to_numpy()
        dir_z = table['dir_z'].to_numpy()
    else:
        dir_x = table['dir_x_mc'].to_numpy()
        dir_y = table['dir_y_mc'].to_numpy()
        dir_z = table['dir_z_mc'].to_numpy()

    nu_directions = np.vstack([dir_x, dir_y, dir_z]).T
    theta = zenith(nu_directions)  # zenith angles in rad [0:pi]
    return theta


def edisp_3D(e_bins, m_bins, t_bins, dataset, weights=1):
    if 'theta_mc' not in dataset.keys():
        dataset['theta_mc'] = calc_theta(dataset, mc=True)
    if 'migra' not in dataset.keys():
        dataset['migra'] = dataset.E / dataset.energy_mc

    theta_bins = pd.cut(dataset.theta_mc, t_bins, labels=False).to_numpy()
    energy_bins = pd.cut(dataset.energy_mc, e_bins, labels=False).to_numpy()
    migra_bins = pd.cut(dataset.migra, m_bins, labels=False).to_numpy()

    edisp = fill_edisp_3D(e_bins, m_bins, t_bins, energy_bins, migra_bins, theta_bins, weights)
    return edisp


@jit(nopython=True, fastmath=False, parallel=True)
def fill_edisp_3D(e_bins, m_bins, t_bins, energy_bins, migra_bins, theta_bins, weights):
    edisp = np.zeros((len(t_bins)-1, len(m_bins)-1, len(e_bins)-1))
    for i in prange(len(t_bins)-1):
        for j in range(len(m_bins)-1):
            for k in range(len(e_bins)-1):
                mask = (energy_bins == k) & (migra_bins == j) & (theta_bins == i)
                edisp[i, j, k] = np.sum(mask * weights)
    return edisp


def psf_3D(e_bins, r_bins, t_bins, dataset, weights=1):
    if 'theta_mc' not in dataset.keys():
        dataset['theta_mc'] = calc_theta(dataset, mc=True)

    scalar_prod = dataset.dir_x * dataset.dir_x_mc + dataset.dir_y * dataset.dir_y_mc + dataset.dir_z * dataset.dir_z_mc
    scalar_prod[scalar_prod > 1.0] = 1.0
    rad = np.arccos(scalar_prod) * 180 / np.pi  # in degrees
    dataset['rad'] = rad

    theta_bins = pd.cut(dataset.theta_mc, t_bins, labels=False).to_numpy()
    energy_bins = pd.cut(dataset.energy_mc, e_bins, labels=False).to_numpy()
    rad_bins = pd.cut(rad, r_bins, labels=False).to_numpy()

    psf = fill_psf_3D(e_bins, r_bins, t_bins, energy_bins, rad_bins, theta_bins, weights)
    return psf


@jit(nopython=True, fastmath=False, parallel=True)
def fill_psf_3D(e_bins, r_bins, t_bins, energy_bins, rad_bins, theta_bins, weights):
    psf = np.zeros((len(r_bins)-1, len(t_bins)-1, len(e_bins)-1))
    for j in prange(len(r_bins)-1):
        for i in range(len(t_bins)-1):
            for k in range(len(e_bins)-1):
                mask = (energy_bins == k) & (rad_bins == j) & (theta_bins == i)
                psf[j, i, k] = np.sum(mask * weights)
    return psf


def aeff_2D(e_bins, t_bins, dataset, gamma=1.4, nevents=2e7):
    if 'theta_mc' not in dataset.keys():
        dataset['theta_mc'] = calc_theta(dataset, mc=True)

    theta_bins = pd.cut(dataset.theta_mc, t_bins, labels=False).to_numpy()
    energy_bins = pd.cut(dataset.energy_mc, e_bins, labels=False).to_numpy()

    w2 = dataset.weight_w2.to_numpy()
    E = dataset.energy_mc.to_numpy()
    aeff = fill_aeff_2D(e_bins, t_bins, energy_bins, theta_bins, w2, E, gamma, nevents)
    return aeff


@jit(nopython=True, fastmath=False, parallel=True)
def fill_aeff_2D(e_bins, t_bins, energy_bins, theta_bins, w2, E, gamma, nevents):
    T = 365 * 24 * 3600
    aeff = np.empty((len(e_bins)-1, len(t_bins)-1))
    for k in prange(len(e_bins)-1):
        for i in range(len(t_bins)-1):
            mask = (energy_bins == k) & (theta_bins == i)
            d_omega = -(np.cos(t_bins[i+1]) - np.cos(t_bins[i]))
            d_E = (e_bins[k+1])**(1-gamma) - (e_bins[k])**(1-gamma)
            aeff[k, i] = (1-gamma) * np.sum(E[mask]**(-gamma) * w2[mask]) / (T * d_omega * d_E * nevents * 2 * np.pi)
    return aeff


class KM3NetIRFGenerator:
    def __init__(self,
                 filename_nu,
                 filename_nubar,
                 filename_mu10,
                 filename_mu50,
                 save_dir,
                 weight_factor=-2.5):
        self.filename_nu = filename_nu
        self.filename_nubar = filename_nubar
        self.filename_mu10 = filename_mu10
        self.filename_mu50 = filename_mu50
        self.save_dir = save_dir
        self.weight_factor = weight_factor

        # Will be filled by subsequent methods
        self.df_nu = None
        self.df_nubar = None
        self.df_mu = None
        self.df_nu_q = None
        self.df_nubar_q = None
        self.df_mu_q = None
        self.df_nu_all_q = None
        self.weights = None
        self.weights_all = None

        # bins (static definitions, can be modified if needed)
        self.cos_bins_fine = np.linspace(1, -1, 13)
        self.t_bins_fine = np.arccos(self.cos_bins_fine)
        self.e_bins_fine = np.logspace(2, 8, 49)

        self.cos_bins_coarse = np.linspace(1, -1, 7)
        self.t_bins_coarse = np.arccos(self.cos_bins_coarse)
        self.e_bins_coarse = np.logspace(2, 8, 25)

        self.migra_bins = np.logspace(-5, 2, 57)
        self.rad_bins = np.concatenate((np.linspace(0, 1, 21),
                                        np.linspace(1, 5, 41)[1:],
                                        np.linspace(5, 30, 51)[1:],
                                        [180.]))

        self.e_binc_fine = np.sqrt(self.e_bins_fine[:-1] * self.e_bins_fine[1:])
        self.e_binc_coarse = np.sqrt(self.e_bins_coarse[:-1] * self.e_bins_coarse[1:])
        self.e_bins_fine_ext = np.logspace(0, 8, 65)

        # will hold IRF arrays
        self.aeff = None
        self.psf_smoothed_normed = None
        self.edisp_smoothed_normed = None
        self.atm_conv_rate = None
        self.atm_prompt_rate = None
        self.atm_mu_rate_out = None

    def read_data(self):
        # Read neutrino files with km3io
        self.f_nu_km3io = km3io.OfflineReader(self.filename_nu)
        self.f_nubar_km3io = km3io.OfflineReader(self.filename_nubar)

        # Read neutrino files with uproot
        f_nu_uproot = uproot.open(self.filename_nu)
        f_nubar_uproot = uproot.open(self.filename_nubar)

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

        self.df_nu = pd.DataFrame(data_uproot['nu'])
        self.df_nubar = pd.DataFrame(data_uproot['nubar'])

        # Atmospheric muons
        files_atm_mu = [self.filename_mu10, self.filename_mu50]
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
        self.df_mu = pd.DataFrame(data_mu)

    def apply_selection_cuts(self):
        def get_q_mask(bdt0, bdt1, dir_z):
            mask_down = bdt0 >= 11
            clear_signal = (bdt0 == 12)
            loose_up = (np.arccos(dir_z)*180/np.pi < 80) & (bdt1 > 0.)
            strong_horizontal = (np.arccos(dir_z)*180/np.pi > 80) & (bdt1 > 0.7)
            return mask_down & (clear_signal | loose_up | strong_horizontal)

        q_mask_nu = get_q_mask(self.df_nu.bdt0, self.df_nu.bdt1, self.df_nu.dir_z)
        self.df_nu_q = self.df_nu[q_mask_nu].copy()

        q_mask_nubar = get_q_mask(self.df_nubar.bdt0, self.df_nubar.bdt1, self.df_nubar.dir_z)
        self.df_nubar_q = self.df_nubar[q_mask_nubar].copy()

        self.df_nu_all_q = pd.concat([self.df_nu_q, self.df_nubar_q], ignore_index=True)

        q_mask_mu = get_q_mask(self.df_mu.bdt0, self.df_mu.bdt1, self.df_mu.dir_z)
        self.df_mu_q = self.df_mu[q_mask_mu].copy()

    def apply_weights(self):
        # Apply weights based on weight_factor
        self.weights = dict()
        alpha_nu = self.f_nu_km3io.header.spectrum.alpha
        alpha_nubar = self.f_nubar_km3io.header.spectrum.alpha

        w_nu = (self.df_nu_q.energy_mc**(self.weight_factor - alpha_nu)).to_numpy()
        w_nu *= len(self.df_nu_q) / w_nu.sum()

        w_nubar = (self.df_nubar_q.energy_mc**(self.weight_factor - alpha_nubar)).to_numpy()
        w_nubar *= len(self.df_nubar_q) / w_nubar.sum()

        self.weights['nu'] = w_nu
        self.weights['nubar'] = w_nubar
        self.weights_all = np.concatenate([w_nu, w_nubar])

    def compute_aeff(self):
        # Compute Effective Area
        nevents = self.f_nu_km3io.header.genvol.numberOfEvents + self.f_nubar_km3io.header.genvol.numberOfEvents
        self.aeff = aeff_2D(self.e_bins_fine, self.t_bins_fine, self.df_nu_all_q,
                            gamma=(-self.weight_factor), nevents=nevents)*2

    def compute_psf(self):
        # Compute PSF and smooth
        psf = psf_3D(self.e_bins_coarse, self.rad_bins, self.t_bins_coarse,
                     self.df_nu_all_q, self.weights_all)
        sizes_rad_bins = np.diff(self.rad_bins**2)
        psf_weighted = psf / (sizes_rad_bins[:,None,None] * (np.pi/180)**2 * np.pi)

        s1 = gaussian_filter1d(psf_weighted, 0.5, axis=0, mode='nearest')
        s2 = gaussian_filter1d(psf_weighted, 2,   axis=0, mode='nearest')
        s3 = gaussian_filter1d(psf_weighted, 4,   axis=0, mode='nearest')
        s4 = gaussian_filter1d(psf_weighted, 6,   axis=0, mode='constant')
        psf_smoothed = np.concatenate((s1[:10], s2[10:20], s3[20:60], s4[60:-1], [psf_weighted[-1]]), axis=0)
        psf_smoothed[10:-1] = gaussian_filter1d(psf_smoothed[10:-1], 1, axis=0, mode='nearest')
        norm_psf_sm = (psf_smoothed * sizes_rad_bins[:,None,None] * (np.pi/180)**2 * np.pi).sum(axis=0, keepdims=True)
        self.psf_smoothed_normed = np.nan_to_num(psf_smoothed / norm_psf_sm)

    def compute_edisp(self):
        # Compute EDISP and smooth
        edisp = edisp_3D(self.e_bins_coarse, self.migra_bins, self.t_bins_coarse,
                         self.df_nu_all_q, self.weights_all)

        sizes_migra_bins = np.diff(self.migra_bins)
        edisp /= sizes_migra_bins[:,np.newaxis]
        m_normed = edisp * sizes_migra_bins[:,np.newaxis]
        norm_edisp = m_normed.sum(axis=0, keepdims=True)
        edisp_normed = np.nan_to_num(edisp / norm_edisp)

        edisp_smoothed = np.zeros_like(edisp)
        for i in range(edisp.shape[-1]):
            for j in range(edisp.shape[0]):
                kernel_size = 2 - 0.25*max(0, np.log10(edisp[j,:,i].sum()))
                edisp_smoothed[j,:,i] = gaussian_filter1d(edisp[j,:,i]*sizes_migra_bins,
                                                          kernel_size, axis=0, mode='nearest')
        edisp_smoothed /= sizes_migra_bins[:,None]
        m_normed = edisp_smoothed * sizes_migra_bins[:,np.newaxis]
        norm_edisp_sm = m_normed.sum(axis=1, keepdims=True)
        self.edisp_smoothed_normed = np.nan_to_num(edisp_smoothed / norm_edisp_sm)

    def _compute_atmospheric_backgrounds(self):
        # Compute atmospheric muon background
        self.df_mu_q['theta'] = calc_theta(self.df_mu_q, mc=False)
        atm_mu_bkg = np.histogram2d(self.df_mu_q.E, self.df_mu_q.theta,
                                    bins=(self.e_bins_fine_ext, self.t_bins_fine),
                                    weights=self.df_mu_q.weight)[0]
        atm_mu_bkg *= 2
        atm_mu_rate = atm_mu_bkg / (np.pi * 4 / len(self.t_bins_fine) * np.diff(self.e_bins_fine_ext)[:,None])

        # Smooth muon background at horizon
        mu_horizon_raw = atm_mu_rate[:,5]
        mu_horizon_smoothed = gaussian_filter(mu_horizon_raw, [2.0], mode='nearest')
        atm_mu_rate_out = np.zeros_like(atm_mu_rate)
        atm_mu_rate_out[:,5] = mu_horizon_smoothed

        # Compute atmospheric neutrino background
        honda_knee_e, honda_knee_f = np.loadtxt('honda_knee_correction_gaisser_H3a.dat', unpack=True)
        enberg_knee_e, enberg_knee_f = np.loadtxt('enberg_knee_correction_gaisser_H3a.dat', unpack=True)
        honda_knee_correction = lambda e:np.interp(np.log10(e), np.log10(honda_knee_e), honda_knee_f)
        enberg_knee_correction = lambda e:np.interp(np.log10(e), np.log10(enberg_knee_e), enberg_knee_f)

        e_binc_fine_ext = np.sqrt(self.e_bins_fine_ext[:-1]*self.e_bins_fine_ext[1:])
        t_binc_fine = np.arccos((self.cos_bins_fine[:-1]+self.cos_bins_fine[1:])*0.5)

        aeff_nu = aeff_2D(self.e_bins_fine, self.t_bins_fine, self.df_nu_q,
                          gamma=(-self.weight_factor),
                          nevents=self.f_nu_km3io.header.genvol.numberOfEvents)*2
        aeff_nubar = aeff_2D(self.e_bins_fine, self.t_bins_fine, self.df_nubar_q,
                             gamma=(-self.weight_factor),
                             nevents=self.f_nubar_km3io.header.genvol.numberOfEvents)*2

        atm_conv_flux = dict()
        atm_prompt_flux = dict()
        atm_conv_flux['nu'] = np.zeros((len(self.e_binc_fine), len(self.t_bins_fine)))
        atm_conv_flux['nubar'] = np.zeros((len(self.e_binc_fine), len(self.t_bins_fine)))
        atm_prompt_flux['nu'] = np.zeros((len(self.e_binc_fine), len(self.t_bins_fine)))
        atm_prompt_flux['nubar'] = np.zeros((len(self.e_binc_fine), len(self.t_bins_fine)))

        for l,pt in zip(['nu', 'nubar'], [14, -14]):
            for i in range(len(self.t_bins_fine)):
                conv_flux = np.array([km3_flux.atmospheric_conventional(mc_type=pt, zenith=self.t_bins_fine[i], energy=e) 
                                      for e in self.e_binc_fine])
                prompt_flux = np.array([km3_flux.atmospheric_prompt(mc_type=pt, zenith=self.t_bins_fine[i], energy=e)
                                        for e in self.e_binc_fine])
                conv_flux *= honda_knee_correction(self.e_binc_fine)
                prompt_flux *= enberg_knee_correction(self.e_binc_fine)
                atm_conv_flux[l][:,i] = conv_flux
                atm_prompt_flux[l][:,i] = prompt_flux

        atm_conv_rate_etrue = atm_conv_flux['nu']*aeff_nu + atm_conv_flux['nubar']*aeff_nubar
        atm_prompt_rate_etrue = atm_prompt_flux['nu']*aeff_nu + atm_prompt_flux['nubar']*aeff_nubar

        # Apply EDISP to get atm_conv_rate, atm_prompt_rate
        edisp_cls = EnergyDispersion2D(self.e_bins_coarse[:-1]*u.GeV, self.e_bins_coarse[1:]*u.GeV,
                                       self.migra_bins[:-1], self.migra_bins[1:],
                                       self.t_bins_coarse[:-1]*u.deg, self.t_bins_coarse[1:]*u.deg,
                                       self.edisp_smoothed_normed.T * np.diff(self.migra_bins)[:,None])

        atm_conv_rate = np.zeros((len(self.e_bins_fine_ext[:-1]), len(self.t_bins_fine)))
        atm_prompt_rate = np.zeros((len(self.e_bins_fine_ext[:-1]), len(self.t_bins_fine)))
        for i in range(len(self.t_bins_fine)):
            edisp_matrix = edisp_cls.to_energy_dispersion(self.t_bins_fine[i]*u.deg,
                                                          e_true = self.e_bins_fine*u.GeV,
                                                          e_reco = self.e_bins_fine_ext*u.GeV)
            conv_etrue = atm_conv_rate_etrue[:,i]*np.diff(self.e_bins_fine)
            prompt_etrue = atm_prompt_rate_etrue[:,i]*np.diff(self.e_bins_fine)
            conv = np.dot(conv_etrue, edisp_matrix.pdf_matrix)
            prompt = np.dot(prompt_etrue, edisp_matrix.pdf_matrix)
            conv /= np.diff(self.e_bins_fine_ext)
            prompt /= np.diff(self.e_bins_fine_ext)
            atm_conv_rate[:,i] = conv
            atm_prompt_rate[:,i] = prompt

        self.atm_conv_rate = atm_conv_rate
        self.atm_prompt_rate = atm_prompt_rate
        self.atm_mu_rate_out = atm_mu_rate_out

    def write_irfs(self):
        # write AEFF
        hdu = fits.PrimaryHDU()
        col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(self.e_bins_fine)), unit='GeV', array=[self.e_bins_fine[:-1]])
        col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(self.e_bins_fine)), unit='GeV', array=[self.e_bins_fine[1:]])
        col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(self.t_bins_fine)), unit='rad', array=[self.t_bins_fine[:-1]])
        col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(self.t_bins_fine)), unit='rad', array=[self.t_bins_fine[1:]])
        col5 = fits.Column(name='EFFAREA', format='{}D'.format(len(self.e_bins_fine)*len(self.t_bins_fine)),
                           dim='({},{})'.format(len(self.e_bins_fine), len(self.t_bins_fine)), unit='m2', array=[self.aeff.T])
        cols = fits.ColDefs([col1, col2, col3, col4, col5])
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header['EXTNAME'] = 'EFFECTIVE AREA'
        hdu2.header['HDUCLASS'] = 'GADF'
        hdu2.header['HDUCLAS1'] = 'RESPONSE'
        hdu2.header['HDUCLAS2'] = 'EFF_AREA'
        hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu2.header['HDUCLAS4'] = 'AEFF_2D'
        aeff_fits = fits.HDUList([hdu, hdu2])
        aeff_fits.writeto(self.save_dir+'aeff.fits', overwrite=True)

        # PSF
        col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(self.e_bins_coarse)), unit='GeV', array=[self.e_bins_coarse[:-1]])
        col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(self.e_bins_coarse)), unit='GeV', array=[self.e_bins_coarse[1:]])
        col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(self.t_bins_coarse)), unit='rad', array=[self.t_bins_coarse[:-1]])
        col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(self.t_bins_coarse)), unit='rad', array=[self.t_bins_coarse[1:]])
        col5 = fits.Column(name='RAD_LO', format='{}E'.format(len(self.rad_bins)), unit='deg', array=[self.rad_bins[:-1]])
        col6 = fits.Column(name='RAD_HI', format='{}E'.format(len(self.rad_bins)), unit='deg', array=[self.rad_bins[1:]])
        col7 = fits.Column(name='RPSF', format='{}D'.format(len(self.e_bins_coarse)*len(self.t_bins_coarse)*len(self.rad_bins)),
                           dim='({},{},{})'.format(len(self.e_bins_coarse), len(self.t_bins_coarse), len(self.rad_bins)),
                           unit='sr-1', array=[self.psf_smoothed_normed])
        cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7])
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header['EXTNAME'] = 'PSF_2D_TABLE'
        hdu2.header['HDUCLASS'] = 'GADF'
        hdu2.header['HDUCLAS1'] = 'RESPONSE'
        hdu2.header['HDUCLAS2'] = 'RPSF'
        hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu2.header['HDUCLAS4'] = 'PSF_TABLE'
        psf_fits = fits.HDUList([hdu, hdu2])
        psf_fits.writeto(self.save_dir+'psf.fits', overwrite=True)

        # EDISP
        col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(self.e_bins_coarse)), unit='GeV', array=[self.e_bins_coarse[:-1]])
        col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(self.e_bins_coarse)), unit='GeV', array=[self.e_bins_coarse[1:]])
        col3 = fits.Column(name='MIGRA_LO', format='{}E'.format(len(self.migra_bins)), array=[self.migra_bins[:-1]])
        col4 = fits.Column(name='MIGRA_HI', format='{}E'.format(len(self.migra_bins)), array=[self.migra_bins[1:]])
        col5 = fits.Column(name='THETA_LO', format='{}E'.format(len(self.t_bins_coarse)), unit='rad', array=[self.t_bins_coarse[:-1]])
        col6 = fits.Column(name='THETA_HI', format='{}E'.format(len(self.t_bins_coarse)), unit='rad', array=[self.t_bins_coarse[1:]])
        col7 = fits.Column(name='MATRIX', format='{}D'.format(len(self.e_bins_coarse)*len(self.migra_bins)*len(self.t_bins_coarse)),
                           dim='({},{},{})'.format(len(self.e_bins_coarse), len(self.migra_bins), len(self.t_bins_coarse)),
                           array=[self.edisp_smoothed_normed * np.diff(self.migra_bins)[:,None]])
        cols = fits.ColDefs([col1, col2, col3, col4, col5, col6, col7])
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header['EXTNAME'] = 'EDISP_2D'
        hdu2.header['HDUCLASS'] = 'GADF'
        hdu2.header['HDUCLAS1'] = 'RESPONSE'
        hdu2.header['HDUCLAS2'] = 'EDISP'
        hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu2.header['HDUCLAS4'] = 'EDISP_2D'
        edisp_fits = fits.HDUList([hdu, hdu2])
        edisp_fits.writeto(self.save_dir+'edisp.fits', overwrite=True)

        # BKG NU
        col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(self.e_bins_fine_ext)), unit='GeV',
                           array=[self.e_bins_fine_ext[:-1]])
        col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(self.e_bins_fine_ext)), unit='GeV',
                           array=[self.e_bins_fine_ext[1:]])
        col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(self.t_bins_fine)), unit='rad',
                           array=[self.t_bins_fine[:-1]])
        col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(self.t_bins_fine)), unit='rad',
                           array=[self.t_bins_fine[1:]])
        col5 = fits.Column(name='BKG', format='{}D'.format(len(self.e_bins_fine_ext)*len(self.t_bins_fine)),
                           dim='({},{})'.format(len(self.t_bins_fine), len(self.e_bins_fine_ext)), unit='MeV-1 sr-1 s-1',
                           array=[(self.atm_conv_rate + self.atm_prompt_rate)*1e-3])
        cols = fits.ColDefs([col1, col2, col3, col4, col5])
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header['EXTNAME'] = 'BACKGROUND'
        hdu2.header['HDUCLASS'] = 'GADF'
        hdu2.header['HDUCLAS1'] = 'RESPONSE'
        hdu2.header['HDUCLAS2'] = 'BKG'
        hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu2.header['HDUCLAS4'] = 'BKG_2D'
        bkg_fits = fits.HDUList([hdu, hdu2])
        bkg_fits.writeto(self.save_dir+'bkg_nu.fits', overwrite=True)

        # BKG MU
        t_bins_mu_out = np.insert(np.arccos(self.cos_bins_fine[1:] + 1./len(self.t_bins_fine)), 0, 0.0)
        col1 = fits.Column(name='ENERG_LO', format='{}E'.format(len(self.e_bins_fine_ext)), unit='GeV',
                           array=[self.e_bins_fine_ext[:-1]])
        col2 = fits.Column(name='ENERG_HI', format='{}E'.format(len(self.e_bins_fine_ext)), unit='GeV',
                           array=[self.e_bins_fine_ext[1:]])
        col3 = fits.Column(name='THETA_LO', format='{}E'.format(len(self.t_bins_fine)), unit='rad',
                           array=[t_bins_mu_out[:-1]])
        col4 = fits.Column(name='THETA_HI', format='{}E'.format(len(self.t_bins_fine)), unit='rad',
                           array=[t_bins_mu_out[1:]])
        col5 = fits.Column(name='BKG', format='{}D'.format(len(self.e_bins_fine_ext)*len(self.t_bins_fine)),
                           dim='({},{})'.format(len(self.t_bins_fine), len(self.e_bins_fine_ext)), unit='MeV-1 sr-1 s-1',
                           array=[self.atm_mu_rate_out*1e-3])
        cols = fits.ColDefs([col1, col2, col3, col4, col5])
        hdu2 = fits.BinTableHDU.from_columns(cols)
        hdu2.header['EXTNAME'] = 'BACKGROUND'
        hdu2.header['HDUCLASS'] = 'GADF'
        hdu2.header['HDUCLAS1'] = 'RESPONSE'
        hdu2.header['HDUCLAS2'] = 'BKG'
        hdu2.header['HDUCLAS3'] = 'FULL-ENCLOSURE'
        hdu2.header['HDUCLAS4'] = 'BKG_2D'
        bkg_fits_mu = fits.HDUList([hdu, hdu2])
        bkg_fits_mu.writeto(self.save_dir +'bkg_mu.fits', overwrite=True)

    def create(self):
        self.read_data()
        self.apply_selection_cuts()
        self.apply_weights()
        self.compute_aeff()
        self.compute_psf()
        self.compute_edisp()
        self._compute_atmospheric_backgrounds()
        self.write_irfs()

