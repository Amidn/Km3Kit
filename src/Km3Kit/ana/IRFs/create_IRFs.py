import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import km3io

from astropy.io import fits
import astropy.units as u

from gammapy.irf import EnergyDispersion2D

from scipy.stats import binned_statistic
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

import pandas as pd
import uproot
from collections import defaultdict

import sys
sys.path.append('./ppflux')
sys.path.append('../')
from irf_utils import calc_theta, aeff_2D, psf_3D, edisp_3D
import flux as km3_flux
import plot_utils

 # Fix auto completion
get_ipython().Completer.use_jedi = False


filename_nu = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.1.km3_numuCC.ALL.dst.bdt.root'
filename_nubar = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.1.km3_anumuCC.ALL.dst.bdt.root'
filename_mu10 = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.2.mupage_10T.sirene_mupage.ALL.bdt.root'  
filename_mu50 = '/home/saturn/capn/mpo1217/cta/mcv5.1.bdt/bdt/mcv5.2.mupage_50T.sirene_mupage.ALL.bdt.root'


# Offline reader is not able to read the friend-tree "T" where the BDT variables are stored
# However, we will read the other variables and make sure we obtain the same arrays with uproot
# Also, we use it to read the header, just for information
f_nu_km3io = km3io.OfflineReader(filename_nu)
f_nubar_km3io = km3io.OfflineReader(filename_nubar)

 # Print header info
print('nu:')
print(f_nu_km3io.header.ngen)
print(f_nu_km3io.header.genvol)
print(f_nu_km3io.header.spectrum)
print('\nnubar:')
print(f_nubar_km3io.header.ngen)
print(f_nubar_km3io.header.genvol)
print(f_nubar_km3io.header.spectrum)

%%time
# Access data arrays
data_km3io = dict()

for l,f in zip(['nu', 'nubar'], [f_nu_km3io, f_nubar_km3io]):
    data_km3io[l] = dict()

    data_km3io[l]['E'] = f.tracks.E.to_numpy()[:,0]
    data_km3io[l]['dir_x'] = f.tracks.dir_x.to_numpy()[:,0]
    data_km3io[l]['dir_y'] = f.tracks.dir_y.to_numpy()[:,0]
    data_km3io[l]['dir_z'] = f.tracks.dir_z.to_numpy()[:,0]

    data_km3io[l]['energy_mc'] = f.mc_tracks.E.to_numpy()[:,0]
    data_km3io[l]['dir_x_mc'] = f.mc_tracks.dir_x.to_numpy()[:,0]
    data_km3io[l]['dir_y_mc'] = f.mc_tracks.dir_y.to_numpy()[:,0]
    data_km3io[l]['dir_z_mc'] = f.mc_tracks.dir_z.to_numpy()[:,0]

    data_km3io[l]['weight_w2'] = f.w.to_numpy()[:,1]

f_nu_uproot = uproot.open(filename_nu)
f_nubar_uproot = uproot.open(filename_nubar)


%%time
# Retrieve data
data_uproot = dict()

for l,f in zip(['nu', 'nubar'], [f_nu_uproot, f_nubar_uproot]):
    # Retrieve trees
    E = f['E;1']
    T = f['T;1']

    # Access data arrays
    data_uproot[l] = dict()

    data_uproot[l]['E'] = E['Evt/trks/trks.E'].array().to_numpy()[:,0]
    data_uproot[l]['dir_x'] = E['Evt/trks/trks.dir.x'].array().to_numpy()[:,0]
    data_uproot[l]['dir_y'] = E['Evt/trks/trks.dir.y'].array().to_numpy()[:,0]
    data_uproot[l]['dir_z'] = E['Evt/trks/trks.dir.z'].array().to_numpy()[:,0]

    data_uproot[l]['energy_mc'] = E['Evt/mc_trks/mc_trks.E'].array().to_numpy()[:,0]
    data_uproot[l]['dir_x_mc'] = E['Evt/mc_trks/mc_trks.dir.x'].array().to_numpy()[:,0]
    data_uproot[l]['dir_y_mc'] = E['Evt/mc_trks/mc_trks.dir.y'].array().to_numpy()[:,0]
    data_uproot[l]['dir_z_mc'] = E['Evt/mc_trks/mc_trks.dir.z'].array().to_numpy()[:,0]

    data_uproot[l]['weight_w2'] = E['Evt/w'].array().to_numpy()[:,1]
    bdt = T['bdt'].array().to_numpy()
    data_uproot[l]['bdt0'] = bdt[:,0]
    data_uproot[l]['bdt1'] = bdt[:,1]

# Make sure we have read the same arrays with km3io and uproot
for l in ['nu', 'nubar']:
    print(l)
    for k in data_km3io[l].keys():
        print(k, np.all(data_km3io[l][k] == data_uproot[l][k]))
    print('')


 # Create pandas DataFrames
df_nu = pd.DataFrame(data_uproot['nu'])
df_nubar = pd.DataFrame(data_uproot['nubar'])


files_atm_mu = [filename_mu10, filename_mu50]
live_times_mu = []
for fname in files_atm_mu:
    f = km3io.OfflineReader(fname)
    t = f.header.livetime.numberOfSeconds
    print(fname.split('/')[-1], t)
    live_times_mu.append(t)
print('Total', np.sum(live_times_mu))

%%time
# Access data arrays for atm. muons
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

    # compute a weight, based on the livetime
    w = np.full(len(data_mu['E'][-1]), 1. / live_times_mu[i])
    data_mu['weight'].append(w)

for k in data_mu:
    data_mu[k] = np.concatenate(data_mu[k])


# Create data frame
df_mu = pd.DataFrame(data_mu) 


# Implement selection cuts
def get_q_mask(bdt0, bdt1, dir_z):
    '''
    bdt0: to determine groups to which BDT cut should be applied (upgoing/horizontal/downgoing).
    bdt1: BDT score in the range [-1, 1]. Closer to 1 means more signal-like.
    dir_z: is the reconstructed z-direction of the event
    '''
    mask_down = bdt0 >= 11  # remove downgoing events
    clear_signal = bdt0 == 12 # very clear signal
    loose_up = (np.arccos(dir_z)*180/np.pi < 80) & (bdt1 > 0.) # apply loose cut on upgoing events
    strong_horizontal = (np.arccos(dir_z)*180/np.pi > 80) & (bdt1 > 0.7) # apply strong cut on horizontal events
    return mask_down & ( clear_signal | loose_up | strong_horizontal )



# Apply the cuts

# nu
q_mask_nu = get_q_mask(df_nu.bdt0, df_nu.bdt1, df_nu.dir_z)
df_nu_q = df_nu[q_mask_nu].copy()

# nubar
q_mask_nubar = get_q_mask(df_nubar.bdt0, df_nubar.bdt1, df_nubar.dir_z)
df_nubar_q = df_nubar[q_mask_nubar].copy()


print('nu: {:d} events survive cuts ({:.4g}%)'.format(len(df_nu_q), len(df_nu_q)/len(df_nu)*100))
print('nubar: {:d} events survive cuts ({:.4g}%)'.format(len(df_nubar_q), len(df_nubar_q)/len(df_nubar)*100))


 # calculate the normalized weight factor for each event
weight_factor = -2.5  # Spectral index to re-weight to

weights = dict()
for l,df,f in zip(['nu', 'nubar'], [df_nu_q, df_nubar_q], [f_nu_km3io, f_nubar_km3io]):
    weights[l] = (df.energy_mc**(weight_factor - f.header.spectrum.alpha)).to_numpy()
    weights[l] *= len(df) / weights[l].sum()


# Create DataFrames with neutrinos and anti-neutrinos
df_nu_all = pd.concat([df_nu, df_nubar], ignore_index=True)
df_nu_all_q = pd.concat([df_nu_q, df_nubar_q], ignore_index=True)

# Also create a concatenated array for the weights
weights_all = np.concatenate([weights['nu'], weights['nubar']])


q_mask_mu = get_q_mask(df_mu.bdt0, df_mu.bdt1, df_mu.dir_z)
df_mu_q = df_mu[q_mask_mu].copy()


print('{:d} events survive cuts ({:.4g}%)'.format(len(df_mu_q), len(df_mu_q)/len(df_mu)*100))


# Define bins for IRFs
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
# rad_bins = 10**np.linspace(-3, 3, 41)  # used by Rasa


# Bin centers
e_binc_fine = np.sqrt(e_bins_fine[:-1] * e_bins_fine[1:])
e_binc_coarse = np.sqrt(e_bins_coarse[:-1] * e_bins_coarse[1:])

t_binc_fine = np.arccos(0.5*(cos_bins_fine[:-1] + cos_bins_fine[1:]))
t_binc_coarse = np.arccos(0.5*(cos_bins_coarse[:-1] + cos_bins_coarse[1:]))

migra_binc = np.sqrt(migra_bins[:-1] * migra_bins[1:])
rad_binc = 0.5*(rad_bins[1:] + rad_bins[:-1])


 # Also define energy binning that extends to lower energies, since events get shifted there
e_bins_fine_ext = np.logspace(0, 8, 65)
e_binc_fine_ext = np.sqrt(e_bins_fine_ext[:-1] * e_bins_fine_ext[1:])

assert (e_bins_fine_ext[16:] == e_bins_fine).all()

%%time
# Fill histograms
aeff = aeff_2D(
    e_bins  = e_bins_fine,
    t_bins  = t_bins_fine,
    dataset = df_nu_all_q,
    gamma   = (-weight_factor),
    nevents = f_nu_km3io.header.genvol.numberOfEvents + f_nubar_km3io.header.genvol.numberOfEvents
) * 2 # two building blocks

psf = psf_3D(
    e_bins  = e_bins_coarse,
    r_bins  = rad_bins,
    t_bins  = t_bins_coarse,
    dataset = df_nu_all_q,
    weights = weights_all
)

edisp = edisp_3D(
    e_bins  = e_bins_coarse,
    m_bins  = migra_bins,
    t_bins  = t_bins_coarse,
    dataset = df_nu_all_q,
    weights = weights_all
)


aeff_noreweight = aeff_2D(
    e_bins  = e_bins_fine,
    t_bins  = t_bins_fine,
    dataset = df_nu_all_q,
    gamma = (2.),
    nevents = f_nu_km3io.header.genvol.numberOfEvents + f_nubar_km3io.header.genvol.numberOfEvents
) * 2 # two building blocks


aeff_ratio = aeff/aeff_noreweight

aeff_ratio.shape

np.nanmin(aeff_ratio), np.nanmax(aeff_ratio), np.nanmean(aeff_ratio), np.nanstd(aeff_ratio)



# Also compute effective area for nu and nubar separately
aeff_nu = aeff_2D(
    e_bins  = e_bins_fine,
    t_bins  = t_bins_fine,
    dataset = df_nu_q,
    gamma   = (-weight_factor),
    nevents = f_nu_km3io.header.genvol.numberOfEvents
) * 2 # two building blocks

aeff_nubar = aeff_2D(
    e_bins  = e_bins_fine,
    t_bins  = t_bins_fine,
    dataset = df_nubar_q,
    gamma   = (-weight_factor),
    nevents = f_nubar_km3io.header.genvol.numberOfEvents
) * 2 # two building blocks


# Also fill an effective area without cuts, for comparison
aeff_uncut = aeff_2D(
    e_bins  = e_bins_fine,
    t_bins  = t_bins_fine,
    dataset = df_nu_all,
    gamma   = (-weight_factor),
    nevents = f_nu_km3io.header.genvol.numberOfEvents + f_nubar_km3io.header.genvol.numberOfEvents
) * 2 # two building blocks


# Norm for PSF
sizes_rad_bins = np.diff(rad_bins**2)

# take different size of solid angle into account, i.e. compute dP/dOmega
psf_weighted = psf / (sizes_rad_bins[:,None,None] * (np.pi/180)**2 * np.pi)

# normalise
norm_psf = psf.sum(axis=0, keepdims=True)
psf_normed = np.nan_to_num(psf_weighted / norm_psf)

 # Norm for EDISP
sizes_migra_bins = np.diff(migra_bins)

# important - matrix entries should be dP/dmigra!
edisp /= sizes_migra_bins[:,np.newaxis]

# normalise
m_normed = edisp * sizes_migra_bins[:,np.newaxis]
norm_edisp = m_normed.sum(axis=1, keepdims=True)
edisp_normed = np.nan_to_num(edisp / norm_edisp)

 # Smooth the energy dispersion
edisp_smoothed = np.zeros_like(edisp)

# loop over energy and zenith bins
for i in range(edisp.shape[-1]):
    for j in range(edisp.shape[0]):
        # choose kernel size according to available statistics
        # found to work well empirically
        kernel_size = 2 - 0.25*max(0, np.log10(edisp[j,:,i].sum()))
        edisp_smoothed[j,:,i] = gaussian_filter1d(edisp[j,:,i] * sizes_migra_bins,  # smooth P, not dP/dmigra
                                                  kernel_size, axis=0, mode='nearest')

# convert to dP/dmigra again
edisp_smoothed /= sizes_migra_bins[:,None]

# normalise
m_normed = edisp_smoothed * sizes_migra_bins[:,np.newaxis]
norm_edisp_sm = m_normed.sum(axis=1, keepdims=True)
edisp_smoothed_normed = np.nan_to_num(edisp_smoothed / norm_edisp_sm)

# Check smoothing of edisp
plt.figure(dpi=150)
z = 4   # choose the zenith bin for plotting

step_e_idx = int(len(e_binc_coarse) / (np.log10(e_bins_coarse[-1]) - np.log10(e_bins_coarse[0])))
for i in range(0, len(e_binc_coarse), step_e_idx):
    plt.plot(migra_bins[:-1], edisp_normed[z,:,i] * sizes_migra_bins,
             color=plt.cm.plasma(i/len(e_binc_coarse)), 
             label='{:g} GeV'.format(e_bins_coarse[i]), alpha=0.6, ds='steps-post')
    plt.plot(migra_binc, edisp_smoothed_normed[z,:,i] * sizes_migra_bins,
             color=plt.cm.plasma(i/len(e_binc_coarse)), alpha=1)

plt.semilogx()
plt.grid(ls='--')
plt.legend(fontsize='small')
plt.xlabel('Energy migration $\mu$')
plt.ylabel('Matrix value')
plt.title('Edisp at {:.1f} deg zenith'.format(t_bins_coarse[z]/np.pi*180));

 # Smooth the psf
# use different kernel sizes in different RAD axis ranges
s1 = gaussian_filter1d(psf_weighted, 0.5, axis=0, mode='nearest')
s2 = gaussian_filter1d(psf_weighted, 2,   axis=0, mode='nearest')
s3 = gaussian_filter1d(psf_weighted, 4,   axis=0, mode='nearest')
s4 = gaussian_filter1d(psf_weighted, 6,   axis=0, mode='constant')
psf_smoothed = np.concatenate((s1[:10], s2[10:20], s3[20:60], s4[60:-1], [psf_weighted[-1]]), axis=0)
# smooth edges between the different ranges
psf_smoothed[10:-1] = gaussian_filter1d(psf_smoothed[10:-1], 1, axis=0, mode='nearest')

# normalise
norm_psf_sm = (psf_smoothed * sizes_rad_bins[:,None,None] * (np.pi/180)**2 * np.pi).sum(axis=0, keepdims=True)
psf_smoothed_normed = np.nan_to_num(psf_smoothed / norm_psf_sm)


plt.figure(dpi=150)
z = 5   # choose the zenith bin for plotting

step_e_idx = int(len(e_binc_coarse) / (np.log10(e_bins_coarse[-1]) - np.log10(e_bins_coarse[0])))
for i in range(0, len(e_binc_coarse), step_e_idx):
    plt.plot(rad_bins[:-1], psf_normed[:,z,i],
             color=plt.cm.plasma(i/len(e_binc_coarse)), 
             label='{:g} GeV'.format(e_bins_coarse[i]), ds='steps-post', alpha=0.6)
    plt.plot(rad_binc, psf_smoothed_normed[:,z,i],
             color=plt.cm.plasma(i/len(e_binc_coarse)), alpha=1)

for v in [0.5, 1, 5, 30]:
    plt.axvline([v], color='k', ls='--')
plt.loglog()
plt.legend()
plt.ylim(1e-5, 1e5)
plt.xlabel('Radial angle [deg]')
plt.ylabel('Matrix value')
plt.title('PSF at {:.1f} deg'.format(t_bins_coarse[z]/np.pi*180))


def reference_flux(E):
    return 0.5 * 1.2e-4 * E**(-2) * np.exp(-E/3e6)    # in [GeV-1 s-1 m-2 sr-1]

def reference_flux(E):
    return 0.5 * 1.2e-4 * E**(-2) * np.exp(-E/3e6)    # in [GeV-1 s-1 m-2 sr-1]

ref_flux = reference_flux(e_binc_fine)


e_content = np.diff(e_bins_fine)
s_content = 4 * np.pi / len(t_binc_fine)
event_rate_nu = aeff_nu * ref_flux[:,None] * e_content[:,None] * s_content
event_rate_nubar = aeff_nubar * ref_flux[:,None] * e_content[:,None] * s_content 


 # events per year for one building block
print('nu: {:.5g} (internal note: 49.6053)'.format(event_rate_nu.sum()*3600*24*365/2))
print('nubar: {:.5g} (internal note: 39.957)'.format(event_rate_nubar.sum()*3600*24*365/2))


 # get the quantiles directly from events 
km3_psf_cr50 = binned_statistic(df_nu_all_q.energy_mc, df_nu_all_q.rad, statistic=lambda a:np.quantile(a, 0.5),
                                bins=e_bins_fine)[0]
km3_psf_cr90 = binned_statistic(df_nu_all_q.energy_mc, df_nu_all_q.rad, statistic=lambda a:np.quantile(a, 0.9),
                                bins=e_bins_fine)[0]


# Write to disk
np.savetxt('psf_edisp_quantiles/psf_q50.dat', np.array([e_binc_fine, km3_psf_cr50]).T, header='E[GeV]  CR[deg]')
np.savetxt('psf_edisp_quantiles/psf_q90.dat', np.array([e_binc_fine, km3_psf_cr90]).T, header='E[GeV]  CR[deg]') 


# # TO DO compute any quantiles here?


# load knee corrections
honda_knee_e, honda_knee_f = np.loadtxt('honda_knee_correction_gaisser_H3a.dat', unpack=True)
enberg_knee_e, enberg_knee_f = np.loadtxt('enberg_knee_correction_gaisser_H3a.dat', unpack=True)

# define interpolation methods
honda_knee_correction = lambda e:np.interp(np.log10(e), np.log10(honda_knee_e), honda_knee_f)
enberg_knee_correction = lambda e:np.interp(np.log10(e), np.log10(enberg_knee_e), enberg_knee_f)

atm_conv_flux = dict()
atm_prompt_flux = dict()

atm_conv_flux['nu'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))
atm_conv_flux['nubar'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))
atm_prompt_flux['nu'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))
atm_prompt_flux['nubar'] = np.zeros((len(e_binc_fine), len(t_binc_fine)))

for l,pt in zip(['nu', 'nubar'], [14, -14]):
    for i in range(len(t_binc_fine)):
        # retrieve fluxes
        # conventional
        conv_flux = np.array([km3_flux.atmospheric_conventional(mc_type=pt, zenith=t_binc_fine[i], energy=e) \
                              for e in e_binc_fine])

        # prompt
        prompt_flux = np.array([km3_flux.atmospheric_prompt(mc_type=pt, zenith=t_binc_fine[i], energy=e) \
                               for e in e_binc_fine])

        # apply knee corrections
        conv_flux *= honda_knee_correction(e_binc_fine)
        prompt_flux *= enberg_knee_correction(e_binc_fine)
        # prompt_flux *= honda_knee_correction(e_binc_fine) # for testing: use correction as used in KM3NeT

        # store
        atm_conv_flux[l][:,i] = conv_flux
        atm_prompt_flux[l][:,i] = prompt_flux

# compute rate (this is still in true energy!)
atm_conv_rate_etrue = 0
atm_prompt_rate_etrue = 0
for l,a in zip(['nu', 'nubar'], [aeff_nu, aeff_nubar]):
    atm_conv_rate_etrue += atm_conv_flux[l] * a
    atm_prompt_rate_etrue += atm_prompt_flux[l] * a


# Create EnergyDispersion2D instance

# Normally, it is not necessary to multiply with the migra bin size here, since
# the stored quantity should be dP/dmigra. However, there is a bug in Gammapy 0.17 - 
# when the response is evaluated, the multiplication with the migra bin size is
# missing. Hence, we already perform the multiplication here (i.e. pass dP instead
# of dP/dmigra). The same is true when we write the EDISP IRF to disk (see further
# below).

edisp_cls = EnergyDispersion2D(e_bins_coarse[:-1]*u.GeV, e_bins_coarse[1:]*u.GeV,
                               migra_bins[:-1], migra_bins[1:],
                               t_bins_coarse[:-1]*u.deg, t_bins_coarse[1:]*u.deg,
                               edisp_smoothed_normed.T * sizes_migra_bins[:,None]) 


# apply energy dispersion to go from e_true to e_reco
atm_conv_rate = np.zeros((len(e_binc_fine_ext), len(t_binc_fine)))
atm_prompt_rate = np.zeros((len(e_binc_fine_ext), len(t_binc_fine)))

print(' i  zenith  sum along ereco')
for i in range(len(t_binc_fine)):
    # evaluate energy dispersion matrix for this zenith angle
    edisp_matrix = edisp_cls.to_energy_dispersion(t_binc_fine[i]*u.deg,
                                                  e_true = e_bins_fine*u.GeV,
                                                  e_reco = e_bins_fine_ext*u.GeV)

    # printout for debugging
    print('{:>2} {:>6.1f} {:>12.4f}'.format(i, t_binc_fine[i]*180/np.pi, edisp_matrix.pdf_matrix.sum(axis=1).mean()))

    # multiply with energy bin widths for EDISP multiplication
    conv_etrue = atm_conv_rate_etrue[:,i] * np.diff(e_bins_fine)
    prompt_etrue = atm_prompt_rate_etrue[:,i] * np.diff(e_bins_fine)

    # apply EDISP
    conv = np.dot(conv_etrue, edisp_matrix.pdf_matrix)
    prompt = np.dot(prompt_etrue, edisp_matrix.pdf_matrix)

    # divide by energy bin widths again
    conv /= np.diff(e_bins_fine_ext)
    prompt /= np.diff(e_bins_fine_ext)

    # store
    atm_conv_rate[:,i] = conv
    atm_prompt_rate[:,i] = prompt 


 # Compute zenith angle
df_mu_q['theta'] = calc_theta(df_mu_q, mc=False)

# fill histogram, weight with inverse simulated live time
atm_mu_bkg = np.histogram2d(df_mu_q.E, df_mu_q.theta,
                            bins=(e_bins_fine_ext, t_bins_fine), weights=df_mu_q.weight)[0]

atm_mu_bkg *= 2  # two building blocks


 # divide by solid angle and energy bin width
atm_mu_rate = atm_mu_bkg / (np.pi * 4 / len(t_binc_fine) * np.diff(e_bins_fine_ext)[:,None])
# atm_mu_rate_smoothed = gaussian_filter(atm_mu_rate, [0.1, 0.4], mode='nearest')


de_domega = 4 * np.pi / len(t_binc_fine) * np.diff(e_bins_fine)
de_domega_reco = 4 * np.pi / len(t_binc_fine) * np.diff(e_bins_fine_ext)
t_mask = t_binc_fine * 180 / np.pi > 80  # compute rate for analysis region
t_year = 365 * 24 * 3600



# Comparison of atmospheric background rates, across all zenith angle bins
fig,ax = plt.subplots(dpi=150)

ax.set_title('Atmospheric backgrounds')
ax.plot(e_binc_fine_ext, atm_mu_rate[:,t_mask].sum(axis=1) * de_domega_reco * t_year,
        '+-', label='Muons ($E_\mathrm{reco}$)', c='blue', ds='steps-mid')
ax.plot(e_binc_fine, atm_conv_rate_etrue[:,t_mask].sum(axis=1) * de_domega * t_year,
        '+-', label='Conv. neutrinos ($E_\mathrm{true}$)', c='orange', ds='steps-mid')
ax.plot(e_binc_fine_ext, atm_conv_rate[:,t_mask].sum(axis=1) * de_domega_reco * t_year,
        '+-', label='Conv. neutrinos ($E_\mathrm{reco}$)', c='red', ds='steps-mid')
ax.plot(e_binc_fine, atm_prompt_rate_etrue[:,t_mask].sum(axis=1) * de_domega * t_year,
        '+-', label='Prompt neutrinos ($E_\mathrm{true}$)', c='green', ds='steps-mid')
ax.plot(e_binc_fine_ext, atm_prompt_rate[:,t_mask].sum(axis=1) * de_domega_reco * t_year,
        '+-', label='Prompt neutrinos ($E_\mathrm{reco}$)', c='cyan', ds='steps-mid')

ax.loglog()
ax.set_xlabel('$E$ [GeV]')
ax.set_ylabel('Rate [yr$^{-1}$]')
ax.legend(loc='upper right', fontsize='small')
ax.set_ylim(3e-5, 3e4)
ax.grid(ls='--')


# Muon background vs energy and zenith angle
fig,ax = plt.subplots(figsize=(5,4), dpi=150)
im = ax.imshow(
    atm_mu_rate * de_domega_reco[:,None] * t_year, origin='lower',
    extent=[np.cos(t_bins_fine[0]), np.cos(t_bins_fine[-1]),
            np.log10(e_bins_fine_ext[0]), np.log10(e_bins_fine_ext[-1])]
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Rate [yr$^{-1}$]')

ax.set(ylabel=r'$\log_{10}$($E_\mathrm{reco}$ [GeV])', xlabel=r'$\cos(\theta)$')
ax.set_aspect('auto')


# Background rates per zenith angle bin
# (same color code as in previous plot above)

fig,axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(10, 10), dpi=150)

idx_offset = 3
for i in range(len(t_binc_fine)-idx_offset):
    ax = axes[i//3][i%3]
    ax.set_title('Zenith: {:.1f}$^\circ$'.format(t_binc_fine[idx_offset+i]*180/np.pi), fontsize='medium')
    ax.plot(e_binc_fine_ext, atm_mu_rate[:,idx_offset+i] * de_domega_reco * t_year,
            '+-', c='blue', ds='steps-mid')
    ax.plot(e_binc_fine, atm_conv_rate_etrue[:,idx_offset+i] * de_domega * t_year,
            '+-', c='orange', ds='steps-mid')
    ax.plot(e_binc_fine_ext, atm_conv_rate[:,idx_offset+i] * de_domega_reco * t_year,
            '+-', c='red', ds='steps-mid')
    ax.plot(e_binc_fine, atm_prompt_rate_etrue[:,idx_offset+i] * de_domega * t_year,
            '+-', c='green', ds='steps-mid')
    ax.plot(e_binc_fine_ext, atm_prompt_rate[:,idx_offset+i] * de_domega_reco * t_year,
            '+-', c='cyan', ds='steps-mid')

    ax.loglog()
    if i > 5:
        ax.set_xlabel('$E$ [GeV]')
    if i%3 == 0:
        ax.set_ylabel('Rate [yr$^{-1}$]')
    ax.set_xlim(10, 1e7)
    ax.set_ylim(3e-3, 3e3)
    ax.grid(ls='--')
    
    if i < 2:
        ax.fill_between([10, 1e7], [3e-3, 3e-3], [3e3, 3e3], color='0.5', alpha=0.3)
        ax.text(0.5, 0.97, '(not in analysis region)', ha='center', va='top', transform=ax.transAxes) 