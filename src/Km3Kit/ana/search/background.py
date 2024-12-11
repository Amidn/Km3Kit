import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from gammapy.maps import WcsGeom, MapAxis, WcsNDMap
from gammapy.irf import Background2D, EffectiveAreaTable2D
from astropy.coordinates import SkyCoord

# Define constants
time_step = 3600  # 1-hour time step
start = Time('2019-01-01T00:00:00', format='isot').unix
end = Time('2020-01-01T00:00:00', format='isot').unix
times = np.linspace(start, end - time_step, int(365 * 24 * 3600 / time_step))
obstimes = Time(times, format='unix')

# Define ARCA position
pos_arca = EarthLocation.from_geodetic(lat="36° 16'", lon="16° 06'", height=-3500)
frames = AltAz(obstime=obstimes, location=pos_arca)

# Zenith angle binning
cos_zen_bins = np.linspace(-1, 1, 13)
zen_bins = np.arccos(cos_zen_bins) * 180 / np.pi
zen_binc = 0.5 * (zen_bins[:-1] + zen_bins[1:])
bin_mask = zen_binc > 80

# Define map geometry
Crab = SkyCoord(184.5543, -5.7805, frame="galactic",  unit="deg")
energy_axis = MapAxis.from_bounds(1e2, 1e6, nbin=16, unit='GeV', name='energy', interp='log')
geom = WcsGeom.create(
    binsz=0.1 * u.deg,
    width=30 * u.deg,
    skydir=Crab,
    frame='icrs',
    axes=[energy_axis],
)

# Read IRFs
aeff = EffectiveAreaTable2D.read('./IRFs/aeff.fits')
bkg_nu = Background2D.read('./IRFs/bkg_nu.fits')
bkg_mu = Background2D.read('./IRFs/bkg_mu.fits')

# Compute solid angle for each pixel
d_omega = geom.to_image().solid_angle()

# Background evaluation functions
def calc_exposure(offset, aeff, geom):
    energy = geom.axes[0].center
    exposure = aeff.data.evaluate(offset=offset, energy_true=energy[:, np.newaxis, np.newaxis])
    return exposure

# Make arrays the size of the number of obstimes
nu_background_maps = np.zeros((len(obstimes), *geom.data_shape))
mu_background_maps = np.zeros((len(obstimes), *geom.data_shape))

# Use energy centers instead of edges
energy_centers = geom.axes[0].center

print(len(obstimes))

# Loop over all observation times
for i in range(len(obstimes)):
    # Zenith angle map for the current time bin
    zen_vals = geom.to_image().get_coord().skycoord.transform_to(frames[i]).zen.value

    # Neutrino background
    bkg_nu_de = bkg_nu.evaluate(
        offset=zen_vals * u.deg,
        energy=energy_centers[:, np.newaxis, np.newaxis]
    )
    nu_background_maps[i] += (bkg_nu_de * d_omega).to_value('1 / (MeV s)')

    # Muon background
    bkg_mu_de = bkg_mu.evaluate(
        offset=zen_vals * u.deg,
        energy=energy_centers[:, np.newaxis, np.newaxis]
    )
    mu_background_maps[i] += (bkg_mu_de * d_omega).to_value('1 / (MeV s)')

# Save the background maps
np.save('nu_background_map.npy', nu_background_maps)
np.save('mu_background_map.npy', mu_background_maps)