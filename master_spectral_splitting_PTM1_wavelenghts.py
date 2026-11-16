#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:07:43 2024

@author: bafana
"""
#%matplotlib inline
import os
import matplotlib.pyplot as plt
from wavespectra import read_ww3
import numpy as np

# Constants
g = 9.81  # acceleration due to gravity in m/s^2

# =========================================================================================================
path = '/media/bafana/Bafana_TB2/School_work_2020/WW3_Simulations'
os.chdir(path)
# =========================================================================================================
# In this code we attempt to plot wave spectra from WWIII simulations (with & without surface currents ...)
# Here we read wave spectra file from WWIII spectral wave model

#dset = read_ww3('./Chapter_4/model_8km/GLORYS_cur_9km_data/PTM1_2017/NO_CURRENTS/point_spectra/ww3.NO_ACR1_2017_spec.nc', chunks={"site":1}) # No currents
dset = read_ww3('./Chapter_4/model_8km/GLORYS_cur_9km_data/PTM1_2017/CURRENTS/point_spectra/ww3.YES_ACR1_2017_spec.nc', chunks={"site":1}) # With Currents

# =========================================================================================================
# Plotting 2D Energy density spectra
#ds = dset.isel(site=0, time=1159).spec.split(); # 2017-05-25T21:00:00
#ds = dset.isel(site=0, time=10).spec.split(); # 2017-01-02T06:00:00
#ds = dset.isel(site=0, time=1886).spec.split(); # 2017-08-24T18:00:00

ds = dset.isel(site=0, time=1886).spec.split();


# Partitioning wave spectra for 2017-08-11T03:00:00
dspart = ds.spec.partition.ptm1(dset.wspd, dset.wdir, dset.dpt, agefac=1.7, wscut=0.3333, swells=3, smooth=False, freq_window=3, dir_window=3, ihmax=100)

# Plot the partitioned spectra excluding the first part
f = dspart.isel(part=[1, 2, 3], time=1886, site=0, drop=True).spec.plot(kind="contourf", col="part", 
    show_direction_label=True, as_period=True, logradius=False, radii_labels_size=10, 
    cmap="Spectral_r", cbar_kwargs={"label": "E [$m^{2}s/deg$]"}, figsize=(15, 4))

# Extract integrated wave parameters for each wave system
Hs_1 = dspart.isel(part=1).spec.hs().isel(time=1886, site=0);
Tp_1 = dspart.isel(part=1).spec.tp().isel(time=1886, site=0);
Dir_1 = dspart.isel(part=1).spec.dm().isel(time=1886, site=0);

Hs_2 = dspart.isel(part=2).spec.hs().isel(time=1886, site=0);
Tp_2 = dspart.isel(part=2).spec.tp().isel(time=1886, site=0);
Dir_2 = dspart.isel(part=2).spec.dm().isel(time=1886, site=0);

Hs_3 = dspart.isel(part=3).spec.hs().isel(time=1886, site=0);
Tp_3 = dspart.isel(part=3).spec.tp().isel(time=1886, site=0);
Dir_3 = dspart.isel(part=3).spec.dm().isel(time=1886, site=0);

# Calculate wavelengths for each partition using the deep water approximation
Lambda_1 = (g * Tp_1**2) / (2 * np.pi);
Lambda_2 = (g * Tp_2**2) / (2 * np.pi);
Lambda_3 = (g * Tp_3**2) / (2 * np.pi);

# Add titles to the subplots and text boxes with wave parameters
Hs_values = [Hs_1, Hs_2, Hs_3];
Tp_values = [Tp_1, Tp_2, Tp_3];
Dir_values = [Dir_1, Dir_2, Dir_3];
Lambda_values = [Lambda_1, Lambda_2, Lambda_3];

titles = ["Primary Swell", "Secondary Swell", "Decaying Swell"];

for ind, ax in enumerate(f.axes.flat):  
    ax.set_title(titles[ind])
    textstr = (f"Hs = {Hs_values[ind].values:.2f} m\n"
               f"Tp = {Tp_values[ind].values:.2f} s\n"
               f"Dir = {Dir_values[ind].values:.2f}°\n"
               f"λ = {Lambda_values[ind].values:.2f} m")
    ax.text(0.05, 0.95, textstr, horizontalalignment='left', verticalalignment='top', 
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))

# Save the swell spectra plots individually
for i, ax in enumerate(f.axes.flat):
    extent = ax.get_window_extent().transformed(plt.gcf().transFigure.inverted())
  #  plt.savefig(f'./Chapter_6/2D_wavespectra_images/2D_part_spectra_images/YES_ACR2_part_spectra_{titles[i].replace(" ", "_").lower()}_2017-05-25@2100UTC.png', dpi=300, bbox_inches='tight')
    plt.savefig('./Chapter_6/2D_wavespectra_images/2D_part_spectra_images/YES_ACR1_part_spectra_2017-08-24@1800UTC.png', dpi=300, bbox_inches='tight')

plt.show()

# =========================================================================================================
# Criteria for identifying crossing swells

# Function to wrap angle values
def wrap_angle(data):
    return (data + 180) % 360 - 180

# Compute direction difference between primary and secondary swell
raw_direction_difference = Dir_1.values - Dir_2.values;
wrapped_direction_difference = wrap_angle(raw_direction_difference);

# Check if the significant wave height of the secondary swell is greater than half of the primary swell
Hs_condition = Hs_2.values >= 0.5 * Hs_1.values;

# Check if the directional difference is greater than 45°
direction_condition = abs(wrapped_direction_difference) > 40;

# Print the wrapped direction difference
print(f"Wrapped Direction Difference = ({raw_direction_difference:.1f}° + 180) % 360 - 180 = {wrapped_direction_difference:.1f}°")

# Identify if it's a crossing swell
if Hs_condition and direction_condition:
    print("Crossing Swell Identified:")
    print(f"Primary Swell Hs: {Hs_1.values:.2f} m, Tp: {Tp_1.values:.2f} s, Direction: {Dir_1.values:.2f}°, Wavelength: {Lambda_1.values:.2f} m")
    print(f"Secondary Swell Hs: {Hs_2.values:.2f} m, Tp: {Tp_2.values:.2f} s, Direction: {Dir_2.values:.2f}°, Wavelength: {Lambda_2.values:.2f} m")
    print(f"Direction Difference: {wrapped_direction_difference:.1f}°")
    print("This condition is also satisfied since the direction difference is greater than 40°.")
else:
    if not Hs_condition:
        print("Crossing Swell Not Identified: The significant wave height condition is not satisfied.")
    if not direction_condition:
        print("Crossing Swell Not Identified: The direction difference condition is not satisfied.")
