#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:24:31 2024

@author: bafana
"""
import os
import time
import numpy as np
from wavespectra import read_ww3

# Constants
#g = 9.81  # acceleration due to gravity in m/s^2

# =========================================================================================================
#path = '/media/bafana/Bafana_TB2/School_work_2020/WW3_Simulations'
path = '/home/modelling/Waves_Analysis/Crossing_Swells/'
os.chdir(path)
# =========================================================================================================

start_time = time.time();

# Load wave spectra data
#dset = read_ww3('./Chapter_4/model_8km/GLORYS_cur_9km_data/PTM1_2017/NO_CURRENTS/point_spectra/ww3.NO_ACR2_201706_spec.nc', chunks={"site":1}) # No currents
dset = read_ww3('./data/ww3.NO_ACR2_201706_spec.nc', chunks={"site":1})

# Initialize a counter for crossing swell events
crossing_swell_count = 0;

# Function to wrap angle values
def wrap_angle(data):
    return (data + 180) % 360 - 180

# Iterate over all time steps
for i in range(dset.time.size):
    # Partition the wave spectra for the current time step
    ds = dset.isel(site=0, time=i).spec.split();
    dspart = ds.spec.partition.ptm1(dset.wspd, dset.wdir, dset.dpt, agefac=1.7, wscut=0.3333, swells=3, smooth=False, freq_window=3, dir_window=3, ihmax=100);

    # Extract wave parameters for each wave system
    Hs_1 = dspart.isel(part=1).spec.hs().isel(site=0);
    Dir_1 = dspart.isel(part=1).spec.dm().isel(site=0);
    
    Hs_2 = dspart.isel(part=2).spec.hs().isel(site=0);
    Dir_2 = dspart.isel(part=2).spec.dm().isel(site=0);

    # Compute direction difference between primary and secondary swell
    raw_direction_difference = Dir_1.values - Dir_2.values;
    wrapped_direction_difference = wrap_angle(raw_direction_difference);

    # Check conditions for crossing swells
    Hs_condition = Hs_2.values >= 0.5 * Hs_1.values;
    direction_condition = abs(wrapped_direction_difference) > 40;

    # If both conditions are met, count as a crossing swell event
    # Use np.any() to determine if any elements satisfy the conditions
#    if np.any(Hs_condition) and np.any(direction_condition):
        #if Hs_condition and direction_condition:
#            crossing_swell_count += 1

        # Ensure both conditions are met for at least one pair
    if np.any(Hs_condition & direction_condition):
        crossing_swell_count += 1    


# Calculate the percentage of crossing swell cases
total_time_steps = dset.time.size;
percentage_crossing_swells = (crossing_swell_count / total_time_steps) * 100;

# Print results
print(f"Total Time Steps: {total_time_steps}")
print(f"Crossing Swell Events: {crossing_swell_count}")
print(f"Percentage of Crossing Swell Cases: {percentage_crossing_swells:.2f}%")

end_time = time.time();

elapsed_time_seconds = end_time - start_time;
elapsed_time_minutes = elapsed_time_seconds / 60

print(f"Execution time: {elapsed_time_seconds: .2f} seconds")
print(f"Execution time: {elapsed_time_minutes : .2f} minutes")

# =========================================================================================================
# =========================================================================================================

