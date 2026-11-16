#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from wavespectra import read_ww3
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Constants
path = '/home/modelling/Waves_Analysis/Crossing_Swells'
os.chdir(path)

def wrap_angle(data):
    """Wrap angle values to [-180, 180] range"""
    return (data + 180) % 360 - 180

def process_timestep(timestep_data):
    """Process a single timestep - receives pre-loaded data"""
    i, ds_timestep = timestep_data
    
    try:
        # Partition the wave spectra for this timestep
        ds = ds_timestep.spec.split()
        
        dspart = ds.spec.partition.ptm1(
            ds_timestep.wspd, ds_timestep.wdir, ds_timestep.dpt,
            agefac=1.7, wscut=0.3333, swells=3, 
            smooth=False, freq_window=3, dir_window=3, ihmax=100
        )
        
        # Extract wave parameters - note: no .isel(site=0) needed since we already selected site=0
        Hs_1 = dspart.isel(part=1).spec.hs().values
        Dir_1 = dspart.isel(part=1).spec.dm().values
        Hs_2 = dspart.isel(part=2).spec.hs().values
        Dir_2 = dspart.isel(part=2).spec.dm().values
        
        # Compute direction difference
        raw_direction_difference = Dir_1 - Dir_2
        wrapped_direction_difference = wrap_angle(raw_direction_difference)
        
        # Check conditions
        Hs_condition = Hs_2 >= 0.5 * Hs_1
        direction_condition = np.abs(wrapped_direction_difference) > 40
        
        # Return 1 if crossing swell event detected
        return 1 if (Hs_condition & direction_condition) else 0
        
    except Exception as e:
        print(f"Error at timestep {i}: {e}")
        return 0

def main():
    start_time = time.time()
    
    # Load data with site dimension
    print("Loading data...")
    dset = read_ww3('2017/YES/ww3.YES_ACR2_201706_spec.nc', chunks={"site": 1})
    
    # Select site 0 once (assuming you only have one site)
    dset_site = dset.isel(site=0)
    
    total_time_steps = dset_site.time.size
    print(f"Total time steps: {total_time_steps}")
    
    # Prepare data for each timestep - keep as xarray objects, not scalars
    timesteps_data = []
    for i in range(total_time_steps):
        # Extract the full timestep as an xarray object
        ds_t = dset_site.isel(time=i)
        timesteps_data.append((i, ds_t))
    
    # Determine number of cores to use (leave 1-2 cores free for system)
    n_cores = max(1, min(cpu_count() - 1, total_time_steps))
    print(f"Using {n_cores} CPU cores...")
    
    # Process in parallel
    with Pool(processes=n_cores) as pool:
        results = pool.map(process_timestep, timesteps_data)
    
    crossing_swell_count = sum(results)
    
    # Calculate percentage
    percentage_crossing_swells = (crossing_swell_count / total_time_steps) * 100
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Total Time Steps: {total_time_steps}")
    print(f"Crossing Swell Events: {crossing_swell_count}")
    print(f"Percentage of Crossing Swell Cases: {percentage_crossing_swells:.2f}%")
    print(f"{'='*50}")
    
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    
    print(f"\nExecution time: {elapsed_time_seconds:.2f} seconds")
    print(f"Execution time: {elapsed_time_minutes:.2f} minutes")
    
    # Calculate speedup compared to original (3150.82 seconds)
    speedup = 3150.82 / elapsed_time_seconds if elapsed_time_seconds > 0 else 0
    print(f"Speedup: {speedup:.1f}x compared to serial version")

if __name__ == "__main__":
    main()
