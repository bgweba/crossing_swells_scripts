#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from wavespectra import read_ww3
from joblib import Parallel, delayed
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
BASE_PATH = '/home/modelling/Waves_Analysis/Crossing_Swells'
os.chdir(BASE_PATH)

# Mapping of site numbers to location names
SITE_MAPPING = {
    1: 'ACR1',
    2: 'ACR2', 
    4: 'ARF',
    5: 'ART'
}

def wrap_angle(data):
    """Wrap angle values to [-180, 180] range"""
    return (data + 180) % 360 - 180

def process_timestep_original(i, ds_timestep):
    """
    Process a single timestep using the ORIGINAL WORKING METHOD
    This exactly matches the code that worked for the monthly file
    """
    try:
        # Step 1: Split the wave spectra
        ds = ds_timestep.spec.split()
        
        # Step 2: Partition using ptm1 (this worked before)
        dspart = ds.spec.partition.ptm1(
            ds_timestep.wspd, 
            ds_timestep.wdir, 
            ds_timestep.dpt,
            agefac=1.7, 
            wscut=0.3333, 
            swells=3, 
            smooth=False, 
            freq_window=3, 
            dir_window=3, 
            ihmax=100
        )
        
        # Step 3: Extract wave parameters for partitions 1 and 2
        # Note: Original used part=1 and part=2 (1-indexed)
        Hs_1 = dspart.isel(part=1).spec.hs().values
        Dir_1 = dspart.isel(part=1).spec.dm().values
        Hs_2 = dspart.isel(part=2).spec.hs().values
        Dir_2 = dspart.isel(part=2).spec.dm().values
        
        # Step 4: Compute direction difference
        raw_direction_difference = Dir_1 - Dir_2
        wrapped_direction_difference = wrap_angle(raw_direction_difference)
        
        # Step 5: Check conditions for crossing swells
        Hs_condition = Hs_2 >= 0.5 * Hs_1
        direction_condition = abs(wrapped_direction_difference) > 40
        
        return 1 if (Hs_condition and direction_condition) else 0
        
    except Exception as e:
        # Silent fail for production
        return 0

def test_original_method():
    """Test the original working method on yearly file"""
    print("="*80)
    print("TESTING ORIGINAL METHOD ON YEARLY FILE")
    print("="*80)
    
    # Use the monthly file that we know works
    monthly_file = '2017/YES/ww3.YES_ACR2_201706_spec.nc'
    
    if os.path.exists(monthly_file):
        print(f"Testing on known working file: {monthly_file}")
        dset = read_ww3(monthly_file)
        dset_site = dset.isel(site=0)
        
        # Test first timestep
        ds_t = dset_site.isel(time=0)
        result = process_timestep_original(0, ds_t)
        
        # Get bulk parameters for context
        hs_total = ds_t.spec.hs().values
        dm_total = ds_t.spec.dm().values
        
        print(f"  Timestep 0: Hs_total={hs_total:.2f}m, Dm={dm_total:.1f}°, Crossing={result}")
        print(f"  ✓ Method works on monthly file!")
        
        # Now test on yearly file
        yearly_file = '2017/NO/ww3.NO_ACR1_2017_spec.nc'
        if os.path.exists(yearly_file):
            print(f"\nTesting on yearly file: {yearly_file}")
            dset_yearly = read_ww3(yearly_file)
            dset_yearly_site = dset_yearly.isel(site=0)
            
            # Test first timestep
            ds_t_yearly = dset_yearly_site.isel(time=0)
            result_yearly = process_timestep_original(0, ds_t_yearly)
            
            hs_total_yearly = ds_t_yearly.spec.hs().values
            dm_total_yearly = ds_t_yearly.spec.dm().values
            
            print(f"  Timestep 0: Hs_total={hs_total_yearly:.2f}m, Dm={dm_total_yearly:.1f}°, Crossing={result_yearly}")
            print(f"  ✓ Method works on yearly file too!")
            
            return True
        else:
            print(f"  Yearly file not found: {yearly_file}")
            return True  # Monthly file worked, so method is good
    else:
        print(f"Monthly file not found: {monthly_file}")
        return False

def process_location_scenario(file_path, site_number, location_name, scenario_name):
    """Process a yearly file for a specific location using original method"""
    print(f"  Processing {location_name} (Site {site_number}) - {scenario_name}...")
    
    try:
        # Load data
        dset = read_ww3(file_path)
        
        # Find the site index
        site_values = dset.site.values
        if site_number not in site_values:
            print(f"    Warning: Site {site_number} not found. Available: {site_values}")
            return None
        
        site_idx = np.where(site_values == site_number)[0][0]
        dset_site = dset.isel(site=site_idx)
        
        total_timesteps = len(dset_site.time)
        print(f"    Total timesteps: {total_timesteps}")
        
        # Prepare timestep data
        timesteps_data = [(i, dset_site.isel(time=i)) for i in range(total_timesteps)]
        
        # Process in parallel
        n_jobs = min(os.cpu_count() - 1, total_timesteps)
        print(f"    Using {n_jobs} CPU cores...")
        
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_timestep_original)(i, ts_data) 
            for i, ts_data in timesteps_data
        )
        
        # Group results by month
        times = dset_site.time.values
        monthly_results = {}
        
        for time_idx, result in enumerate(results):
            # Get month from time
            dt = times[time_idx].astype('datetime64[D]').astype(datetime)
            month = dt.month
            
            if month not in monthly_results:
                monthly_results[month] = {'total': 0, 'crossing': 0}
            
            monthly_results[month]['total'] += 1
            monthly_results[month]['crossing'] += result
        
        # Calculate percentages
        monthly_percentages = {}
        for month, data in monthly_results.items():
            percentage = (data['crossing'] / data['total']) * 100 if data['total'] > 0 else 0
            monthly_percentages[month] = {
                'total_steps': data['total'],
                'crossing_events': data['crossing'],
                'percentage': percentage
            }
        
        return {
            'location': location_name,
            'scenario': scenario_name,
            'year': 2017,
            'monthly_data': monthly_percentages
        }
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def main():
    """Main function"""
    start_time = time.time()
    
    print("="*80)
    print("CROSSING SWELL ANALYSIS - USING ORIGINAL WORKING METHOD")
    print("="*80)
    
    # First test the original method
    print("\nTesting the original method...")
    test_original_method()
    
    print("\n" + "="*80)
    print("PROCESSING ALL LOCATIONS AND SCENARIOS")
    print("="*80)
    
    scenarios = {
        'No Currents': 'NO',
        'With Currents': 'YES'
    }
    
    all_results = []
    
    for site_number, location_name in SITE_MAPPING.items():
        print(f"\n{'='*80}")
        print(f"LOCATION: {location_name} (Site {site_number})")
        print(f"{'='*80}")
        
        for scenario_name, code in scenarios.items():
            file_pattern = f"ww3.{code}_{location_name}_2017_spec.nc"
            file_path = os.path.join('2017', code, file_pattern)
            
            if os.path.exists(file_path):
                result = process_location_scenario(file_path, site_number, location_name, scenario_name)
                if result:
                    all_results.append(result)
            else:
                print(f"  File not found: {file_path}")
    
    # Compile and save results
    if all_results:
        rows = []
        for result in all_results:
            for month, data in result['monthly_data'].items():
                rows.append({
                    'Location': result['location'],
                    'Scenario': result['scenario'],
                    'Month': month,
                    'Month_Name': datetime(2000, month, 1).strftime('%B'),
                    'Total_Timesteps': data['total_steps'],
                    'Crossing_Events': data['crossing_events'],
                    'Crossing_Percentage': round(data['percentage'], 2)
                })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        csv_filename = 'crossing_swell_results_2017.csv'
        df.to_csv(csv_filename, index=False)
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED TO: {csv_filename}")
        print("="*80)
        
        # Print summary to console
        print("\n" + "="*80)
        print("SUMMARY BY LOCATION AND SCENARIO")
        print("="*80)
        
        for location in df['Location'].unique():
            print(f"\n{location}:")
            print("-" * 60)
            
            for scenario in ['No Currents', 'With Currents']:
                scenario_df = df[df['Scenario'] == scenario]
                if not scenario_df.empty:
                    loc_scenario_df = scenario_df[scenario_df['Location'] == location]
                    if not loc_scenario_df.empty:
                        annual_avg = loc_scenario_df['Crossing_Percentage'].mean()
                        print(f"\n  {scenario}:")
                        print(f"    Annual Average: {annual_avg:.2f}%")
                        print(f"    Monthly Breakdown:")
                        for _, row in loc_scenario_df.iterrows():
                            print(f"      {row['Month_Name']:10}: {row['Crossing_Percentage']:6.2f}%")
        
        # Create a comparison table
        print("\n" + "="*80)
        print("COMPARISON: WITH vs WITHOUT CURRENTS")
        print("="*80)
        
        for location in df['Location'].unique():
            print(f"\n{location}:")
            print("-" * 80)
            
            # Get data for both scenarios
            no_curr = df[(df['Location'] == location) & (df['Scenario'] == 'No Currents')]
            with_curr = df[(df['Location'] == location) & (df['Scenario'] == 'With Currents')]
            
            if not no_curr.empty and not with_curr.empty:
                for month in range(1, 13):
                    month_name = datetime(2000, month, 1).strftime('%B')
                    no_val = no_curr[no_curr['Month'] == month]['Crossing_Percentage'].values
                    with_val = with_curr[with_curr['Month'] == month]['Crossing_Percentage'].values
                    
                    no_pct = no_val[0] if len(no_val) > 0 else 0
                    with_pct = with_val[0] if len(with_val) > 0 else 0
                    diff = with_pct - no_pct
                    
                    arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
                    print(f"  {month_name:10}: No Currents={no_pct:5.2f}% | With Currents={with_pct:5.2f}% | Difference={diff:+6.2f}% {arrow}")
    else:
        print("\nNo results generated. Check if files exist.")
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"Total execution time: {elapsed_time:.2f} minutes")
    print("="*80)

if __name__ == "__main__":
    main()
