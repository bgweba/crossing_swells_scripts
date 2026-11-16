#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from wavespectra import read_ww3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Constants
BASE_PATH = '/home/modelling/Waves_Analysis/Crossing_Swells'
os.chdir(BASE_PATH)

def wrap_angle(data):
    """Wrap angle values to [-180, 180] range"""
    return (data + 180) % 360 - 180

def process_timestep_original(i, ds_timestep, verbose=False):
    """
    Process a single timestep using the ORIGINAL WORKING METHOD
    """
    try:
        # Step 1: Split the wave spectra
        ds = ds_timestep.spec.split()
        
        # Step 2: Partition using ptm1
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
        
        result = 1 if (Hs_condition and direction_condition) else 0
        
        if verbose:
            print(f"    Timestep {i}: Hs1={Hs_1:.2f}, Hs2={Hs_2:.2f}, Dir1={Dir_1:.1f}, Dir2={Dir_2:.1f}, Diff={wrapped_direction_difference:.1f}, Crossing={result}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"    Timestep {i} error: {type(e).__name__}: {str(e)[:100]}")
        return 0

def process_year(year, location, scenario):
    """
    Process a single year for a specific location and scenario
    
    Parameters:
    -----------
    year : int
        Year to process (2016, 2017, or 2018)
    location : str
        Location name (e.g., 'ART')
    scenario : str
        'NO' for no currents, 'YES' for with currents
    
    Returns:
    --------
    dict : Monthly results for the year
    """
    scenario_name = 'No Currents' if scenario == 'NO' else 'With Currents'
    file_path = os.path.join(str(year), scenario, f'ww3.{scenario}_{location}_{year}_spec.nc')
    
    if not os.path.exists(file_path):
        print(f"  Warning: File not found - {file_path}")
        return None
    
    print(f"\n  Processing {location} - {scenario_name} ({year})...")
    
    try:
        # Load data
        dset = read_ww3(file_path)
        
        # Select site (ART is site 1, index 0)
        dset_site = dset.isel(site=0)
        
        total_timesteps = len(dset_site.time)
        print(f"    Total timesteps: {total_timesteps}")
        
        # Process ALL timesteps sequentially
        all_results = []
        start_time = time.time()
        
        for i in range(total_timesteps):
            ds_t = dset_site.isel(time=i)
            result = process_timestep_original(i, ds_t, verbose=False)
            all_results.append(result)
            
            # Print progress every 500 timesteps
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{total_timesteps} timesteps...")
        
        elapsed = time.time() - start_time
        print(f"    Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        
        # Group by month
        times = dset_site.time.values
        monthly_results = {}
        
        for time_idx, result in enumerate(all_results):
            dt = times[time_idx].astype('datetime64[D]').astype(datetime)
            month = dt.month
            
            if month not in monthly_results:
                monthly_results[month] = {'total': 0, 'crossing': 0, 'percentage': 0}
            
            monthly_results[month]['total'] += 1
            monthly_results[month]['crossing'] += result
        
        # Calculate percentages
        for month in monthly_results:
            monthly_results[month]['percentage'] = (monthly_results[month]['crossing'] / monthly_results[month]['total']) * 100
        
        # Ensure all months have entries
        for month in range(1, 13):
            if month not in monthly_results:
                monthly_results[month] = {'total': 0, 'crossing': 0, 'percentage': 0}
        
        return monthly_results
        
    except Exception as e:
        print(f"    ERROR processing {year}: {e}")
        return None

def process_multi_year(location='ART', years=[2016, 2017, 2018]):
    """
    Process multiple years for a specific location
    
    Parameters:
    -----------
    location : str
        Location name (default: 'ART')
    years : list
        List of years to process (default: [2016, 2017, 2018])
    """
    print("="*80)
    print(f"CROSSING SWELL ANALYSIS - {location} (2016-2018)")
    print("="*80)
    
    all_yearly_results = {}
    
    for year in years:
        print(f"\n{'='*80}")
        print(f"YEAR: {year}")
        print(f"{'='*80}")
        
        yearly_data = {'NO': None, 'YES': None}
        
        # Process without currents
        yearly_data['NO'] = process_year(year, location, 'NO')
        
        # Process with currents
        yearly_data['YES'] = process_year(year, location, 'YES')
        
        all_yearly_results[year] = yearly_data
    
    return all_yearly_results

def save_multi_year_results(all_results, location='ART'):
    """Save multi-year results to CSV and text files"""
    
    # Create comprehensive DataFrame
    rows = []
    for year in all_results.keys():
        for month in range(1, 13):
            month_name = datetime(2000, month, 1).strftime('%B')
            
            no_data = all_results[year]['NO'].get(month, {'total': 0, 'crossing': 0, 'percentage': 0})
            yes_data = all_results[year]['YES'].get(month, {'total': 0, 'crossing': 0, 'percentage': 0})
            
            rows.append({
                'Year': year,
                'Month': month,
                'Month_Name': month_name,
                'No_Currents_Total': no_data['total'],
                'No_Currents_Crossing': no_data['crossing'],
                'No_Currents_Percentage': round(no_data['percentage'], 2),
                'With_Currents_Total': yes_data['total'],
                'With_Currents_Crossing': yes_data['crossing'],
                'With_Currents_Percentage': round(yes_data['percentage'], 2),
                'Difference': round(yes_data['percentage'] - no_data['percentage'], 2)
            })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    csv_filename = f'{location}_crossing_swell_2016_2018.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\n✓ Detailed results saved to: {csv_filename}")
    
    # Create summary by year
    yearly_summary = df.groupby('Year').agg({
        'No_Currents_Percentage': 'mean',
        'With_Currents_Percentage': 'mean',
        'Difference': 'mean'
    }).round(2)
    
    yearly_summary.to_csv(f'{location}_yearly_summary_2016_2018.csv')
    print(f"✓ Yearly summary saved to: {location}_yearly_summary_2016_2018.csv")
    
    # Create monthly average across years
    monthly_avg = df.groupby('Month_Name').agg({
        'No_Currents_Percentage': 'mean',
        'With_Currents_Percentage': 'mean',
        'Difference': 'mean'
    }).round(2)
    
    monthly_avg.to_csv(f'{location}_monthly_average_2016_2018.csv')
    print(f"✓ Monthly average (2016-2018) saved to: {location}_monthly_average_2016_2018.csv")
    
    # Save formatted text report
    txt_filename = f'{location}_crossing_swell_report_2016_2018.txt'
    with open(txt_filename, 'w') as f:
        f.write("="*100 + "\n")
        f.write(f"{location} CROSSING SWELL ANALYSIS - 2016 to 2018\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*100 + "\n\n")
        
        # Year-by-year breakdown
        for year in sorted(all_results.keys()):
            f.write(f"\n{'='*100}\n")
            f.write(f"YEAR {year}\n")
            f.write(f"{'='*100}\n\n")
            
            year_df = df[df['Year'] == year]
            
            f.write(f"{'Month':<12} {'No Currents':<25} {'With Currents':<25} {'Difference':<15} {'Impact':<10}\n")
            f.write("-"*100 + "\n")
            
            for _, row in year_df.iterrows():
                diff = row['Difference']
                if diff > 0:
                    impact = "INCREASE ↑"
                elif diff < 0:
                    impact = "DECREASE ↓"
                else:
                    impact = "No change →"
                
                f.write(f"{row['Month_Name']:<12} "
                       f"{row['No_Currents_Percentage']:6.2f}% ({row['No_Currents_Crossing']}/{row['No_Currents_Total']})   "
                       f"{row['With_Currents_Percentage']:6.2f}% ({row['With_Currents_Crossing']}/{row['With_Currents_Total']})   "
                       f"{diff:+6.2f}%     {impact}\n")
            
            # Yearly average
            year_avg_no = year_df['No_Currents_Percentage'].mean()
            year_avg_yes = year_df['With_Currents_Percentage'].mean()
            year_diff = year_avg_yes - year_avg_no
            
            f.write(f"\n{'YEARLY AVERAGE':<12}: "
                   f"No Curr={year_avg_no:6.2f}% | "
                   f"With Curr={year_avg_yes:6.2f}% | "
                   f"Difference={year_diff:+6.2f}%\n")
        
        # Multi-year summary
        f.write(f"\n{'='*100}\n")
        f.write("MULTI-YEAR SUMMARY (2016-2018)\n")
        f.write(f"{'='*100}\n\n")
        
        f.write("ANNUAL AVERAGES:\n")
        f.write("-"*50 + "\n")
        for year in sorted(all_results.keys()):
            year_df = df[df['Year'] == year]
            annual_no = year_df['No_Currents_Percentage'].mean()
            annual_yes = year_df['With_Currents_Percentage'].mean()
            annual_diff = annual_yes - annual_no
            f.write(f"{year}: No Curr={annual_no:6.2f}% | With Curr={annual_yes:6.2f}% | Diff={annual_diff:+6.2f}%\n")
        
        f.write("\nMONTHLY AVERAGES (2016-2018):\n")
        f.write("-"*80 + "\n")
        for month_name in monthly_avg.index:
            no_avg = monthly_avg.loc[month_name, 'No_Currents_Percentage']
            yes_avg = monthly_avg.loc[month_name, 'With_Currents_Percentage']
            diff_avg = monthly_avg.loc[month_name, 'Difference']
            arrow = "↑" if diff_avg > 0 else ("↓" if diff_avg < 0 else "→")
            f.write(f"{month_name:<12}: No Curr={no_avg:6.2f}% | With Curr={yes_avg:6.2f}% | Diff={diff_avg:+6.2f}% {arrow}\n")
        
        # Overall 3-year average
        overall_no = df['No_Currents_Percentage'].mean()
        overall_yes = df['With_Currents_Percentage'].mean()
        overall_diff = overall_yes - overall_no
        
        f.write(f"\n{'='*100}\n")
        f.write("OVERALL 3-YEAR AVERAGE (2016-2018):\n")
        f.write(f"{'='*100}\n")
        f.write(f"No Currents:  {overall_no:.2f}%\n")
        f.write(f"With Currents: {overall_yes:.2f}%\n")
        f.write(f"Difference:    {overall_diff:+.2f}%\n")
        
        if overall_diff > 0:
            f.write(f"\n✓ CONCLUSION: Currents INCREASE crossing swell events by {overall_diff:.2f}% on average over 2016-2018\n")
        elif overall_diff < 0:
            f.write(f"\n✓ CONCLUSION: Currents DECREASE crossing swell events by {abs(overall_diff):.2f}% on average over 2016-2018\n")
        else:
            f.write(f"\n✓ CONCLUSION: Currents have NO EFFECT on crossing swell events over 2016-2018\n")
        
        # Yearly trend
        f.write(f"\nYEARLY TREND:\n")
        f.write("-"*50 + "\n")
        yearly_diffs = []
        for year in sorted(all_results.keys()):
            year_df = df[df['Year'] == year]
            year_diff = year_df['Difference'].mean()
            yearly_diffs.append(year_diff)
            f.write(f"{year}: {year_diff:+.2f}% difference\n")
        
        if len(yearly_diffs) >= 2:
            trend = yearly_diffs[-1] - yearly_diffs[0]
            if trend > 0:
                f.write(f"\n→ Increasing trend in current impact: +{trend:.2f}% over 3 years\n")
            elif trend < 0:
                f.write(f"\n→ Decreasing trend in current impact: {trend:.2f}% over 3 years\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"✓ Detailed report saved to: {txt_filename}")
    
    return df

def print_console_summary(all_results):
    """Print a formatted summary to console"""
    
    print("\n" + "="*80)
    print("MULTI-YEAR SUMMARY (2016-2018)")
    print("="*80)
    
    # Collect all data for DataFrame
    all_data = []
    for year in all_results.keys():
        for month in range(1, 13):
            month_name = datetime(2000, month, 1).strftime('%B')
            no_data = all_results[year]['NO'].get(month, {'percentage': 0})
            yes_data = all_results[year]['YES'].get(month, {'percentage': 0})
            
            all_data.append({
                'Year': year,
                'Month': month,
                'Month_Name': month_name,
                'No_Currents': no_data['percentage'],
                'With_Currents': yes_data['percentage']
            })
    
    df = pd.DataFrame(all_data)
    
    # Print annual averages
    print("\nANNUAL AVERAGES:")
    print("-" * 60)
    for year in sorted(all_results.keys()):
        year_df = df[df['Year'] == year]
        annual_no = year_df['No_Currents'].mean()
        annual_yes = year_df['With_Currents'].mean()
        annual_diff = annual_yes - annual_no
        arrow = "↑" if annual_diff > 0 else ("↓" if annual_diff < 0 else "→")
        print(f"{year}: No Curr={annual_no:6.2f}% | With Curr={annual_yes:6.2f}% | Diff={annual_diff:+6.2f}% {arrow}")
    
    # Print monthly averages
    print("\nMONTHLY AVERAGES (2016-2018):")
    print("-" * 80)
    monthly_avg = df.groupby('Month').agg({
        'No_Currents': 'mean',
        'With_Currents': 'mean'
    }).round(2)
    
    for month in range(1, 13):
        month_name = datetime(2000, month, 1).strftime('%B')
        no_avg = monthly_avg.loc[month, 'No_Currents']
        yes_avg = monthly_avg.loc[month, 'With_Currents']
        diff = yes_avg - no_avg
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
        
        # Determine impact description
        if diff > 5:
            impact = "Strong increase"
        elif diff > 0:
            impact = "Moderate increase"
        elif diff < -5:
            impact = "Strong decrease"
        elif diff < 0:
            impact = "Moderate decrease"
        else:
            impact = "Negligible change"
        
        print(f"{month_name:10}: No Curr={no_avg:6.2f}% | With Curr={yes_avg:6.2f}% | Diff={diff:+6.2f}% {arrow} | {impact}")
    
    # Overall 3-year average
    overall_no = df['No_Currents'].mean()
    overall_yes = df['With_Currents'].mean()
    overall_diff = overall_yes - overall_no
    
    print("\n" + "="*80)
    print("OVERALL 3-YEAR AVERAGE (2016-2018):")
    print("="*80)
    print(f"No Currents:  {overall_no:.2f}%")
    print(f"With Currents: {overall_yes:.2f}%")
    print(f"Difference:    {overall_diff:+.2f}%")
    
    if overall_diff > 0:
        print(f"\n✓ CONCLUSION: Currents INCREASE crossing swell events by {overall_diff:.2f}% on average over 2016-2018")
    elif overall_diff < 0:
        print(f"\n✓ CONCLUSION: Currents DECREASE crossing swell events by {abs(overall_diff):.2f}% on average over 2016-2018")
    else:
        print(f"\n✓ CONCLUSION: Currents have NO EFFECT on crossing swell events over 2016-2018")

def main():
    """Main function"""
    start_time = time.time()
    
    # Process ART for 2016-2018
    all_results = process_multi_year(location='ART', years=[2016, 2017, 2018])
    
    # Save results
    df = save_multi_year_results(all_results, location='ART')
    
    # Print console summary
    print_console_summary(all_results)
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\n{'='*80}")
    print(f"Total execution time: {elapsed_time:.2f} minutes")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
