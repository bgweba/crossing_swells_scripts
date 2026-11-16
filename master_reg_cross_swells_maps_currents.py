#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:21:10 2024
@author: sawsmarine
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
warnings.filterwarnings('ignore')
import cmocean  # Colormaps for oceanography
import os
# ================================================================================================================
# Define paths
path = '/media/bafana/GWEBA_SAWS/WW3_PTM1/'
os.chdir(path)
# ================================================================================================================

# Load datasets
ds_no = xr.open_dataset('./8km_output/NO/ww3.8km_NO_PTM1_201712.nc', chunks={'time': 10})
ds_yes = xr.open_dataset('./8km_output/YES/ww3.8km_YES_PTM1_201712.nc', chunks={'time': 10})
ds_currents = xr.open_dataset('./8km_output/CUR/ww3.8km_YES_201712.nc', chunks={'time': 10})  # Load current data
# ================================================================================================================

# Function to calculate % occurrence of crossing swells
def calculate_crossing_swells(ds):
    Hs_swl1 = ds['phs1']; Hs_swl2 = ds['phs2'];
    wl_swl1 = ds['plp1']; wl_swl2 = ds['plp2'];
    dir_swl1 = ds['pdir1']; dir_swl2 = ds['pdir2'];
    
    Hs_cond2 = Hs_swl2.values > 0.5 * Hs_swl1.values;
    Wl_cond = (wl_swl1 > 180) | (wl_swl2 > 180);
    
    def wrap_angle(data):
        return (data + 180) % 360 - 180

    angle1 = wrap_angle(dir_swl1 - dir_swl2);
    Dir_cond = (abs(angle1) > 45);
    cond_all2 = Hs_cond2 & Wl_cond & Dir_cond;

    ncounts2 = cond_all2.sum(dim="time");
    pp_cross_seas2 = ncounts2 / ds['time'].size * 100;
    return pp_cross_seas2
# ================================================================================================================

# Calculate crossing swells for both datasets
pp_cross_seas_no = calculate_crossing_swells(ds_no);
pp_cross_seas_yes = calculate_crossing_swells(ds_yes);

# Calculate monthly average current speed
u_cur = ds_currents['ucur'].mean(dim='time');
v_cur = ds_currents['vcur'].mean(dim='time');
ws_cur = np.sqrt(u_cur**2 + v_cur**2); # Monthly average current speed

#simulation_period = "January 2017";
# ================================================================================================================
# ================================================================================================================

# Plotting
size = (22, 8)
fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=size, subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'width_ratios': [1, 1, 1]})
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=size, subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'width_ratios': [1, 1, 1]})

# Define discrete color levels
levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

# Plot without currents
p1 = ax1.contourf(ds_no['longitude'], ds_no['latitude'], pp_cross_seas_no, levels=levels, cmap=plt.cm.jet, extend='both', transform=ccrs.PlateCarree())

# Add labels for each subplot in the correct order
ax1.text(18.75, -31.0, '(a)', style='italic', fontsize=15, bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 6})


# Plot with currents
p2 = ax2.contourf(ds_yes['longitude'], ds_yes['latitude'], pp_cross_seas_yes, levels=levels, cmap=plt.cm.jet, extend='both', transform=ccrs.PlateCarree())
#ax2.set_title('Crossing Swells (With Currents)', fontsize=12)

ax2.text(18.75, -31.0, '(c)', style='italic', fontsize=15, bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 6})
# Plot current speed map in the center
p3 = ax3.contourf(ds_currents['longitude'], ds_currents['latitude'], ws_cur, cmap=cmocean.cm.speed, extend='both', transform=ccrs.PlateCarree())
#ax3.set_title('Monthly Average Current Speed with Direction', fontsize=12)

ax3.text(18.75, -31.0, '(b)', style='italic', fontsize=15, bbox={'facecolor': 'white', 'alpha': 0.6, 'pad': 6})

# Add quiver for current direction
quiver_density = 6; # Control density of quivers; lower is denser
ax3.quiver(ds_currents['longitude'][::quiver_density], ds_currents['latitude'][::quiver_density],
           u_cur[::quiver_density, ::quiver_density], v_cur[::quiver_density, ::quiver_density],
           transform=ccrs.PlateCarree(), scale=40, color='black')

# Add physical features
for ax in [ax1, ax2, ax3]:
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='beige'))
    ax.add_feature(cfeature.LAND); ax.add_feature(cfeature.OCEAN);
    ax.add_feature(cfeature.COASTLINE); ax.add_feature(cfeature.BORDERS, linestyle=':');
    ax.add_feature(cfeature.LAKES, alpha=0.5); ax.add_feature(cfeature.RIVERS);

# Add gridlines and point labels
x_points = [28.00, 25.00, 20.0113, 25.25];
y_points = [-33.50, -35.00, -38.00, -37.1907];
labels = ['ACR1', 'ACR2', 'ARF', 'ART'];

# Configure gridlines for each axis
for ax in [ax1, ax2, ax3]:
    ax.plot(x_points, y_points, 'r*', ms=10, zorder=10)
    for i, label in enumerate(labels):
        ax.annotate(label, (x_points[i], y_points[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=15, zorder=20,
                    color='blue', bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
        
# Gridlines and label configurations
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.1, color='k', alpha=1, linestyle='--')
gl1.top_labels = False; gl1.right_labels = False;  # Left labels only for ax1
gl1.xformatter = LONGITUDE_FORMATTER; gl1.yformatter = LATITUDE_FORMATTER;
gl1.xlabel_style = {'size': 8}; gl1.ylabel_style = {'size': 8};

gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.1, color='k', alpha=1, linestyle='--')
gl2.top_labels = False; gl2.right_labels = False;  # Bottom labels for ax2 only
gl2.left_labels = False;  # No left labels for ax2
gl2.xformatter = LONGITUDE_FORMATTER; gl2.yformatter = LATITUDE_FORMATTER;
gl2.xlabel_style = {'size': 8}; gl2.ylabel_style = {'size': 8};

gl3 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.1, color='k', alpha=1, linestyle='--')
gl3.top_labels = False; gl3.left_labels = False; gl3.right_labels = False;  # No side labels for ax3
gl3.xformatter = LONGITUDE_FORMATTER; gl3.yformatter = LATITUDE_FORMATTER;
gl3.xlabel_style = {'size': 8}; gl3.ylabel_style = {'size': 8};

# Add colorbars
# Vertical color bar for % occurrence plots (adjusted to match map height)
cbar_ax1 = fig.add_axes([0.999, 0.11, 0.02, 0.75])  # Adjust position and height
cbar1 = fig.colorbar(p2, cax=cbar_ax1, orientation='vertical')
cbar1.ax.set_ylabel('$\%$ occurrence of crossing swells', fontsize=12)
cbar1.ax.tick_params(labelsize=8)

# Horizontal color bar for current speed plot, placed just below the map
cbar_ax2 = fig.add_axes([0.33, 0.08, 0.34, 0.02])  # Centered below ax3
cbar2 = fig.colorbar(p3, cax=cbar_ax2, orientation='horizontal')
cbar2.ax.set_xlabel('Current Speed (m/s)', fontsize=12)
cbar2.ax.tick_params(labelsize=8)

plt.tight_layout()
#plt.savefig('./images/reg_cross_swl/reg_cross_swells_201712_Hs_CUR_condition.png', bbox_inches='tight')
plt.show()
