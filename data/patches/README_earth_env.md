# earth_patches_env.csv: Earth patch environment, climate, soil and geology features

One row per `patch_id` from `earth_candidates.csv`. Each patch is 10 km x 10 km (half-width 5 km) centred on
`lat`/`lon`. The script is `tools/build_earth_env.py`, and it can be re-run. Global rasters are downloaded once to
`~/.cache/earth_analog_raw/{jaxa,soilgrids,permafrost}`. Point-API responses are cached as JSON in
`~/.cache/earth_analog_raw/{power,macrostrat}`, so a re-run only fetches what is missing.

The script rewrites the CSV atomically after each phase. The columns are always the final schema; anything not
finished yet is NaN. `earth_patches_env.status.json` records the phase, which sources are done, the resolution used
for each raster, the API progress, and the non-null coverage of each column.

Patch aggregation: raster columns are the mean of all pixels that intersect the 10 km box (the box always includes
the pixel that contains the point). Monthly products are aggregated over the 12 monthly normals after that.

__COVERAGE__

## Columns

### Temperature and environmental extremes
| column | unit | source / definition |
|---|---|---|
| `lst_day_max_k` | K | MODIS Terra LST **daytime** (MOD11C1 v061) monthly normals 2012-2021 (JAXA Earth COG `NASA.EOSDIS_Terra.MODIS_MOD11C1-LST.daytime.v061_global_monthly-normal`, DN x 0.02); max over the 12 months (hottest month) |
| `lst_day_mean_k` | K | mean over the valid months of daytime LST |
| `lst_night_min_k` | K | MODIS LST **nighttime** normals (`...nighttime.v061_global_monthly-normal`); min over months (coldest month) |
| `lst_night_mean_k` | K | mean over the valid months of nighttime LST |
| `lst_mean_k` | K | mean over months of (day+night)/2. A month with only one of day/night uses that value. Polar-night months have no daytime LST, so this is biased toward night values at high latitude |
| `lst_diurnal_range_k` | K | mean over months of (day - night), using months where both exist |
| `lst_annual_range_k` | K | max - min of the monthly (day+night)/2 |
| `lst_day_months_valid`, `lst_night_months_valid` | count | months with data (MODIS LST is clear-sky only, and there is no daytime LST during polar night) |
| `lst_res_deg` | deg | grid used: 0.05 (JAXA level 1, native) or 0.2 (level 0; interim fast pass only) |
| `power_t2m_c` | degC | NASA POWER climatology (point API, `community=RE`, MERRA-2 0.5 x 0.625 deg) annual mean 2 m air temperature |
| `power_t2m_min_c`, `power_t2m_max_c` | degC | annual mean of daily min / max T2M |
| `power_t2m_coldest_month_min_c` | degC | min over months of monthly-mean daily minimum (T2M_MIN) |
| `power_t2m_warmest_month_max_c` | degC | max over months of monthly-mean daily maximum (T2M_MAX) |
| `power_t2m_annual_range_c` | degC | warmest minus coldest monthly-mean T2M |
| `power_t2m_diurnal_range_c` | degC | T2M_RANGE annual |
| `power_frost_days` | days/yr | FROST_DAYS annual total |
| `power_rh2m_pct` | % | RH2M |
| `power_ws2m_ms` | m/s | WS2M (kept only as context; wind is not a Moon-analog criterion) |
| `power_ps_kpa` | kPa | surface pressure PS (works as an altitude proxy) |
| `power_precip_mm_yr` | mm/yr | PRECTOTCORR x 365.25 (MERRA-2; global, including the poles) |
| `power_sw_allsky_kwh_m2_d`, `power_sw_clrsky_kwh_m2_d` | kWh/m2/day | ALLSKY / CLRSKY_SFC_SW_DWN (CERES/SRB, 1 deg) |
| `power_uv_index` | - | ALLSKY_SFC_UV_INDEX annual |
| `power_albedo` | - | ALLSKY_SRF_ALB annual (months without sun are excluded by POWER) |
| `power_cell_lat`, `power_cell_lon` | deg | the 0.5 x 0.625 deg cell that was queried. Points are snapped to cell centres and deduplicated, so patches in the same cell share POWER values |

### Water, ice and permafrost
| column | unit | source / definition |
|---|---|---|
| `pf_frac_pct` | % | ESA CCI Permafrost v5.0, year 2023, permafrost fraction **PFR** (CryoGrid driven by MODIS LST, 0.01 deg). Northern Hemisphere file covers 25-85 N; Antarctica file covers 60-85 S, ice-free ground only. Downloaded anonymously over HTTP from CEDA (no login needed) |
| `pf_gst_c` | degC | CCI v5.0 2023 GTD: mean annual ground **surface** temperature (GST) |
| `pf_magt_2m_c` | degC | CCI v5.0 2023 GTD: mean annual ground temperature at **2 m** depth (T2m), the MAGT used here |
| `pf_magt_10m_c` | degC | CCI v5.0 2023 GTD at 10 m depth (T10m) |
| `pf_in_cci_domain` | 0/1 | the point falls inside a CCI grid's latitude range. A NaN inside the domain means water, ice sheet or glacier, which CCI does not model |
| `permafrost_proxy` | 0/1 | **proxy**: POWER annual T2M < -2 degC |
| `permafrost_flag` | 0/1 | CCI `pf_frac_pct >= 50` where CCI has a value, otherwise `permafrost_proxy` |
| `permafrost_src` | text | `esa_cci_pfr_2023` or `proxy_power_t2m_lt_-2C` |
| `soil_moist_mean`, `soil_moist_max` | vol % | JAXA AMSR2 L3 soil moisture content (SMC) daytime monthly normals 2012-2021 (`JAXA.G-Portal_GCOM-W.AMSR2_standard.L3-SMC.daytime.v3_global_monthly-normal`, DN x 0.1, 0.1 deg). Mean and max over months |
| `soil_moist_months_valid` | count | months with data (AMSR2 has no retrieval over snow, ice, dense forest or frozen ground) |
| `soil_moist_widened_25km` | 0/1 | 1 means the 10 km box was empty and the value is the mean within +/-25 km |
| `precip_mm_yr` | mm/yr | JAXA GSMaP Gauge v6 monthly-normal precipitation **rate** (mm/h, `JAXA.EORC_GSMaP_standard.Gauge.00Z-23Z.v6_monthly-normal`), converted as rate x 24 x days-in-month and summed over 12 months. 0.2 deg grid (level 0). **Covers 60 S - 60 N only**. NaN if fewer than 9 valid months |
| `precip_wettest_month_mm` | mm | wettest month |
| `precip_widened_25km` | 0/1 | value taken from a +/-25 km box (coast / grid edge) |
| `precip_best_mm_yr`, `precip_best_src` | mm/yr, text | GSMaP where available, otherwise POWER/MERRA-2 (`power_merra2`), e.g. at the poles |

### Biology and habitability (Earth "sterility")
| column | unit | source / definition |
|---|---|---|
| `ndvi_mean`, `ndvi_max` | - | JAXA JASMES Terra+Aqua MODIS NDVI monthly normals 2012-2021 (`JAXA.JASMES_Terra.MODIS-Aqua.MODIS_ndvi.v811_global_monthly-normal`, DN x 1e-4, range 0-1; negative NDVI is stored as 0 / no data) |
| `ndvi_res_deg` | deg | 0.05 (native, level 1) or 0.2 (interim) |

### Dust and aerosol (no wind metrics: the Moon has no air)
| column | unit | source / definition |
|---|---|---|
| `aot_mean`, `aot_max` | - | JAXA JASMES Terra+Aqua MODIS aerosol optical thickness at 500 nm, monthly normals (`JAXA.JASMES_Terra.MODIS-Aqua.MODIS_taua.v811_global_monthly-normal`, DN x 0.001). Uses the 0.2 deg level-0 grid: AOT is smooth, and this is ~12x less to download than 0.05 deg |
| `aot_widened_25km` | 0/1 | value taken from a +/-25 km box |

### Geology and surface materials
| column | unit | source / definition |
|---|---|---|
| `sg_sand_pct`, `sg_silt_pct`, `sg_clay_pct` | % (mass) | ISRIC SoilGrids 2.0, 0-5 cm mean (g/kg / 10). From the global aggregated GeoTIFFs `https://files.isric.org/soilgrids/latest/data_aggregated/{1000m,5000m}/<prop>/<prop>_0-5cm_mean_<res>.tif` (Interrupted Goode Homolosine; points are reprojected and a +/-5 km window is averaged). Pixels with value 0 (no soil: water, ice, bare-rock mask) are excluded |
| `sg_bdod_g_cm3` | g/cm3 | bulk density of the fine earth (cg/cm3 / 100) |
| `sg_cfvo_pct` | vol % | coarse fragments > 2 mm (cm3/dm3 / 10). Set to NaN where texture is NaN |
| `sg_res_m` | m | 1000 or 5000: which aggregate was sampled |
| `mstrat_unit` | text | Macrostrat `geologic_units/map` at the point: name of the unit from the finest map scale available (large > medium > small > tiny) |
| `mstrat_lith` | text | lithology string of that unit (falls back to the next scale if empty) |
| `mstrat_age` | text | `b_int_name-t_int_name` (or `best_int_name`) |
| `mstrat_descrip` | text | `descrip` + `comments` of the finest unit that has any. Kept as text for the geology text-embedding model |
| `mstrat_scale` | text | map scale of the chosen unit |
| `mstrat_t_age_ma`, `mstrat_b_age_ma` | Ma | top and bottom age of the chosen unit |
| `mstrat_text_all` | text | de-duplicated name; lith; descrip; comments; interval across **all** map scales at the point (max 1500 chars), for text embedding |

## Coverage gaps and caveats
* **Antarctica / ice sheets.** SoilGrids has no data (NaN). AMSR2 soil moisture is mostly empty. CCI permafrost
  only models ice-free ground, so ice-sheet patches use `permafrost_proxy`. GSMaP stops at 60 S, so use
  `precip_best_mm_yr`, which falls back to POWER. MODIS LST and NDVI are available.
* **> 60 N / 60 S precipitation.** GSMaP has no data there, so `precip_best_mm_yr` uses POWER/MERRA-2.
* **> 85 N** is outside the CCI Northern Hemisphere grid, so the proxy is used.
* **POWER** is a 0.5 x 0.625 deg reanalysis grid (solar 1 deg), so it is a regional, not a patch-scale, climate. It is
  queried once per unique cell.
* **Macrostrat** coverage outside North America is mostly the global "tiny" or "small" scale map, so descriptions
  there can be generic (e.g. "Cenozoic sedimentary rocks"). Ice-covered points return "Phanerozoic ice".
* **Monthly normals** cover 2012-2021 for every JAXA product.
* Values are rounded to 4 decimals.

## Source URLs
* JAXA Earth COGs: `https://s3.ap-northeast-1.wasabisys.com/je-pds/cog/v1/<collection>/<MM>/<level>/<lonband>/<tile>.tif`.
  Metadata such as units, DN scale and nodata comes from the STAC items (`.../<MM>/<level>/<lonband>/<latband>.json`)
* NASA POWER: `https://power.larc.nasa.gov/api/temporal/climatology/point`
* ESA CCI Permafrost v5.0 (CEDA, anonymous): `https://dap.ceda.ac.uk/neodc/esacci/permafrost/data/{permafrost_extent,ground_temperature}/L4/area4/pp/v05.0/{northern_hemisphere,Antarctica}/2023/`
* SoilGrids 2.0: `https://files.isric.org/soilgrids/latest/data_aggregated/`
* Macrostrat: `https://macrostrat.org/api/v2/geologic_units/map?lat=..&lng=..`
