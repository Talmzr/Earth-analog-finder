// ══════════════════════════════════════
// SIMILARITY ASSESSMENT DASHBOARD — "signal spectrum" layout
// Final calibrated scores from the scoring pipeline (ml/score_all.py, bundled by tools/build_globe.py).
// Scores run from -1 to +1: >= 0.5 sufficient, 0..0.5 partial, < 0 avoid.
// Overall match = composite: experiment-weighted mean of the categories that compare with this specific
// Moon target (terrain, geology, illumination, water/ice, + caves for the Tranquillitatis pit), capped by the
// ground-cover gate (0.8 - 2 x share of plants, open water or towns).
// ══════════════════════════════════════
document.head.insertAdjacentHTML('beforeend', `<style>
.sp-axis,.sp-main{grid-template-columns:240px 1fr 104px}
.sp-detail{grid-template-columns:240px 1fr}
.sp-name{letter-spacing:1.2px}
.sp-name>span{display:flex;flex-direction:column;gap:4px;min-width:0}
.sp-sub{font-style:normal;font-family:'Inter',sans-serif;font-size:10.5px;font-weight:400;letter-spacing:.2px;text-transform:none;color:#6f7f98;text-shadow:none;line-height:1.35}
.sp-val small{display:block;font-size:9.5px;font-weight:500;letter-spacing:1.3px;text-transform:uppercase;color:#8a9ab4;text-shadow:none;margin-top:2px}
.sp-row.sp-overall{border-bottom:1px solid rgba(255,255,255,.14)}
.sp-row.sp-overall .sp-name{font-size:13.5px}
.sp-sep{font-family:'Space Grotesk',sans-serif;font-size:9.5px;letter-spacing:1.8px;text-transform:uppercase;color:#4a5a74;padding:16px 0 4px;border-top:1px solid rgba(255,255,255,.06)}
.sd-method{font-size:11.5px;color:#55657e;line-height:1.6;margin:0 0 12px}
.sd-method b{color:#8a9ab4;font-weight:500}
.tile.cc-x,.panel.cc-x{--c:251,191,36}
.tile.cc-q,.panel.cc-q{--c:126,240,196}
.v-suff{--v:52,211,153}.v-part{--v:251,191,36}.v-avoid{--v:255,77,141}.v-na{--v:110,122,148}
.xr-sum{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 16px}
.xr-sum span{font-family:'Space Grotesk',sans-serif;font-size:12px;padding:5px 12px;border-radius:20px;border:1px solid rgba(var(--v),.45);color:rgb(var(--v));background:rgba(var(--v),.06)}
.xr-list{display:flex;flex-direction:column;gap:10px}
.xr{border:1px solid rgba(var(--v),.38);border-radius:10px;padding:13px 16px;background:rgba(var(--v),.025)}
.xr-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
.xr-name{font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:600;color:#dce8ff;line-height:1.4}
.xr-badge{flex-shrink:0;font-family:'Space Grotesk',sans-serif;font-size:10px;letter-spacing:1.4px;text-transform:uppercase;color:rgb(var(--v));border:1px solid rgba(var(--v),.55);border-radius:4px;padding:2px 8px;text-shadow:0 0 6px rgba(var(--v),.5)}
.xr-req{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}
.xr-chip{font-family:'Space Grotesk',sans-serif;font-size:11px;color:#b4c2d8;padding:3px 9px;border:1px solid rgba(255,255,255,.12);border-radius:6px;background:#000}
.xr-chip em{font-style:normal;color:#55657e;margin-left:4px}
.xr-chip.ok{border-color:rgba(52,211,153,.5);color:#7ef0c4}
.xr-chip.low{border-color:rgba(255,77,141,.5);color:#ff8db5}
.xr-chip.sup{border-style:dashed}
.xr-why{font-size:12px;color:#8a9ab4;margin-top:8px;line-height:1.55}
.xr-why b{color:#c8d4ea;font-weight:600}
.xr details{margin-top:8px}
.xr summary,.xr-nr summary{cursor:pointer;font-size:11.5px;color:#64b4ff;font-family:'Space Grotesk',sans-serif;letter-spacing:.3px}
.xr ul,.xr-nr ul{margin:6px 0 0 16px;padding:0;font-size:12px;color:#8a9ab4;line-height:1.6}
.xr-nr{margin-top:14px}
.nr-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-top:4px}
.nr-item{border:1px solid rgba(100,180,255,.2);border-radius:9px;padding:12px 14px;background:rgba(100,180,255,.02)}
.nr-item b{font-family:'Space Grotesk',sans-serif;font-size:12.5px;color:#dce8ff;display:block;margin-bottom:6px}
.nr-item p{font-size:11.5px;color:#8a9ab4;line-height:1.55;margin:3px 0}
.nr-item p i{font-style:normal;color:#55657e}
.nr-item .nr-how{color:#64b4ff}
.q-tbl td.v{width:auto}
.q-tbl td.p small{display:block;color:#55657e;font-size:10.5px;line-height:1.45;margin-top:3px}
.q-big{font-family:'Space Grotesk',sans-serif;font-size:13px;color:#dce8ff;line-height:1.6;margin:0 0 14px}
@media (max-width:860px){
  .sp-axis,.sp-main{grid-template-columns:120px 1fr 72px;gap:0 10px}
  .sp-detail{grid-template-columns:1fr}
  .xr-head{flex-direction:column;gap:6px}
}
</style>`);

const CAT_RGB = { terrain: '52,211,153', geology: '251,191,36', illumination: '246,196,83', water_ice: '96,165,250', operations: '100,180,255',
  caves: '255,159,67', thermal: '167,139,250', dust: '214,178,135', biology: '126,240,196', composite: '220,230,255' };
const CAT_BRIEF = {
  terrain: 'How the ground is shaped: steepness, height differences and ruggedness, measured the same way on both bodies (80 m elevation, 500 m slope baseline). Similar terrain means comparable conditions for landing, rover driving and walking.',
  geology: 'What the ground is made of: rock type and geologic setting. Lunar map units are compared with Earth bedrock descriptions by a fine-tuned text model, plus a rock-type match (anorthosite for the highlands, basalt for the maria).',
  illumination: 'How the Sun lights the surface: noon sun height and the share of daylight with the Sun grazing the horizon, compared in the Earth month whose sun is closest to the lunar one (the light-twin month).',
  water_ice: 'Whether water is held as ice in the ground, and how far that ice lies from landable ground. On Earth only frozen ground counts (an ice sheet is bulk ice, not ice-cemented soil), and liquid water counts against.',
  operations: 'How practical landing and surface operations are: the terrain and ice-access scores plus slope hazard, with penalties for forest, towns and open water.',
  caves: 'Whether the site has lava tubes like the one under the Tranquillitatis pit. Rule-based: documented tubes, tubes plausible (basaltic volcano), or no tubes.',
  thermal: 'How cold the site gets. No Earth site comes close to lunar temperatures, so this ranks how near Earth gets (closest approach), not true similarity.',
  dust: 'How bare and regolith-like the surface is, compared with the known Apollo regolith (bare, dry, fine). The same for every Moon target.',
  biology: 'How lifeless the site is (plant cover from vegetation greenness above the bare-soil level, rainfall), compared with the sterile Moon. The same for every Moon target.'
};
const CAT_ORDER = ['terrain', 'geology', 'illumination', 'water_ice', 'operations', 'caves', 'thermal', 'dust', 'biology'];
const SIM_CATS = CAT_ORDER.filter(k => CAT[k]).map(k => ({ key: k, title: CAT[k].label || k, rgb: CAT_RGB[k] || '100,180,255',
  brief: CAT_BRIEF[k] || CAT[k].how || '', how: CAT[k].how || '', cap: CAT[k].cap, inComp: !!CAT[k].in_composite, invariant: !!CAT[k].target_invariant }));
const catTitle = k => (CAT[k] && CAT[k].label) || k;

// Final scores for one pair: { terrain, geology, thermal, illumination, water_ice, dust, biology, caves, operations, composite }
function getSimilarityScores(earth, moon) {
  return SC.scores(moon.id, earth.idx);
}

// ── small helpers (null-safe) ──
const isNa = v => v === null || v === undefined || v === '' || (typeof v === 'number' && !isFinite(v));
const num = v => isNa(v) ? null : +v;
const yesNo = v => isNa(v) ? null : (+v ? 'yes' : 'no');
const twinOf = (t, r) => (t && r) ? SC.twinMonth(r.id, t.idx) : 0;
const twinSun = (t, r) => { const m = twinOf(t, r); return m ? earthSun(t, m) : { noon: null, graz: null }; };
const MOON_PERIOD = { 'pre-nectarian': 'older than ~3,920', 'nectarian': '~3,920–3,850', 'imbrian': '~3,850–3,200', 'eratosthenian': '~3,200–1,100', 'copernican': 'younger than ~1,100' };
const periodOf = r => { if (!r || !r.geo_age) return null; const a = String(r.geo_age).toLowerCase(); const k = Object.keys(MOON_PERIOD).find(p => a.startsWith(p)); return k ? MOON_PERIOD[k] : null; };
const isMare = r => !!r && (!!r.pit || (num(r.mix_mare) ?? 0) >= 50 || /mare/i.test(r.geo_name || ''));
const dryTarget = r => !!r && (!!r.pit || num(r.ice1m) === null || r.ice1m < 5);
const moonNoon = r => num(r.noon_max) ?? noonSun(r.lat, 1.54);
const cleanLith = t => (t.lith || '').replace(/[,;\s]+$/, '') || null;
const ANORTH = { 2: 'yes (main rock)', 1: 'minor component only', 0: 'none' };
const caveRule = v => isNa(v) ? null : v >= 0.5 ? 'documented lava tubes' : v >= 0.3 ? 'tubes plausible (basaltic volcanism)' : v < 0 ? 'no lava tubes (sedimentary or ice)' : 'no evidence either way';
const finesOf = t => (isNa(t.silt) || isNa(t.clay)) ? null : t.silt + t.clay;
const flatToIce = t => {
  const fr = num(EX(t, 'x_frozen_ground_frac'));
  if (fr !== null && fr >= 0.5) return '0 (ground ice)';
  return EX(t, 'trav_dist_gt10_flag') === 1 ? '> 10' : fmt(EX(t, 'trav_dist_flat_to_ice_km'), 1);
};
const landingIce = r => isNa(r.landing_to_ice_km) ? null : fmt(r.landing_to_ice_km, 2) + (isNa(r.landing_to_ice_min_km) ? '' : ` (min ${fmt(r.landing_to_ice_min_km, 2)})`);
const mixStr = r => {
  const p = [['crater', r.mix_crater], ['basin', r.mix_basin], ['plains', r.mix_plains], ['terra', r.mix_terra], ['mare', r.mix_mare]]
    .filter(([, v]) => !isNa(v) && +v > 0).map(([k, v]) => `${k} ${Math.round(v)}`);
  return p.length ? p.join(' · ') : null;
};
const settingOf = t => {
  if (t.source === 'volcano') return ((t.xinfo || '').split('|')[0] || 'volcano').trim();
  if (t.source === 'impact_structure') return 'impact structure' + (t.xinfo ? ' (' + t.xinfo.split('|')[0] + ')' : '');
  if (t.source === 'named_site') { const g = EX(t, 'site_tags'); return g ? String(g).split(';').map(s => s.replace(/_/g, ' ')).slice(0, 4).join(', ') : 'named analog site'; }
  return t.stratum ? t.stratum.replace(/_/g, ' ') + ' sample' : null;
};
const DUST_REF = META.dust_ref || {};

// One list per category, used by BOTH the expandable score rows (name + unit + meaning)
// and the Compared data tables (name + unit + Earth value + Moon value).
// [parameter, unit, Earth value (tile, target), Moon value (target, tile), what it means] — null renders as "—".
// Either argument after the first can be null when only one place is chosen.
const SIM_DATA = {
  terrain: [
    ['Mean slope', '°', t => fmt(t.slope, 1), r => fmt(r.slope, 1),
      'Average steepness at a 500 m baseline. Drives lander stability and rover traversability.'],
    ['90th-percentile slope', '°', t => fmt(EX(t, 'slope500_p90_deg'), 1), r => fmt(r.slope_p90, 1),
      'Steepness of the steepest tenth of the ground: the slopes a traverse must still handle.'],
    ['Ground steeper than 10°', '% of area', t => pct(num(EX(t, 'frac_slope500_gt10'))), r => fmt(r.steep10, 0),
      'Share of the area too steep for most landers and wheeled rovers.'],
    ['Ground steeper than 20°', '% of area', t => pct(num(EX(t, 'frac_slope500_gt20'))), r => fmt(r.steep20, 0),
      'Share of the area that is effectively no-go for rovers and crews.'],
    ['Surface roughness (ruggedness index)', 'm', t => fmt(t.rough, 0), r => fmt(r.tri, 0),
      'How much neighbouring ground heights vary. Rougher ground means more obstacles.'],
    ['Local relief', 'm', t => fmt(t.relief, 0), r => fmt(r.relief, 0),
      'Height range between high and low ground (95th–5th percentile). Reveals crater rims and massifs, and limits line of sight.'],
    ['Mean elevation', 'm', t => fmt(t.elev, 0), r => fmt(r.elev, 0),
      "Average height above each body's reference surface. Context only, since the Earth and Moon datums differ."]
  ],
  geology: [
    ['Rock type / map unit', 'class', t => cleanLith(t), r => r.geo_name ? r.geo_name + (r.geo_unit ? ` (${r.geo_unit})` : '') : null,
      'Earth: first-listed bedrock of the Macrostrat map unit (named sites lead with curated rock tags). Moon: unit of the USGS Unified Geologic Map of the Moon.'],
    ['Geologic setting', 'text', t => settingOf(t), r => r.geo_desc || null,
      'Earth: catalog type of the site. Moon: description of the mapped unit.'],
    ['Geologic age', 'period', t => t.geo_age || null, r => r.geo_age || null,
      'When the surface formed. Old, little-weathered surfaces keep their impact and volcanic record.'],
    ['Age range', 'million years', t => { const a = EX(t, 'mstrat_t_age_ma'), b = EX(t, 'mstrat_b_age_ma'); return isNa(a) || isNa(b) ? null : `${Math.round(a).toLocaleString('en-US')}–${Math.round(b).toLocaleString('en-US')}`; },
      r => periodOf(r),
      'Earth: youngest–oldest age of the mapped unit. Moon: span of the lunar period on the standard lunar time scale.'],
    ['Anorthosite (lunar highland rock)', 'flag', t => { const a = num(EX(t, 'anorth')); return a === null ? null : (ANORTH[a] ?? null); },
      r => isMare(r) ? 'no (mare basalt)' : 'yes (anorthositic highland crust)',
      'Counts only when it is the first-listed rock, in the site name or a curated site tag. A minor component caps the geology score at +0.30.'],
    ['Surface mix', '% of area', t => null, r => mixStr(r),
      'Share of crater, basin, plains, terra and mare units in the lunar target area.']
  ],
  illumination: [
    ['Light-twin month', 'month', (t, r) => { if (!r) return null; const m = twinOf(t, r); return m ? MONTHS[m] : 'none'; }, r => null,
      'The Earth month whose noon sun is closest to the lunar target. The sun terms are compared in that month.'],
    ['Noon sun in the light-twin month', '° above horizon', (t, r) => fmt(twinSun(t, r).noon, 1), r => fmt(r.noon_max, 1),
      'Earth: noon sun height in the light-twin month. Moon: highest noon sun at the target. At the lunar pole it stays within a few degrees of the horizon.'],
    ['Grazing sun (< 12°) in that month', '% of daylight', (t, r) => pct(twinSun(t, r).graz), r => fmt(r.grazing, 0),
      'Share of daylight with the Sun less than 12° above the horizon: long shadows, glare and low solar-panel angles.'],
    ['Highest noon sun of the year', '° above horizon', t => fmt(EX(t, 'x_noon_sun_max_deg'), 1), r => fmt(r.noon_max, 1),
      'Context: over a full year Earth sites see a much higher sun than any lunar polar target.'],
    ['Polar night', '% of year', t => pct(num(EX(t, 'x_polar_night_frac'))), r => fmt(r.polar_night, 0),
      'Share of the year with no sunrise. Sets battery and heater demands.'],
    ['Average sunlight', '% of time', t => null, r => fmt(r.sun, 0),
      'Share of time the Sun is visible, including terrain shadowing (LOLA, Moon). Drives solar power.'],
    ['Best-lit spot', '% of time', t => null, r => fmt(r.sun_max, 0),
      'Sunlight share at the best-lit point of the target, where solar panels would go.'],
    ['Area lit more than half the time', '% of area', t => null, r => fmt(r.sunlit50, 0),
      'Share of the target that is sunlit more than 50% of the time.']
  ],
  water_ice: [
    ['Frozen ground', 'Earth: % frozen · Moon: % ice stable ≤ 1 m', t => pct(num(EX(t, 'x_frozen_ground_frac'))),
      r => !isNa(r.ice1m) ? fmt(r.ice1m, 0) : r.pit ? '0 (dry mare)' : null,
      'Earth: share of permanently frozen ground. Moon: share of the area where ice is stable within 1 m of the surface (Diviner).'],
    ['Ice stable at any depth', '% of area', t => null, r => fmt(r.ice_stable, 0),
      'Share of the lunar target where buried ice would survive at some depth.'],
    ['Surface ice', '% of area', t => pct(num(EX(t, 'x_surface_ice_frac')), 1), r => fmt(r.surf_ice, 1),
      'Share of the area with exposed ice or snow (Earth) or ice stable right at the surface (Moon).'],
    ['Cold traps < 110 K', '% of area', t => null, r => fmt(r.coldtrap, 1),
      'Share of the area cold enough to hold water ice at the surface for billions of years (Moon).'],
    ['Permanent shadow', '% of area', t => null, r => fmt(r.psr, 1),
      'Share of the target never reached by sunlight (Moon), where the coldest ice traps sit.'],
    ['Depth to stable ice', 'm', t => null, r => fmt(r.ice_depth, 2),
      'How deep a drill must reach before ice stays frozen (Moon, median).'],
    ['Permafrost share', '% of area', t => fmt(EX(t, 'pf_frac_pct'), 0), r => null,
      'Modelled share of the area underlain by permafrost (ESA CCI, Earth).'],
    ['Ground temperature at 10 m', '°C', t => fmt(EX(t, 'pf_magt_10m_c'), 1), r => null,
      'Mean ground temperature at 10 m depth (Earth). Colder permafrost is harder and closer to lunar frozen regolith.'],
    ['Flat ground to ice', 'km', t => flatToIce(t), r => landingIce(r),
      'Earth: from flat bare ground to the nearest ice (0 where the ground itself is frozen). Moon: landing zone to permanent shadow (median, minimum in brackets). Only longer Earth distances lower the score.'],
    ['Precipitation', 'mm/yr', t => fmt(t.precip, 0), r => '0 (no liquid water)',
      'Rain and snow per year (Earth). At 500 mm/yr or more liquid water dominates and the water/ice score is capped at 0.'],
    ['Soil moisture', '%', t => fmt(t.moist, 1), r => '0',
      'Water held in the soil (Earth). Liquid water does not exist on the Moon, so lower is more lunar-like.']
  ],
  operations: [
    ['Ground steeper than 10°', '% of area', t => pct(num(EX(t, 'frac_slope500_gt10'))), r => fmt(r.haz_gt10, 0),
      'Earth: the 10 km tile. Moon: the target pixels themselves (landing or ice zone), the slope-hazard term.'],
    ['Ground steeper than 20°', '% of area', t => pct(num(EX(t, 'frac_slope500_gt20'))), r => fmt(r.haz_gt20, 0),
      'No-go ground for landers and rovers.'],
    ['90th-percentile slope', '°', t => fmt(EX(t, 'slope500_p90_deg'), 1), r => fmt(r.haz_p90, 1),
      'Slope of the steepest tenth of the ground.'],
    ['Flat bare ground', '% of area', t => pct(num(EX(t, 'trav_flat_bare_frac'))), r => null,
      'Share of the tile that is flat and bare: where a lander could set down and a rover start.'],
    ['Tree cover', '% of area', t => pct(num(EX(t, 'wc_frac_tree'))), r => '0',
      'Forest blocks landers and rovers: more than 30% lowers the operations score by 0.5.'],
    ['Built-up land', '% of area', t => pct(num(EX(t, 'wc_frac_built'))), r => '0',
      'Towns and roads: more than 10% lowers the operations score by 0.5.'],
    ['Open water', '% of area', t => pct(num(EX(t, 'wc_frac_water'))), r => '0',
      'A tile that is mostly lake or sea is capped at −0.8.'],
    ['Area compared', 'km²', t => '100 (10 × 10 km tile)', r => fmt(r.area_km2, 1),
      'Size of the ground behind each number.']
  ],
  caves: [
    ['Lava-tube evidence', 'rule level', t => caveRule(num(EX(t, 'cave_rule'))),
      r => r.pit ? 'LRO-imaged skylight into a probable lava tube' : 'no known pit (not scored)',
      'Documented tubes score +0.5, plausible (basaltic volcano or known tube region) +0.3, none 0, sedimentary or ice ground −0.5.'],
    ['Volcanic setting', 'class', t => t.source === 'volcano' ? settingOf(t) : (t.source === 'named_site' && /basalt|volcan|lava/i.test(String(EX(t, 'site_tags') || ''))) ? 'named volcanic site' : 'none',
      r => isMare(r) ? 'mare basalt flows' : 'highland crust',
      'Lava tubes form in fluid basaltic lava flows.'],
    ['Rock type', 'class', t => cleanLith(t), r => r.geo_name || null,
      'Bedrock (Earth) and mapped unit (Moon).']
  ],
  thermal: [
    ['Minimum temperature', 'K', t => fmt(t.tmin, 0), r => fmt(r.t_min, 0),
      'Coldest surface temperature at night or in shadow. Earth surfaces stay above ~180 K; lunar polar shadows reach 25–110 K.'],
    ['Mean temperature', 'K', t => fmt(EX(t, 'x_temp_mean_k'), 0), r => fmt(r.t_mean, 0),
      'Average surface temperature.'],
    ['Maximum temperature', 'K', t => fmt(t.tmax, 0), r => fmt(r.t_max, 0),
      'Hottest surface temperature reached in daylight.'],
    ['Temperature swing', 'K', t => (isNa(t.tmax) || isNa(t.tmin)) ? null : fmt(t.tmax - t.tmin, 0), r => fmt(r.t_swing, 0),
      'Day-to-night range that equipment and habitats must survive.'],
    ['Closest-approach rank', 'percentile', (t, r) => { if (!r) return null; const p = SC.thermalPct(r.id, t.idx); return p === null ? null : `colder than ${p}% of Earth sites`; }, r => null,
      'Rank of this Earth site among all 4,310 by how close it gets to the lunar target. The thermal score is a closest-approach ranking.']
  ],
  dust: [
    ['Bare ground', '% of area', t => pct(num(EX(t, 'bare_f_site'))), r => '100 (bare regolith)',
      'Bare or sparsely vegetated share (ESA WorldCover; in Antarctica the ice-free share; named regolith test spots count as at least 60%). Used in the score.'],
    ['Volcanic tephra / cinder', 'flag', t => yesNo(EX(t, 'flag_tephra')), r => 'glassy, fine regolith',
      'Volcanic ash and cinder are the closest natural Earth match to fine, glass-rich lunar soil. Used in the score.'],
    ['Fines (silt + clay)', '% of mass', t => fmt(finesOf(t), 0), r => fmt(DUST_REF.fines_pct, 0),
      'Context only: SoilGrids is 5 km modelled agricultural soil, so it was dropped from the score.'],
    ['Bulk density', 'g/cm³', t => fmt(t.bdod, 2), r => fmt(DUST_REF.bulk_density, 1),
      'Mass per volume of the surface material. Sets bearing strength and digging effort (context only).'],
    ['Coarse fragments', '% of volume', t => fmt(EX(t, 'sg_cfvo_pct'), 0), r => fmt(DUST_REF.coarse_pct, 0),
      'Share of gravel and rock pieces in the soil (context only).'],
    ['Soil texture (sand / silt / clay)', '%', t => (isNa(t.sand) || isNa(t.silt) || isNa(t.clay)) ? null : `${Math.round(t.sand)} / ${Math.round(t.silt)} / ${Math.round(t.clay)}`,
      r => 'median grain ~70 µm', 'Grain-size shares of the loose surface layer (context only).']
  ],
  biology: [
    ['Vegetation greenness (NDVI)', 'index', t => fmt(EX(t, 'ndvi_mean'), 2), r => '0 (none)',
      'MODIS vegetation index; 0 or below means bare ground. Used in the score.'],
    ['Precipitation', 'mm/yr', t => fmt(t.precip, 0), r => '0',
      'Water sustains life. Used in the score.'],
    ['Tree cover', '% of area', t => pct(num(EX(t, 'wc_frac_tree'))), r => '0',
      'ESA WorldCover (context).'],
    ['Moss and lichen cover', '% of area', t => pct(num(EX(t, 'wc_frac_moss_lichen'))), r => '0',
      'Polar and alpine life that satellite greenness under-reports (context).'],
    ['Cropland', '% of area', t => pct(num(EX(t, 'wc_frac_crop'))), r => '0',
      'Farmed land (context).'],
    ['Documented life', 'note', t => EX(t, 'known_life') || 'none recorded', r => 'sterile',
      'Microbial life that satellites cannot see (warm springs, acidophiles, microbialites) caps the sterility score at −0.6.']
  ]
};


// ══════════════════════════════════════
// CONCLUSIONS ENGINE — implements similarity_assessment_guide.md (v1.1), fed with the final scores
// Three sections (terrain, comp = geology, ice = water/ice) → four blocks: Overview, Strongest match, Main caution, Earth vs Moon
// The overall verdict uses the composite (overall match) score.
// ══════════════════════════════════════
const CONCL_KEYS = ['terrain', 'comp', 'ice'];
const CONCL_CAT = { terrain: 'terrain', comp: 'geology', ice: 'water_ice' };
const SEC_NAME = { terrain: catTitle('terrain'), comp: catTitle('geology'), ice: catTitle('water_ice') };
const SEC_SHORT = { terrain: 'Terrain', comp: 'Geology', ice: 'Water / ice' };

// §1.1 bands (lower bound inclusive) and §1.2 borderline boundaries
const BANDS = [
  { id: 'S', min: 0.65, label: 'strong analog' },
  { id: 'G', min: 0.40, label: 'good analog' },
  { id: 'P', min: 0.15, label: 'partial analog' },
  { id: 'N', min: -0.15, label: 'neutral' },
  { id: 'C', min: -0.50, label: 'contrasting' },
  { id: 'X', min: -Infinity, label: 'strongly contrasting' }
];
const BAND_SHORT = { S: 'strong', G: 'good', P: 'partial', N: 'neutral', C: 'contrasting', X: 'strongly contrasting' };
const BOUNDS = [0.65, 0.40, 0.15, -0.15, -0.50];
const BORDER = 0.05;
const bandOf = s => BANDS.find(b => s >= b.min).id;
const bandIdx = id => BANDS.findIndex(b => b.id === id);
function classify(s) {
  const b = BOUNDS.find(x => Math.abs(s - x) <= BORDER);
  if (b === undefined) { const id = bandOf(s); return { band: id, upper: id, lower: id, border: false, above: true }; }
  const upper = bandOf(b), lower = BANDS[bandIdx(upper) + 1].id;
  return { band: bandOf(s), upper, lower, border: true, above: s >= b };
}
const bandLabel = c => (c.border ? `${BAND_SHORT[c.upper]} / ${BAND_SHORT[c.lower]}` : BAND_SHORT[c.band]) + (c.capped ? ' (low confidence)' : '');
const OVERALL = { S: 'strong analog', G: 'good analog', P: 'partial analog', N: 'weak analog', C: 'contrasting site', X: 'strongly contrasting site' };
const overallLabel = c => { if (!c.border) return OVERALL[c.band]; const u = OVERALL[c.upper], l = OVERALL[c.lower]; return u.endsWith('analog') && l.endsWith('analog') ? u.replace(' analog', '') + ' / ' + l : u + ' / ' + l; };

// §3.1 / §4.1 / §5.1 band texts [analog value, limitations]
const BAND_TEXT = {
  terrain: {
    S: ['The ground at {earth} has slopes, relief and roughness close to {moon}. It can host full-scale rover traverses, landing-hazard detection trials and EVA route planning, with results transferable to the lunar region.',
        'Terrain alone does not capture lunar gravity: wheel sinkage and slip at one-sixth g differ. Correct mobility results for gravity, or use reduced-weight rovers.'],
    G: ['The terrain at {earth} is clearly similar to {moon} in steepness and ruggedness. It suits mobility and navigation testing and hazard-avoidance algorithm training, as long as the differences below are tracked.',
        'Some terrain statistics differ. Mark test routes that exceed or fall short of lunar slope and roughness, and weight results accordingly.'],
    P: ['{earth} shares some terrain traits with {moon}, typically similar steepness or similar relief, but not both. Useful for testing individual subsystems such as slope-climbing or obstacle detection.',
        'The overall landscape differs, so end-to-end traverse or landing rehearsals would not be representative.'],
    N: ['Terrain resemblance is weak. {earth} can still serve for basic functional checks of mobility hardware.',
        'Results on navigation, traverse time or hazard rates would not transfer to {moon}. Do not use this site for terrain-dependent conclusions.'],
    C: ['Little terrain value: the ground at {earth} is markedly flatter or steeper and rougher than {moon}.',
        'Mobility and landing tests would be biased: too easy if the site is flatter, overly conservative if it is steeper.'],
    X: ['None for terrain.',
        "The landscape is the opposite of the lunar region's. Exclude this site from any terrain, mobility or landing study."]
  },
  comp: {
    S: ['The rocks at {earth} are among the closest Earth matches to the {moon} highland crust in type and setting. They suit spectrometer calibration, sample-handling procedures and geology EVA training.',
        'Earth rock is chemically weathered and contains water-bearing minerals that lunar rock lacks. Calibrate instruments for hydrated minerals, and do not treat Earth soil as a regolith simulant without checking grain shape.'],
    G: ['{earth} shares the key material traits of {moon}, typically the rock family or the impact-processed setting. It suits instrument field trials and training in sampling and rock identification.',
        'Mineral proportions and age differ. Document which minerals differ so instrument detections are interpreted correctly.'],
    P: ['Partial material match: {earth} offers one useful trait, such as basaltic or impact-processed rock, or regolith-like fine soil, but not the overall highland composition of {moon}. Useful for hardware tests (drills, scoops, dust seals).',
        'Geology results would not transfer to the {moon} highland crust.'],
    N: ["Low material relevance: {earth}'s rocks are neither lunar-like nor opposite to them. Can serve as a neutral testbed for mechanical sampling hardware.",
        'Do not use this site for mineralogy or geology training aimed at {moon}.'],
    C: ["Little material value: {earth}'s ground is clearly unlike lunar highland material.",
        'Instruments and procedures would be trained on the wrong materials.'],
    X: ['None for composition.',
        'The material is opposite to lunar crust (for example clay-rich, wet or organic ground). Exclude this site from material-related tests.']
  },
  ice: {
    S: ['{earth} has ice in the ground and within reach of landable terrain, much like {moon}. It suits end-to-end rehearsal of ice prospecting: detection, drilling, sampling and the traverse from a landing site to ice.',
        'Earth ice is almost pure and abundant, while lunar ice is a few percent mixed into regolith at 25–110 K and sublimates in vacuum.'],
    G: ['Frozen ground or ice near landable terrain makes {earth} a good site for testing ice-detection instruments and drilling into ice-cemented ground.',
        'Ice amount, depth or distance differ from {moon}. Treat detection rates as optimistic.'],
    P: ['{earth} offers some ice-related conditions, typically permafrost without accessible surface ice, or ice far from flat ground. Useful for instrument calibration or drilling hardware trials.',
        'Not representative of the full ice-access chain at {moon}.'],
    N: ['Little ice relevance; {earth} can still host dry-run tests of drilling hardware.',
        'Ice prospecting results would not transfer to {moon}.'],
    C: ['Little ice value: {earth} has no lasting frozen ground, or its water is mostly liquid or seasonal.',
        'Instruments would be tested on the wrong water form (liquid, seasonal).'],
    X: ['None for ice and resources.',
        'Wet, warm or ice-free ground is the opposite of lunar cold traps. Exclude this site from ice and ISRU studies.']
  }
};
// Default compensation actions per section (used when the triggered notes give fewer than one)
const DEFAULT_ACTIONS = {
  terrain: ['Pair with a second analog that covers the missing terrain traits, or use simulators fed with lunar elevation data.'],
  comp: ['Pair with an anorthosite site for mineralogy work, or use a lunar highland regolith simulant on site.'],
  ice: ['Run ice tests in a cold chamber at lunar temperatures, or pair with an ice-rich analog.']
};
// "Suitable for" lists; the last item is the end-to-end rehearsal (dropped below band S)
const EXPERIMENTS = {
  terrain: { full: ['landing-hazard detection', 'rover mobility and navigation', 'EVA traverse planning', 'habitat and power-mast siting', 'line-of-sight and communications planning', 'end-to-end landing and traverse rehearsals'],
             partial: ['slope-climbing and obstacle-detection subsystem tests'] },
  comp: { full: ['spectrometer and mineralogy field tests', 'sample collection and handling', 'geology EVA training', 'regolith excavation and construction', 'dust mitigation', 'end-to-end sampling campaigns'],
          partial: ['drill, scoop and dust-seal hardware tests'] },
  ice: { full: ['ice-prospecting instruments (ground-penetrating radar, neutron spectrometers)', 'drilling and coring in frozen ground', 'ISRU ice extraction', 'ice-sample handling', 'end-to-end landing-to-ice logistics'],
         partial: ['instrument calibration and drilling hardware trials'] }
};
const USABLE = {
  terrain: 'Basic functional checks of mobility hardware can still be run here.',
  comp: 'Drills, scoops and dust seals can still be tested here.',
  ice: 'Dry-run tests of drilling hardware can still be run here.'
};

// §2.3 site profile
const PROFILE = {
  'terrain,comp,ice': ['Full-site analog', '{earth} matches {moon} on all three sections, a rare candidate for integrated mission rehearsals.'],
  'terrain,comp': ['Geology traverse analog', '{earth} is best suited to surface operations and geology field training that do not depend on ice.'],
  'terrain,ice': ['Ice-access operations analog', '{earth} suits rehearsals of driving to, and working at, ice-bearing ground, while its rocks are not lunar-like.'],
  'comp,ice': ['Resource science analog', '{earth} suits instrument and sampling work on lunar-like materials and ice, but not mobility tests.'],
  'terrain': ['Mobility and landing analog', '{earth} is mainly a terrain analog: landing-hazard, navigation and rover-mobility tests.'],
  'comp': ['Materials analog', '{earth} is mainly a materials analog: instruments, sample handling, excavation and dust.'],
  'ice': ['Ice prospecting analog', '{earth} is mainly a frozen-ground analog: prospecting instruments and drilling.'],
  mid: ['Generalist, partial', '{earth} partly resembles {moon} on every section without excelling at any; useful for low-cost early testing.'],
  low: ['Not an analog', '{earth} offers little lunar-relevant similarity to {moon}; prefer another site.']
};

// Ice section texts for a DRY Moon target (the Tranquillitatis pit): the water/ice score rewards dry, ice-free Earth ground there
const ICE_DRY_TEXT = {
  S: ['{earth} is dry and ice-free like {moon}: no frozen ground or liquid water to distort dust, drilling or instrument tests.',
      'Earth air still carries humidity; lunar regolith holds no water at all outside polar cold traps.'],
  G: ['{earth} is mostly dry, close to the waterless ground at {moon}.',
      'Some moisture or seasonal frost remains. Dry samples before handling tests.'],
  P: ['{earth} is partly dry; some moisture or frost is present, unlike {moon}.',
      'Water-sensitive tests (dust, sintering, spectra) need dried samples or a simulant.'],
  N: ['Moisture conditions at {earth} neither match nor oppose the dry {moon}.',
      'Treat water-sensitive results with care.'],
  C: ['Little value: {earth} has frozen or wet ground, while {moon} is dry.',
      'Water in the ground will bias dust, drilling and spectral tests.'],
  X: ['None: {earth} is icy or wet, the opposite of the dry {moon}.',
      'Exclude this site from tests that depend on dry regolith.']
};
const PROFILE_DRY = {
  'terrain,ice': ['Dry-ground operations analog', '{earth} suits surface operations on dry, ice-free ground like {moon}, while its rocks are not lunar-like.'],
  'comp,ice': ['Dry materials analog', '{earth} suits instrument and sampling work on dry, lunar-like materials, but not mobility tests.'],
  'ice': ['Dry-ground analog', '{earth} mainly offers dry, ice-free ground like {moon}; little else matches.']
};
// Highland wording -> mare wording when the Moon target is mare basalt (the Tranquillitatis pit)
function crustify(s, r) {
  if (!isMare(r)) return s;
  return s.replace(/highland crust/g, 'mare basalt').replace(/overall highland composition/g, 'overall mare-basalt composition')
    .replace(/lunar highland material/g, 'lunar mare basalt').replace(/an anorthosite site/g, 'a basalt site')
    .replace(/lunar highland regolith simulant/g, 'lunar mare regolith simulant').replace(/highland/g, 'mare');
}

// Rock-type class from the pipeline's anorthosite flag, then the primary lithology (first-listed), then context (§4.2)
function rockClass(t) {
  const lith = (t.lith || '').toLowerCase(), first = lith.split(/[,;]/)[0], ctx = (t.xinfo || '').toLowerCase(), an = num(EX(t, 'anorth'));
  if (an === 2 || (an === null && /anorthosit|norite|troctolit/.test(first))) return 'anorth';
  if (t.source === 'impact_structure' || /breccia|suevite|impact melt/.test(lith)) return 'impact';
  if (/basalt|andesit|tholeiit|volcan|lava|tephra|scoria|pyroclast|tuff|trachy|dacit|rhyolit|picrit/.test(first) || t.source === 'volcano' || /basalt/.test(ctx)) return 'basalt';
  if (/sandstone|shale|limestone|mudstone|siltstone|conglomerate|sediment|alluvi|evaporit|dolomit|clay|sand|gravel|till|loess|colluvi/.test(first)) return 'sedimentary';
  if (/granit|gneiss|schist|quartzit|migmatit|granodiorit|tonalit/.test(first)) return 'granitic';
  return lith ? 'other' : null;
}

// Parameter notes (§3.2 / §4.2 / §5.2). Each: {id, side:'value'|'limit', text, w (sort weight), act (compensation, optional)}
function sectionNotes(key, t, r, ctx) {
  const N = [], add = (id, side, text, w, act) => N.push({ id, side, text, w, act });
  const f1 = v => (+v).toFixed(1), f0 = v => Math.round(+v).toLocaleString('en-US');
  if (key === 'terrain') {
    const es = num(t.slope), ms = num(r.slope);
    if (es !== null && ms !== null) {
      const d = es - ms;
      if (d < -4) add('T1', 'limit', `${ctx.earth} is gentler (mean slope ${f1(es)}° vs ${f1(ms)}°). Rover tests will underestimate lunar slope demands.`, Math.abs(d) / 4, 'Add dedicated steep-slope test lanes or tilt-table tests.');
      if (d > 4) add('T2', 'limit', `${ctx.earth} is steeper (mean slope ${f1(es)}° vs ${f1(ms)}°). Mobility results are conservative, but landing-zone hazard rates will be overstated.`, Math.abs(d) / 4, 'Correct landing-hazard rates for the steeper ground.');
      if (Math.abs(d) <= 2) add('T3', 'value', `Mean slope is almost the same (${f1(es)}° vs ${f1(ms)}°).`, Math.abs(d) / 2);
    }
    const er = num(t.relief), mr = num(r.relief);
    if (er !== null && mr) {
      const q = Math.max(er, 1) / mr;
      if (q < 0.5) add('T4', 'limit', `Much less vertical relief (${f0(er)} m vs ${f0(mr)} m): crater-rim shadowing, line-of-sight and comms blackouts are under-represented.`, 1 / q, 'Simulate rim shadowing and line-of-sight with lunar elevation data.');
      if (q > 2) add('T5', 'limit', `Far more vertical relief (${f0(er)} m vs ${f0(mr)} m) than the lunar target: traverses will be more constrained than on the Moon.`, q, 'Plan test routes on the lunar slope map, not the local one.');
      if (q >= 0.75 && q <= 1.33) add('T6', 'value', `Comparable relief (${f0(er)} m vs ${f0(mr)} m), which suits line-of-sight and communications planning.`, Math.abs(Math.log(q)));
    }
    const eg = num(t.rough), mg = num(r.tri);
    if (eg !== null && mg) {
      const q = Math.max(eg, 0.1) / mg;
      if (q > 1.5) add('T7', 'limit', `Rougher ground (roughness index ${f0(eg)} m vs ${f0(mg)} m): obstacle density is higher than at ${ctx.moon}.`, q, 'Weight obstacle-detection results by the lower lunar obstacle density.');
      if (q < 0.5) add('T8', 'limit', `Smoother ground (roughness index ${f0(eg)} m vs ${f0(mg)} m): obstacle density is lower than at ${ctx.moon}.`, 1 / q, 'Add artificial obstacle fields to the test area.');
    }
    const g20 = num(EX(t, 'frac_slope500_gt20')), m20 = num(r.steep20);
    if (g20 !== null && m20 !== null && Math.abs(g20 * 100 - m20) > 15)
      add('T9', 'limit', `The share of very steep ground (> 20°) differs (${f0(g20 * 100)}% vs ${f0(m20)}%), so no-go zones in route planning will not match.`, Math.abs(g20 * 100 - m20) / 15, 'Plan routes with the lunar no-go map, not the local one.');
    const fez = num(EX(t, 'frac_elev_zero'));
    if (fez !== null && fez > 0.2) add('T10', 'limit', `About ${f0(fez * 100)}% of the tile lies at sea or lake level; part of its terrain statistics reflect water, not ground.`, fez * 5, 'Restrict tests to the land part of the tile.');
  }
  if (key === 'comp') {
    const rc = rockClass(t), mare = isMare(r), rockName = (cleanLith(t) || 'This rock').split(/[,;]/)[0];
    const an = num(EX(t, 'anorth'));
    if (rc === 'anorth') {
      if (mare) add('C1', 'limit', `Anorthosite-family rock matches the lunar highlands, not the mare basalt at ${ctx.moon}.`, 1.5, 'Pair with a basalt site for mare mineralogy work.');
      else add('C1', 'value', 'Anorthosite-family rock: the closest Earth match to lunar highland crust, ideal for spectrometer and mineral-identification work.', 0);
    }
    if (an === 1 && rc !== 'anorth' && !mare)
      add('C1b', 'value', 'Anorthosite is present only as a minor component: a partial highland match (the geology score is capped at +0.30).', 0.8);
    if (rc === 'basalt') {
      if (mare) add('C2', 'value', `Basaltic rock: the closest Earth match to the mare basalt at ${ctx.moon}, good for regolith mechanics, volcanic-glass and dust tests.`, 0);
      else {
        add('C2', 'value', 'Basaltic rock: good for regolith mechanics, volcanic-glass and dust tests.', 0.5);
        add('C2', 'limit', `Basalt matches the lunar maria, not the ${ctx.moon} highlands; mineral proportions (more iron, less feldspar) differ.`, 2, 'Pair with an anorthosite site for mineralogy work.');
      }
    }
    if (rc === 'impact') add('C3', 'value', 'Impact-processed rock (breccia, shocked minerals): directly relevant to lunar craters and ejecta.', 0.2);
    if (rc === 'sedimentary' || rc === 'granitic')
      add('C4', 'limit', `${rockName.charAt(0).toUpperCase() + rockName.slice(1)} has no lunar counterpart; mineralogy results will not transfer.`, 2.5, mare ? 'Pair with a basalt site for mineralogy work.' : 'Pair with an anorthosite site for mineralogy work.');
    if (!isNa(t.silt) && !isNa(t.clay)) {
      const fines = t.silt + t.clay;
      if (fines >= 40 && t.clay <= 25) add('C5', 'value', `Fine-grained surface layer (${f0(fines)}% silt and clay), comparable to lunar regolith dust for dust-mitigation tests.`, 0.6);
      if (t.clay > 30) add('C6', 'limit', `High clay content (${f0(t.clay)}%): clay is cohesive and water-bearing, unlike dry, sharp-edged lunar regolith grains.`, t.clay / 30, 'Use a dry lunar regolith simulant for grain-behaviour tests.');
    }
    if (!isNa(t.bdod)) {
      const bd = +t.bdod, dd = Math.abs(bd - 1.5);
      if (dd <= 0.2) add('C7', 'value', `Bulk density (${bd.toFixed(2)} g/cm³) is close to lunar regolith (~1.5 g/cm³), useful for excavation and bearing-strength tests.`, dd);
      if (bd < 1.2 || bd > 1.8) add('C8', 'limit', `Bulk density (${bd.toFixed(2)} g/cm³) differs from lunar regolith (~1.5 g/cm³): digging forces and lander footing will not match.`, dd / 0.3, 'Use a regolith simulant for excavation and bearing tests.');
    }
    const cf = num(EX(t, 'sg_cfvo_pct'));
    if (cf !== null && cf > 30) add('C9', 'limit', `Many coarse fragments (${f0(cf)}% by volume): rockier than typical lunar regolith.`, cf / 30, 'Select fine-grained test patches for regolith work.');
    const ta = num(EX(t, 'mstrat_t_age_ma')), ba = num(EX(t, 'mstrat_b_age_ma'));
    const per = periodOf(r), moonAge = `lunar ${r.geo_age ? r.geo_age + ' ' : ''}crust${per ? ` (${per} million years)` : ' (billions of years old)'}`;
    if (ba !== null && ba >= 2500) add('C10', 'value', `Archean crust (up to ${f0(ba)} million years old), among the oldest rock on Earth, the closest Earth gets to the age of ${moonAge}.`, 0.3);
    if (ta !== null && ta < 66) add('C11', 'limit', `Young rock (${t.geo_age ? t.geo_age + ', ' : ''}${f0(ta)}–${f0(ba ?? ta)} million years) versus ${moonAge}; weathering and alteration history differ greatly.`, 1.5, 'Interpret alteration and hydrated minerals as Earth-specific.');
    if (isNa(t.sand) && isNa(t.bdod)) add('C12', 'limit', 'No soil grain-size or density data for this tile; the regolith comparison relies on rock type only.', 0.4, 'Measure soil properties on site before excavation tests.');
  }
  if (key === 'ice') {
    const fr = num(EX(t, 'x_frozen_ground_frac')), su = num(EX(t, 'x_surface_ice_frac')), mg = num(EX(t, 'pf_magt_10m_c')),
          pf = EX(t, 'pf_frac_pct'), tr = num(EX(t, 'trav_dist_flat_to_ice_km')), g10 = EX(t, 'trav_dist_gt10_flag'),
          pr = num(t.precip), moist = num(t.moist);
    if (pr !== null && pr >= 500) add('W10', 'limit', `Wet climate (${f0(pr)} mm/yr): liquid water dominates the ground, which caps the water/ice score at 0.`, 2, 'Pick a drier site, or treat results as seasonal frost only.');
    if (moist !== null && moist > 20) add('W7', 'limit', `Moist soil (${f1(moist)}%): liquid water does not exist on the Moon, and instruments will read a stronger water signal than on the Moon.`, moist / 20, 'Use dry-handling protocols and calibrate for the stronger water signal.');
    if (dryTarget(r)) {
      const anyIce = Math.max(fr ?? 0, su ?? 0);
      if (anyIce < 0.05 && pr !== null && pr < 250) add('W11', 'value', `Dry, ice-free ground (${f0(pr)} mm/yr, no frozen ground), like the waterless ${ctx.moon}.`, 0.2);
      if (anyIce >= 0.2) add('W12', 'limit', `Frozen or icy ground (${f0(anyIce * 100)}% of the tile), while ${ctx.moon} is dry.`, anyIce * 3, 'Choose an ice-free patch, or dry samples before testing.');
      return N;
    }
    const ice1m = num(r.ice1m), ct = num(r.coldtrap), l2i = num(r.landing_to_ice_km), dep = num(r.ice_depth);
    if (fr !== null && fr >= 0.5 && ice1m !== null && ice1m >= 40) add('W1', 'value', `Both sites have widespread frozen ground (${f0(fr * 100)}% vs ${f0(ice1m)}% with ice stable within 1 m), suitable for drilling into ice-cemented soil.`, Math.abs(fr * 100 - ice1m) / 100);
    if (mg !== null && mg <= -10) add('W2', 'value', `Cold, stable permafrost (${mg.toFixed(1)} °C at 10 m depth): hard, ice-cemented ground mechanically closer to lunar cold-trap regolith.`, 0.2);
    if (mg !== null && mg > -2) add('W3', 'limit', `Warm permafrost (${mg.toFixed(1)} °C at 10 m depth): near thawing, wetter and softer than lunar frozen regolith.`, 1.2, 'Treat drilling results as optimistic for soft, wet ground.');
    if (su !== null && su > 0.3 && ct !== null && ct < 10) add('W4', 'limit', `Massive surface ice (${f0(su * 100)}% cover) versus scattered cold traps (${f1(ct)}% of ${ctx.moon}): accessibility is far easier on Earth than on the Moon.`, su * 3, 'Scale detection thresholds down to lunar ice concentrations (a few percent by mass in regolith).');
    if (l2i) {
      if (fr !== null && fr >= 0.5) add('W5', 'value', `Ice is in the ground itself, so the landing-to-ice traverse (${l2i.toFixed(2)} km at ${ctx.moon}) is never longer here.`, 0.6);
      else if (tr !== null && g10 !== 1) {
        const q = Math.max(tr, 0.05) / l2i;
        if (q >= 0.5 && q <= 2) add('W5', 'value', `Landable ground lies ${f1(tr)} km from ice, comparable to ${l2i.toFixed(2)} km at ${ctx.moon}: realistic for rehearsing traverse logistics.`, Math.abs(Math.log(q)));
      }
    }
    if (g10 === 1 && !(fr !== null && fr >= 0.5)) add('W6', 'limit', 'No ice within 10 km of landable ground: the landing-to-ice traverse cannot be rehearsed here.', 1.8, 'Simulate the landing-to-ice traverse, or pair with an ice-proximal site.');
    if (dep !== null) add('W8', 'value', `At ${ctx.moon}, ice is typically stable from about ${dep.toFixed(2)} m depth; drilling tests should target this depth.`, 0.9);
    if (isNa(pf)) add('W9', 'limit', 'No permafrost-model data for this tile; the frozen-ground share uses a proxy.', 0.3, 'Measure ground temperature on site.');
  }
  return N;
}

// §1.3 confidence: share of the section's parameters available on each side
const CONF_PARAMS = {
  terrain: { earth: t => [t.slope, t.relief, t.rough, EX(t, 'slope500_p90_deg'), EX(t, 'frac_slope500_gt20')], moon: r => [r.slope, r.relief, r.tri, r.slope_p90, r.steep20],
             names: ['mean slope', 'relief', 'roughness', '90th-percentile slope', 'steep-ground share'] },
  comp: { earth: t => [t.lith || null, t.geo_age || null, t.sand, t.bdod, EX(t, 'mstrat_b_age_ma'), EX(t, 'sg_cfvo_pct')], moon: r => [r.geo_name, r.geo_age, 1, 1, periodOf(r), 1],
          names: ['rock type', 'geologic age', 'soil texture', 'bulk density', 'age range', 'coarse fragments'] },
  ice: { earth: t => [EX(t, 'x_frozen_ground_frac'), EX(t, 'x_surface_ice_frac'), EX(t, 'pf_frac_pct'), EX(t, 'pf_magt_10m_c'), t.moist, EX(t, 'trav_dist_gt10_flag') === 1 ? 1 : EX(t, 'trav_dist_flat_to_ice_km')],
         moon: r => dryTarget(r) ? [1, 1, 1, 1, 1, 1] : [r.ice1m, r.surf_ice, r.coldtrap, r.ice_depth, r.psr, r.landing_to_ice_km],
         names: ['frozen ground', 'surface ice', 'permafrost share', 'ground temperature', 'soil moisture', 'distance to ice'] }
};
function confidence(key, t, r) {
  const P = CONF_PARAMS[key], e = P.earth(t), m = P.moon(r);
  const ok = v => !isNa(v);
  const ce = e.filter(ok).length / e.length, cm = m.filter(ok).length / m.length, c = Math.min(ce, cm);
  const missing = [];
  e.forEach((v, i) => { if (!ok(v)) missing.push(P.names[i] + ' (Earth)'); });
  m.forEach((v, i) => { if (!ok(v)) missing.push(P.names[i] + ' (Moon)'); });
  return { level: c >= 0.8 ? 'normal' : c >= 0.5 ? 'moderate' : 'low', missing };
}

// §7 general Earth–Moon differences
const tempRange = r => (!isNa(r.t_min) && !isNa(r.t_max)) ? `about ${Math.round(r.t_min)} K to ${Math.round(r.t_max)} K` : 'deep cryogenic cold to well above 300 K';
function differences(t, r, ctx) {
  const k = v => Math.round(v), mn = moonNoon(r);
  return {
    D1: ['Gravity', `Rovers, landers and crews at ${ctx.earth} carry six times the lunar weight; traction, sinkage, digging forces and dust settling all differ.`, 'weight-offloaded rovers or counterweight rigs, scaling terramechanics results to one-sixth g, and reduced-gravity flights for dust and human-movement tests.'],
    D2: ['Vacuum', `Air at ${ctx.earth} cools equipment by convection, carries dust and keeps ice stable; on the Moon ice sublimates and heat moves only by radiation and conduction.`, 'repeating thermal, dust and ice-handling tests in a thermal-vacuum chamber, treating field results as an upper bound for cooling.'],
    D3: ['Radiation', 'The field site is shielded by the atmosphere and magnetic field; on the Moon, solar particles and cosmic rays reach the surface directly.', 'testing electronics and materials at a radiation beam facility; radiation effects cannot be inferred from field data.'],
    D4: ['Temperature extremes', `${ctx.moon} ranges from ${tempRange(r)}${!isNa(t.tmin) && !isNa(t.tmax) ? `; ${ctx.earth} stays between about ${k(t.tmin)} K and ${k(t.tmax)} K` : ''}.`, 'cryogenic and thermal-vacuum chamber tests at 40–110 K for ice sampling and mechanisms; field results show warm-end behaviour only.'],
    D5: ['Low sun and long shadows', `At ${ctx.moon} the Sun never rises above about ${mn.toFixed(1)}°, so shadows are long and permanent shadow is common; at ${ctx.earth} the noon Sun reaches ${noonSun(t.lat, 23.44).toFixed(0)}°.`, 'running navigation and imaging tests at dawn or dusk, or under low-angle artificial light, and using lunar illumination models for power planning.'],
    D6: ['Regolith and space weathering', `Lunar soil is made by micrometeorite impacts and solar wind: sharp, glassy, electrostatically charged grains with no water or oxidation. Soil at ${ctx.earth} is weathered and rounded.`, `certified lunar simulants (${isMare(r) ? 'mare' : 'highland'} type for this target) where grain shape or charging matters, and dust-adhesion tests in vacuum.`],
    D7: ['Water and life', `${ctx.earth} has liquid water, microbes and organic material; the Moon has none outside its cold traps.`, 'contamination controls (sterilised tools, field blanks, sealed sample containers), recording terrestrial signals so they are not mistaken for lunar ones.'],
    D8: ['Weather and erosion', `Wind, rain and freeze–thaw have smoothed and sorted the surface at ${ctx.earth}; the lunar surface changes only through slow impact gardening.`, 'accounting for erosion-softened slopes and sorted sediments, and sampling below the weathered surface layer for geology work.'],
    D9: ['Navigation and communications', r.pit ? 'There is no GPS on the Moon and a 1.3-second signal delay to Earth.' : 'There is no GPS on the Moon, a 1.3-second signal delay to Earth, and limited Earth visibility at the pole.', 'GNSS-denied navigation tests with simulated signal delay and communication blackouts.']
  };
}
const DIFF_PRIORITY = { terrain: ['D5', 'D9', 'D8', 'D6'], comp: ['D6', 'D8', 'D7'], ice: ['D4', 'D7', 'D5'] };

// Experiments (final readiness) that need a given category and that this pair supports
const expShort = n => n.replace(/\s*\([^)]*\)\s*$/, '');
function pairExperiments(t, r, cat) {
  const rel = (r.relevant || []).filter(id => EXP[id] && (!cat || cat in (EXP[id].required || {})));
  const suf = rel.filter(id => SC.ready(r.id, id, t.idx) === 3).map(id => expShort(EXP[id].name));
  const par = rel.filter(id => SC.ready(r.id, id, t.idx) === 2).map(id => expShort(EXP[id].name) + ' (partial)');
  return { suf, par, rel };
}

function buildConclusions(t, r, scores) {
  const ctx = { earth: t.name, moon: r.name };
  const fill = s => crustify(s.replace(/\{earth\}/g, ctx.earth).replace(/\{moon\}/g, ctx.moon), r);
  const dry = dryTarget(r);
  const text = (key, band, i) => (key === 'ice' && dry ? ICE_DRY_TEXT : BAND_TEXT[key])[band][i];
  const S = CONCL_KEYS.map(key => {
    const raw = scores[CONCL_CAT[key]];
    const score = isNa(raw) ? 0 : Math.max(-1, Math.min(1, raw));
    const cls = classify(score), conf = confidence(key, t, r);
    if (isNa(raw)) { conf.level = 'low'; conf.missing.unshift('score'); }
    // §1.3: low confidence caps the shown band at P
    if (conf.level === 'low' && ['S', 'G'].includes(cls.upper)) { cls.band = cls.upper = cls.lower = 'P'; cls.border = false; cls.capped = true; }
    const notes = sectionNotes(key, t, r, ctx);
    return { key, score, cls, conf, notes,
      value: notes.filter(n => n.side === 'value').sort((a, b) => a.w - b.w),
      limit: notes.filter(n => n.side === 'limit').sort((a, b) => b.w - a.w) };
  });

  // §2.1 ranking, ties and sign rules
  const sorted = [...S].sort((a, b) => b.score - a.score);
  const evenAll = sorted[0].score - sorted[2].score <= BORDER;
  let strong, weak;
  if (evenAll) {
    strong = [[...S].sort((a, b) => b.value.length - a.value.length)[0]];
    weak = [[...S].filter(x => x !== strong[0]).sort((a, b) => b.limit.length - a.limit.length)[0]];
  } else {
    if (sorted[0].score - sorted[1].score <= BORDER) { strong = [sorted[0], sorted[1]]; weak = [sorted[2]]; }
    else {
      strong = [sorted[0]];
      weak = sorted[1].score - sorted[2].score <= BORDER ? [sorted[2], sorted[1]] : [sorted[2]];   // weakest first
    }
  }
  const allLow = sorted[0].score < 0.15, allHigh = sorted[2].score >= 0.40;

  // §2.2 overview — the verdict score is the final composite (overall match to this Moon target)
  const comp = scores.composite;
  const mean = isNa(comp) ? S.reduce((a, x) => a + x.score, 0) / 3 : Math.max(-1, Math.min(1, comp));
  const oc = classify(mean);
  let overLabel = overallLabel(oc);
  const notes = [];
  const xSec = S.find(x => x.cls.band === 'X');
  if (xSec && ['S', 'G'].includes(oc.band)) overLabel = OVERALL.P;
  if (xSec) notes.push(`Not a full-site analog; see ${SEC_NAME[xSec.key]}.`);
  const strongSecs = S.filter(x => ['S', 'G'].includes(x.cls.band));
  if (strongSecs.length === 1 && S.filter(x => x !== strongSecs[0]).every(x => bandIdx(x.cls.band) >= bandIdx('N')))
    notes.push(`Specialised analog: valuable for ${SEC_NAME[strongSecs[0].key].toLowerCase()} tests only.`);
  const lvl = x => x.cls.capped ? 'M' : x.score >= 0.40 ? 'H' : x.score >= 0.15 ? 'M' : 'L';   // low-confidence sections count as Mid
  const highs = S.filter(x => lvl(x) === 'H').map(x => x.key);
  let prof;
  if (S.every(x => lvl(x) === 'M')) prof = PROFILE.mid;
  else if (S.every(x => lvl(x) === 'L')) prof = PROFILE.low;
  else { const k = (highs.length ? highs : [sorted[0].key]).join(','); prof = (dry && PROFILE_DRY[k]) || PROFILE[k]; }
  const confLines = S.filter(x => x.conf.level !== 'normal').map(x =>
    `${x.conf.level === 'low' ? 'Low' : 'Moderate'} confidence on ${SEC_NAME[x.key]}: ${x.conf.missing.join(', ')} not available.`);
  const allExp = pairExperiments(t, r, null);
  const expLine = allExp.rel.length ? `Sufficient for ${allExp.suf.length} and partial for ${allExp.par.length} of the ${allExp.rel.length} experiments relevant to this target.` : null;

  // Blocks 2 and 3
  const one = strong.length === 1;
  const strongBlocks = strong.map(x => {
    let value = fill(text(x.key, x.cls.upper, 0));
    if (x.cls.border) value = (x.cls.above ? 'Largely: ' : 'Partly: ') + value.charAt(0).toLowerCase() + value.slice(1);
    const pe = pairExperiments(t, r, CONCL_CAT[x.key]);
    return { key: x.key, score: x.score, label: bandLabel(x.cls), text: value,
      evidence: x.value.slice(0, one ? 4 : 2).map(n => n.text), experiments: allLow ? [] : [...pe.suf, ...pe.par],
      gap: x.limit[0] ? x.limit[0].text : null, scale: x.key === 'terrain' };
  });
  const weakBlocks = weak.map(x => {
    const lim = fill(text(x.key, x.cls.lower, 1));
    const acts = [...new Set(x.limit.filter(n => n.act).map(n => n.act))].slice(0, 3);
    const defAct = x.key === 'ice' && dry ? ['Dry samples before testing, or use a dry regolith simulant.'] : DEFAULT_ACTIONS[x.key].map(fill);
    return { key: x.key, score: x.score, label: bandLabel(x.cls), text: lim,
      evidence: x.limit.slice(0, weak.length === 1 ? 4 : 2).map(n => n.text),
      actions: acts.length ? acts : defAct,
      usable: x.score >= -0.15 ? USABLE[x.key] : null, scale: x.key === 'terrain' };
  });

  // Block 4 (low sun only applies where the Moon target's sun is low: not at the equatorial pit)
  const D = differences(t, r, ctx), ids = ['D1', 'D2', 'D3'], lowSun = moonNoon(r) < 30;
  [strong[0].key, weak[0].key].forEach(k => { const id = DIFF_PRIORITY[k].find(d => !ids.includes(d) && (d !== 'D5' || lowSun)); if (id && ids.length < 5) ids.push(id); });
  // section-specific items first (strongest, then weakest), then the universal ones
  const order = [...ids.slice(3), ...ids.slice(0, 3)];

  return {
    overview: { verdict: `${ctx.earth} ⟷ ${ctx.moon}`, label: overLabel, mean,
      scores: sorted.map(x => ({ key: x.key, score: x.score, label: bandLabel(x.cls) })),
      profile: [prof[0], fill(prof[1])], notes: expLine ? [...notes, expLine] : notes, confLines, even: evenAll },
    strongHead: allLow ? 'Least different' : 'Strongest match',
    weakHead: allHigh ? 'Points to watch' : 'Main caution',
    allLow, strong: strongBlocks, weak: weakBlocks,
    diffs: order.map(id => [...D[id], id]), t, r
  };
}

// ── Brief view: one-line takeaways; the full sentences stay available as hover tooltips ──
const firstSentence = s => { const m = s.match(/^.+?[.!?](\s|$)/); return m ? m[0].trim() : s; };
// Shorten a parameter note to its factual core: cut at the first ": ", "; ", ", which" or ". " after 20 characters
function brief(s) {
  let cut = s.length;
  [': ', '; ', ', which', ', so ', '. '].forEach(sep => { const i = s.indexOf(sep, 20); if (i > 0 && i < cut) cut = i; });
  return s.slice(0, cut).replace(/[.,;:]$/, '') + '.';
}
// Earth vs Moon: [short difference, short action]
const DIFF_SHORT = {
  D1: (t, r, x) => ['One-sixth gravity on the Moon', 'Offload weight and scale results to 1/6 g'],
  D2: (t, r, x) => ['No air: no convection, ice sublimates', 'Repeat key tests in a thermal-vacuum chamber'],
  D3: (t, r, x) => ['No atmospheric or magnetic shield', 'Test electronics at a radiation beam facility'],
  D4: (t, r, x) => [(!isNa(r.t_min) && !isNa(r.t_max) ? `${Math.round(r.t_min)}–${Math.round(r.t_max)} K on the Moon` : 'Cryogenic lunar cold') + (!isNa(t.tmin) && !isNa(t.tmax) ? ` vs ${Math.round(t.tmin)}–${Math.round(t.tmax)} K here` : ''), 'Cryogenic chamber tests at 40–110 K'],
  D5: (t, r, x) => [`Sun ≤ ${moonNoon(r).toFixed(1)}° at ${x.moon}, ${noonSun(t.lat, 23.44).toFixed(0)}° here`, 'Test at dawn, dusk or under low-angle light'],
  D6: (t, r, x) => ['Glassy, sharp, charged lunar grains', `Use a certified ${isMare(r) ? 'mare' : 'highland'} simulant`],
  D7: (t, r, x) => ['Liquid water and microbes on Earth', 'Sterile tools, field blanks, sealed samples'],
  D8: (t, r, x) => ['Earth surfaces are weathered and sorted', 'Sample below the weathered layer'],
  D9: (t, r, x) => ['No GPS and a 1.3 s signal delay', 'GPS-free navigation with simulated delay']
};
const DIFF_ICON = { D1: '⇣', D2: '◌', D3: '☢', D4: '❄', D5: '☀', D6: '◆', D7: '💧', D8: '≋', D9: '⌖' };

function conclusionsHTML(c, ctx) {
  const tone = v => v >= 0.15 ? 'pos' : v > -0.15 ? 'mid' : 'neg';
  const sec = k => CAT_RGB[CONCL_CAT[k]] || '100,180,255';
  const o = c.overview;
  const tip = s => esc(s).replace(/\n/g, ' ');
  const bullets = (arr, n) => arr.slice(0, n).map(s => `<li title="${tip(s)}">${esc(brief(s))}</li>`).join('');
  const s0 = c.strong[0], w0 = c.weak[0];
  const strongKeys = c.strong.map(b => SEC_NAME[b.key]).join(' · '), weakKeys = c.weak.map(b => SEC_NAME[b.key]).join(' · ');
  return `
  <div class="vx vx-${tone(o.mean)}">
    <div class="vx-kicker">Verdict · overall match</div>
    <div class="vx-main"><span class="vx-label">${esc(o.label)}</span><span class="vx-score">${fmtScore(o.mean)}</span></div>
    <div class="vx-pair">${esc(o.verdict)}</div>
    <div class="vx-profile"><b>${esc(o.profile[0])}</b><span>${esc(o.profile[1])}</span></div>
    <div class="vx-chips">${o.scores.map(s => `<span class="vx-chip" style="--c:${sec(s.key)}"><i>${SEC_SHORT[s.key]}</i><b class="${s.score >= 0 ? 'pos' : 'neg'}">${fmtScore(s.score)}</b><em>${esc(s.label)}</em></span>`).join('')}</div>
    ${[...o.notes, ...o.confLines].map(n => `<div class="vx-note" title="${tip(n)}">${esc(brief(n))}</div>`).join('')}
  </div>

  <div class="bx-grid">
    <div class="bx bx-pos">
      <div class="bx-kicker">✦ ${c.strongHead}</div>
      <div class="bx-title">${esc(strongKeys)}</div>
      <p class="bx-lead" title="${tip(c.strong.map(b => b.text).join(' '))}">${esc(firstSentence(s0.text))}${c.allLow ? ' Not recommended as an analog.' : ''}</p>
      ${s0.evidence.length ? `<ul class="bx-list">${bullets(c.strong.flatMap(b => b.evidence), 2)}</ul>` : ''}
      ${s0.experiments.length ? `<div class="bx-sub">Best for (experiment verdicts)</div><div class="bx-tags">${s0.experiments.slice(0, 3).map(e => `<span>${esc(e)}</span>`).join('')}</div>` : ''}
    </div>
    <div class="bx bx-neg">
      <div class="bx-kicker">⚠ ${c.weakHead}</div>
      <div class="bx-title">${esc(weakKeys)}</div>
      <p class="bx-lead" title="${tip(c.weak.map(b => b.text).join(' '))}">${esc(firstSentence(w0.text))}</p>
      ${w0.evidence.length ? `<ul class="bx-list">${bullets(c.weak.flatMap(b => b.evidence), 2)}</ul>` : ''}
      <div class="bx-sub">Compensate</div>
      <p class="bx-act" title="${tip(w0.actions.join(' '))}">${esc(w0.actions[0])}</p>
    </div>
  </div>

  <div class="ex">
    <div class="ex-kicker">Earth vs Moon · always account for</div>
    <div class="ex-grid">${c.diffs.map(d => { const [what, how] = DIFF_SHORT[d[3]](c.t, c.r, ctx); return `
      <div class="ex-item" title="${tip(d[1] + ' Address by: ' + d[2])}">
        <div class="ex-name"><span class="ex-ic">${DIFF_ICON[d[3]]}</span>${esc(d[0])}</div>
        <div class="ex-what">${esc(what)}</div>
        <div class="ex-how">→ ${esc(how)}</div>
      </div>`; }).join('')}
    </div>
  </div>`;
}


// ══════════════════════════════════════
// EXPERIMENT READINESS — final verdict per experiment for the chosen pair
// ══════════════════════════════════════
const RCLS = { 3: 'v-suff', 2: 'v-part', 1: 'v-avoid', 0: 'v-na', '-1': 'v-na' };
function cannotHTML() {
  const L = META.cannot || [];
  if (!L.length) return '';
  return `<div class="dt-group" style="margin-top:22px"><div class="dt-gtitle">Earth can never reproduce</div>
    <div class="nr-grid">${L.map(c => `<div class="nr-item"><b>${esc(c.condition)}</b>
      <p><i>Moon:</i> ${esc(c.moon || '—')}</p><p><i>Best on Earth:</i> ${esc(c.earth_best || '—')}</p>
      ${c.consequence ? `<p><i>Affects:</i> ${esc(c.consequence)}</p>` : ''}
      ${c.cover_with ? `<p class="nr-how">→ ${esc(c.cover_with)}</p>` : ''}</div>`).join('')}</div></div>`;
}
function experimentsHTML(t, r) {
  if (!t || !r) return `<p class="dt-note" style="margin-top:0">Lock in an Earth place and a Moon target to see the verdict of each of the ${META.experiments.length} field-test experiments (sufficient when every required category meets its minimum, avoid only if a required category is below -0.5, partial otherwise).</p>` + cannotHTML();
  const relIds = new Set(r.relevant || []);
  const rel = META.experiments.filter(x => relIds.has(x.id)).map(x => ({ x, code: SC.ready(r.id, x.id, t.idx) })).sort((a, b) => b.code - a.code);
  const notRel = META.experiments.filter(x => !relIds.has(x.id));
  const count = k => rel.filter(o => o.code === k).length;
  const sv = (c) => SC.score(r.id, c, t.idx);
  const chip = (c, thr, sup) => {
    const v = sv(c), ok = v !== null && v >= thr;
    return `<span class="xr-chip${sup ? ' sup' : ''}${v === null ? '' : ok ? ' ok' : ' low'}" title="${sup ? 'Supporting' : 'Required'}: ${esc(catTitle(c))} at least ${(+thr).toFixed(2)}">${esc(catTitle(c))} ${v === null ? '—' : fmtScore(v)}<em>${sup ? 'helps at' : 'needs'} ≥ ${(+thr).toFixed(2)}</em></span>`;
  };
  const why = o => {
    const req = Object.entries(o.x.required || {});
    if (o.code === 3) {
      const weakSup = Object.entries(o.x.supporting || {}).filter(([c, thr]) => { const v = sv(c); return v !== null && v < thr; }).map(([c]) => catTitle(c));
      return 'Every required category reaches its threshold.' + (weakSup.length ? ` Supporting categories below target: ${esc(weakSup.join(', '))}.` : '');
    }
    if (o.code === 2) {
      const lim = limitingCats(r.id, o.x.id, t.idx);
      return `<b>Held back by:</b> ${lim.map(l => `${esc(catTitle(l.cat))} ${l.score === null ? '—' : fmtScore(l.score)} (needs ≥ ${(+l.need).toFixed(2)})`).join(', ')}.`;
    }
    if (o.code === 1) {
      const neg = req.filter(([c]) => { const v = sv(c); return v !== null && v < 0; }).map(([c]) => `${esc(catTitle(c))} ${fmtScore(sv(c))}`);
      return `<b>Avoid:</b> a required category is below 0 (${neg.join(', ')}).`;
    }
    const miss = req.filter(([c]) => sv(c) === null).map(([c]) => esc(catTitle(c)));
    return `<b>Not available:</b> no score for ${miss.join(', ') || 'a required category'} for this pair.`;
  };
  const card = o => `<div class="xr ${RCLS[o.code]}">
      <div class="xr-head"><span class="xr-name">${esc(o.x.name)}</span><span class="xr-badge">${esc(READY[o.code] ?? 'n/a')}</span></div>
      <div class="xr-req">${Object.entries(o.x.required || {}).map(([c, thr]) => chip(c, thr, false)).join('')}${Object.entries(o.x.supporting || {}).map(([c, thr]) => chip(c, thr, true)).join('')}</div>
      <div class="xr-why">${why(o)}${!isNa(r.n_sufficient && r.n_sufficient[o.x.id]) ? ` <span class="cx-dim">${(+r.n_sufficient[o.x.id]).toLocaleString('en-US')} of ${(META.n_tiles || allTiles.length).toLocaleString('en-US')} Earth tiles are sufficient for this target.</span>` : ''}</div>
      ${(o.x.also || []).length ? `<details><summary>Also account for (${o.x.also.length})</summary><ul>${o.x.also.map(a => `<li>${esc(a)}</li>`).join('')}</ul></details>` : ''}
    </div>`;
  return `<div class="xr-sum">
      <span class="v-suff">${count(3)} sufficient</span><span class="v-part">${count(2)} partial</span><span class="v-avoid">${count(1)} avoid</span>${count(0) ? `<span class="v-na">${count(0)} not available</span>` : ''}
    </div>
    <div class="xr-list">${rel.map(card).join('')}</div>
    ${notRel.length ? `<details class="xr-nr"><summary>Not relevant for this Moon target (${notRel.length})</summary><ul>${notRel.map(x => `<li>${esc(x.name)}</li>`).join('')}</ul></details>` : ''}
    <div class="dt-note">Required categories (solid) decide the verdict: sufficient when all reach their threshold, avoid when any is below 0, partial otherwise. Supporting categories (dashed) only add caveats.</div>
    ${cannotHTML()}`;
}


// ══════════════════════════════════════
// METHOD & RELIABILITY
// ══════════════════════════════════════
function qualityHTML() {
  const cats = META.categories || [];
  const nA = cats.reduce((a, c) => a + (+c.n_anchors || 0), 0), nF = cats.reduce((a, c) => a + (+c.flagged || 0), 0);
  const cell = v => isNa(v) ? '<td class="v na">—</td>' : `<td class="v">${esc(v)}</td>`;
  return `<p class="q-big">Scores are calibrated against ${nA} expert anchors (pairs with a judged score). All of them are drafts awaiting review by a site expert (needs_review = 1), so the scale of every category can still move.${nF ? ` ${nF} anchor${nF > 1 ? 's disagree' : ' disagrees'} with the model by more than 0.3.` : ''}</p>
    <div class="dt-wrap"><table class="dt q-tbl">
      <tr><th>Category</th><th>Anchors</th><th>Leave-one-out error</th><th>Rank agreement</th><th>Cap</th><th>In overall score</th></tr>
      ${cats.map(c => `<tr><td class="p">${esc(c.label)}<small>${esc(c.how || '')}</small></td>${cell(c.n_anchors)}${cell(fmt(c.loo_mae, 2))}${cell(fmt(c.loo_rank ?? c.rank, 2))}${cell(isNa(c.cap) ? null : '+' + (+c.cap).toFixed(1))}${cell(c.in_composite ? (c.id === 'caves' ? 'pit only' : 'yes') : 'no')}</tr>`).join('')}
    </table></div>
    <div class="dt-note">Leave-one-out error: mean absolute error (on the −1..+1 scale) when each anchor is predicted by a fit made without it. Rank agreement: Spearman correlation with the anchors. Cap: the highest score a category can reach, because Earth never fully reproduces it. Bands: ≥ 0.5 sufficient, 0–0.5 partial, &lt; 0 avoid.</div>
    ${(META.limitations || []).length ? `<div class="dt-group" style="margin-top:18px"><div class="dt-gtitle">Known limitations</div><ul class="cx-list">${META.limitations.map(l => `<li>${esc(l)}</li>`).join('')}</ul></div>` : ''}
    ${META.generated ? `<div class="dt-note">Scores generated ${esc(META.generated)} · ${(META.n_targets ?? SC.ids.length)} Moon targets × ${(META.n_tiles ?? allTiles.length).toLocaleString('en-US')} Earth tiles.</div>` : ''}`;
}


// ══════════════════════════════════════
// PAGE RENDERING
// ══════════════════════════════════════
const TILES = [
  { id: 'data', cls: '', title: 'Compared data', sub: 'Earth vs Moon values and units behind each category' },
  { id: 'exp', cls: 'cc-x', title: 'Experiment readiness', sub: 'Sufficient / partial / avoid for each field test, with caveats' },
  { id: 'concl', cls: 'cc-t', title: 'Conclusions', sub: 'Overview, strongest match, main caution, Earth vs Moon' },
  { id: 'qual', cls: 'cc-q', title: 'Method & reliability', sub: 'Calibration quality, caps and known limitations' }
];

let cmpKey = '';
function renderCompare() {
  const e = selectedEarthTile, m = selectedMoonTarget;
  const empty = document.getElementById('cmp-empty'), dash = document.getElementById('cmp-dash');
  empty.hidden = true; dash.hidden = false;          // the layout is always shown; bars stay empty until both places are chosen
  const key = (e ? e.idx : '-') + '|' + (m ? m.id : '-');
  if (key === cmpKey) return;            // same selection: keep the page (and open sections) as they are
  cmpKey = key;
  const openTiles = new Set([...dash.querySelectorAll('.tile.on')].map(x => x.dataset.panel));   // keep opened panels across pair changes
  const ready = !!(e && m);
  const s = ready ? getSimilarityScores(e, m) : null;
  const cell = v => isNa(v) ? '<td class="v na">—</td>' : `<td class="v">${esc(v)}</td>`;
  const safe = f => { try { return f(); } catch (err) { console.warn('compared value failed', err); return null; } };
  const outReason = c => c.invariant ? 'same for every Moon target' : 'no Earth–Moon overlap';

  const subOf = c => {
    const parts = [];
    const v = ready ? s[c.key] : null;
    if (c.key === 'caves' && (!m || !m.pit)) return 'only scored for the Tranquillitatis pit';
    if (ready && c.key === 'thermal') { const p = SC.thermalPct(m.id, e.idx); if (p !== null) parts.push(`colder than ${p}% of Earth sites`); }
    if (ready && c.key === 'illumination') { const mo = SC.twinMonth(m.id, e.idx); parts.push(mo ? `best light in ${MONTHS[mo]}` : 'no light-twin month'); }
    if (!c.inComp) parts.push(`not in overall score · ${outReason(c)}`);
    else if (m && m.composite_weights && !isNa(m.composite_weights[c.key])) {
      const w = +m.composite_weights[c.key];
      parts.push(w > 0 ? `${Math.round(w * 100)}% of overall score` : 'not in overall score for this target');
    }
    if (ready && v === null && c.key !== 'caves') parts.push('no score for this pair');
    return parts.join(' · ');
  };
  const paramList = k => `<ul class="sd-params">${(SIM_DATA[k] || []).map(([n, u, , , d]) => `<li><b>${esc(n)}</b> <span class="sd-unit">${esc(u)}</span><br>${esc(d)}</li>`).join('')}</ul>`;
  const rowHTML = (c, v, sub, detail, extra = '') => {
    const has = ready && !isNa(v), val = has ? Math.max(-1, Math.min(1, v)) : 0, sign = val >= 0 ? 'pos' : 'neg';
    return `<div class="sp-row ${extra}" style="--c:${c.rgb}">
      <button class="sp-main" aria-expanded="false">
        <span class="sp-name"><i class="sp-chev"></i><span>${esc(c.title)}${sub ? `<em class="sp-sub">${esc(sub)}</em>` : ''}</span></span>
        <span class="sp-track"><i class="sp-bar ${sign}" data-v="${val}"></i></span>
        ${has ? `<span class="sp-val ${sign}">${fmtScore(val)}<small>${verdictOf(val)}</small></span>` : '<span class="sp-val na">—</span>'}
      </button>
      <div class="sp-body"><div class="sp-inner"><div class="sp-detail"><span></span><div>${detail}</div></div></div></div>
    </div>`;
  };
  const W = m && m.composite_weights ? Object.entries(m.composite_weights).filter(([, w]) => +w > 0).sort((a, b) => b[1] - a[1]) : [];
  const overall = rowHTML({ title: 'Overall match', rgb: CAT_RGB.composite }, ready ? s.composite : null,
    ready ? `${overallVerdict(m.id, e.idx)} for ${m.name} (from its experiment verdicts)` : 'experiment-weighted mean of the comparable categories',
    `<p class="sd-brief">The overall score for this Moon target: an experiment-weighted mean of the categories that actually compare with this specific Moon place (terrain, geology, illumination and water/ice, plus caves for the Tranquillitatis pit). Each category is weighted by how many relevant experiments require it (1.0) or use it in support (0.5). A category with no score takes the target's median. Operations is left out because it is built from the terrain and ice scores, which would count them twice. Temperature, dust and sterility are scored and drive experiment verdicts, but are not averaged in because they are identical across targets or have no real overlap. Finally the overall score is capped at 0.8 − 2 × the share of the tile under plants (including moss and lichen), open water or towns, so a tile with 40% or more such cover is at most 0.</p>
     ${W.length ? `<ul class="sd-params">${W.map(([k, w]) => `<li><b>${esc(catTitle(k))}</b> <span class="sd-unit">${Math.round(w * 100)}% weight</span><br>${ready && !isNa(s[k]) ? 'score ' + fmtScore(s[k]) + ' · ' + verdictOf(s[k]) : 'no score for this pair'}</li>`).join('')}</ul>` : ''}`,
    'sp-overall');
  const catRow = c => rowHTML(c, ready ? s[c.key] : null, subOf(c),
    `<p class="sd-brief">${esc(c.brief)}</p><p class="sd-method"><b>Method:</b> ${esc(c.how)}${isNa(c.cap) ? '' : ` Capped at +${(+c.cap).toFixed(1)}.`}</p>${paramList(c.key)}`);
  const inC = SIM_CATS.filter(c => c.inComp), outC = SIM_CATS.filter(c => !c.inComp);
  const tileSub = id => {
    if (id === 'exp' && ready) { const rel = (m.relevant || []).length, n3 = (m.relevant || []).filter(x => SC.ready(m.id, x, e.idx) === 3).length; return `${n3} of ${rel} relevant experiments sufficient for this pair`; }
    return TILES.find(x => x.id === id).sub;
  };

  dash.innerHTML = `
    <div class="sd-head">
      <div>
        <div class="sd-title">Similarity Assessment</div>
        <div class="sd-pair">
          ${e ? `<span class="sd-place" data-go="earth" title="Change Earth place">🌍 <b>${esc(e.name)}</b>${e.country ? ' · ' + esc(e.country) : ''}</span>`
               : '<span class="sd-place pending" data-go="earth" title="Go to Earth">🌍 Choose an Earth place</span>'}
          <span class="sd-vs">⟷</span>
          ${m ? `<span class="sd-place" data-go="moon" title="Change Moon target">🌙 <b>${esc(m.name)}</b> · ${m.pit ? 'Mare Tranquillitatis' : 'South Pole'}</span>`
               : '<span class="sd-place pending" data-go="moon" title="Go to Moon">🌙 Choose a Moon target</span>'}
        </div>
      </div>
      ${!ready ? '<span class="sd-pill wait">Waiting for both places</span>' : `<span class="sd-pill" title="${esc((META.limitations || [])[0] || '')}">Draft calibration · anchors await expert review</span>`}
    </div>

    <div class="sp">
      <div class="sp-axis"><span></span><div class="sp-ticks"><span>−1 avoid</span><span>0</span><span>+1 analog</span></div><span></span></div>
      ${overall}
      ${inC.map(catRow).join('')}
      ${outC.length ? `<div class="sp-sep">Scored, but not part of the overall match</div>${outC.map(catRow).join('')}` : ''}
      <div class="sp-foot">${ready ? 'Score ≥ 0.5 sufficient · 0 to 0.5 partial · below 0 avoid. Click a category to see what it measures and which parameters it uses.' : 'Lock in a place on Earth and a target on the Moon to see the scores. Click a category to see what it measures.'}</div>
    </div>

    <div class="sd-tiles">
      ${TILES.map(x => `<button class="tile ${x.cls}" data-panel="${x.id}"><span><div class="tile-t">${x.title}</div><div class="tile-s">${esc(tileSub(x.id))}</div></span><i class="acc-chev"></i></button>`).join('')}
    </div>

    <div class="panel-wrap" id="pnl-data"><div class="panel-in"><div class="panel">
      ${SIM_CATS.map(c => `<div class="dt-group" style="--c:${c.rgb}"><div class="dt-gtitle">${esc(c.title)}</div><div class="dt-wrap"><table class="dt">
        <tr><th>Parameter</th><th>Unit</th><th>🌍 ${e ? esc(e.name) : 'Earth'}</th><th>🌙 ${m ? esc(m.name) : 'Moon'}</th></tr>
        ${(SIM_DATA[c.key] || []).map(([p, u, fe, fm]) => `<tr><td class="p">${esc(p)}</td><td class="u">${esc(u)}</td>${cell(e ? safe(() => fe(e, m)) : null)}${cell(m ? safe(() => fm(m, e)) : null)}</tr>`).join('')}
      </table></div></div>`).join('')}
      <div class="dt-note">Earth: one 10 × 10 km analog tile. Moon: ${m ? esc(m.label.toLowerCase()) + (m.kind === 'zone' ? ` (${fmt(m.area_km2, 1) ?? '—'} km²); terrain describes the 10 km window around the zone centre, as used in scoring, and the operations slope hazard uses the zone pixels themselves` : '; all values describe the 10 km window') : 'the chosen target'}. The light-twin month is the Earth month whose noon sun is closest to the Moon target's. “—” = not available for that body in the current dataset.</div>
    </div></div></div>

    <div class="panel-wrap" id="pnl-exp"><div class="panel-in"><div class="panel cc-x">
      ${experimentsHTML(e, m)}
    </div></div></div>

    <div class="panel-wrap" id="pnl-concl"><div class="panel-in"><div class="panel cc-p">
      ${ready ? conclusionsHTML(buildConclusions(e, m, s), { moon: m.name }) : `      <div class="cc-grid">
        <div class="cc"><h4>✦ Analog value for lunar research</h4><p>What this place can help us test or study for the Moon: the analog characteristics and the experiment environment it offers. Generated from the scores once both places are chosen.</p><div class="skel" style="width:92%"></div><div class="skel" style="width:78%"></div><div class="skel" style="width:85%"></div></div>
        <div class="cc neg"><h4>⚠ Limitations &amp; compensation</h4><p>What differs from the lunar target and what an experiment would need to compensate for. Generated from the scores once both places are chosen.</p><div class="skel" style="width:88%"></div><div class="skel" style="width:70%"></div><div class="skel" style="width:81%"></div></div>
      </div>`}
    </div></div></div>

    <div class="panel-wrap" id="pnl-qual"><div class="panel-in"><div class="panel cc-q">
      ${qualityHTML()}
    </div></div></div>`;

  dash.querySelectorAll('.sp-main').forEach(b => b.addEventListener('click', () => {
    const open = b.parentElement.classList.toggle('open');
    b.setAttribute('aria-expanded', open);
  }));
  dash.querySelectorAll('.tile').forEach(t => {
    if (openTiles.has(t.dataset.panel)) { t.classList.add('on'); document.getElementById('pnl-' + t.dataset.panel).classList.add('on'); }
    t.addEventListener('click', () => {
      const on = t.classList.toggle('on');
      document.getElementById('pnl-' + t.dataset.panel).classList.toggle('on', on);
    });
  });
  dash.querySelectorAll('[data-go]').forEach(b => b.addEventListener('click', () => goTo(b.dataset.go)));
  growSpectrum();
}

// Bars grow out from the zero line; replayed each time the Assessment tab opens
function growSpectrum() {
  const bars = document.querySelectorAll('#cmp-dash .sp-bar');
  bars.forEach(b => { b.style.transition = 'none'; b.style.width = '0'; b.style.left = '50%'; });
  void document.body.offsetWidth;
  bars.forEach(b => { b.style.transition = ''; });
  requestAnimationFrame(() => requestAnimationFrame(() => bars.forEach(b => {
    const v = +b.dataset.v;
    b.style.width = Math.abs(v) * 50 + '%';
    if (v < 0) b.style.left = (50 + v * 50) + '%';
  })));
}
