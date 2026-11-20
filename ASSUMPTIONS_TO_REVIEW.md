# Assumptions to review: data prep, HMM runs and SST matching

Purpose: a checklist for an independent review of this pipeline. Each item states an assumption the code makes,
where it lives, and what could go wrong. Please verify each one against the code and data, and report anything
that is wrong, risky, or undocumented. Line numbers drift; search by function or setting name.

Run R scripts from the project folder (`Rscript pipeline_scripts/<name>.R`); Python scripts resolve paths from
their own location. Use `/usr/bin/python3` (the project `.venv` lacks pandas).

All counts are from the 19 Nov 2026 run (Tema-to-Tema trips only, main fit with 7 starts) unless marked otherwise.

**Changes on 19 Nov 2026:** only trips that start and end in Tema are kept (1.11); the main fit uses 7 starts
instead of 10 (3.8); Meteosat-12 SST is matched to every ping (section 5). The pipeline was rerun from step 1.

**Changes on 15 Nov 2026** (after the review in "Ghana HMM pipeline verified errors"): `pars` restored in
`fit_hmm.R`; depth grid fixed and `fisherDepth/` depth redone (0.4); speed capping replaced by a spike filter
(1.8); crawl seeded (2.9); fishing vs near-stationary check added (3.12); Pi_min sweep now 0.5–0.9 (3.15).

## Pipeline order

| Step | Script | Output |
|---|---|---|
| 0 (upstream, archived) | `archive_old_code/build_fisher_depth.py` | `fisherDepth/*_depth.csv` (raw pings + depth) |
| 0b | `pipeline_scripts/update_fisher_depth.py` | redoes `depth_m` and `on_land` in `fisherDepth/` in place |
| 1 | `pipeline_scripts/build_trip_tracks.py` | `pipeline_outputs/gps_tracks/trip_tracks.csv`, `dropped_non_tema_trips.csv` |
| 2 | `pipeline_scripts/interpolation_script.R` | `pipeline_outputs/gps_tracks/regularised_tracks.csv` |
| 3 | `pipeline_scripts/fit_hmm.R` (main), `fit_hmm_step_only.R`, `fit_hmm_angle_only.R`, `fit_hmm_no_interp.R` | `pipeline_outputs/hmm_results/hmm_fit*.rds`, `hmm_states*.csv`, sweep CSVs |
| 4 | `fishing_events.R`, `plot_state_sweep.R`, `plot_fishing_min_length.py`, `plot_hmm_tracks.py` | event tables, `result_figures/` |
| 5 (independent of 2-4) | `pipeline_scripts/match_sst_meteosat12.py` | `data/sst_meteosat12/*.nc` (hourly crops), `pipeline_outputs/environment/ping_sst_meteosat12.csv` |

Data files (CSV, RDS, XLSX, NetCDF) are git-ignored and exist only locally.

---

## 0. Upstream raw data (`archive_old_code/build_fisher_depth.py`, produces `fisherDepth/`)

0.1 **Raw exports are parsed by column position.** Columns are renamed positionally to `events, timestamp, address, lat_lng, speed, heading, ...`. A different column order silently mislabels fields; a different number of columns raises an error. Skipped files (errors, or fewer than 100 rows with a position) are printed with the reason (since 15 Nov).

0.2 **Timestamps are naive (no time zone).** Everything downstream treats them as UTC, including the SST hour matching (section 5). Ghana is UTC+0, so this is fine only if the exports log in UTC/Ghana time. This cannot be checked from the data; check the export settings or tracker vendor.

0.3 **Speed.** Rows with a missing speed are **dropped** (`dropna(subset=["lat_lng", "speed"])`); only non-numeric speeds become 0. The unit is **knots** (tracker speed ÷ GPS-derived km/h = 0.539). It contains junk values (max 3,883). Nothing downstream uses it. The `heading` column is empty in every export and every `fisherDepth/` file.

0.4 **Depth grid (fixed 15 Nov).** `loadDepthGrid` used to rebuild latitude from a 0.015625° step anchored at row 239 = 5.0° N; that was 0.2578° (about 29 km) too far north (median depth error about 380 m). It now reads latitude from `data/YC.data` (1.008–7.008° N). `update_fisher_depth.py` redid `depth_m` and `on_land` in all 25 `fisherDepth/` files in place (positions and other columns unchanged). Depth changed for 120,193 of 124,513 pings; `on_land` went from 1 to 3,778 pings. Checked against Natural Earth 10m land: 3,453 of the 3,778 are on land there too (325 are coastal sea pings in a coarse 1/64°, about 1.7 km, land cell). 1,465 are within 1 km of Tema port; most others are at beach landing sites (largest: around 5.39° N, 0.49° W, mostly 020 GodFirst), plus a few truly inland pings. Only 1 ping the grid calls sea is more than 2 km inland, so the corrected grid lines up with the coastline. `on_land` marks pings at landing sites, not GPS errors. No HMM script uses depth.

0.5 **`on_land` means depth == 0 in the grid.** Flagged but **not removed** anywhere downstream. 0 pings are outside the grid.

0.6 **`fisherDepth/` was not built from the current `Exported Files/`.** Boat names differ (016 GreatProvider vs Sabbath2, 017 GoodFather vs AyeYi, 021 Otsokobila vs Apostolic, 022 EnsoNyameYe2 vs VictoryNo9No10, 012 IfMenWereGod vs IfMenWereGod_1); 002, 015 and 018 have exports but no depth file; the exports end 20 Apr 2026 but `fisherDepth/` runs to 27 May 2026; and `fisherDepth/` has no `outside_grid` column, so a different script version wrote it. Step 0 cannot be reproduced from the files present. **Open question:** are 016, 017, 021 and 022 the same physical boats under new names, or were trackers moved between boats?

## 1. Trips and segments (`pipeline_scripts/build_trip_tracks.py`)

1.1 **Trips are defined only by Tema port.** A trip is a run of pings more than `TEMA_PORT_RADIUS_KM = 1.0` from Tema (5°38'14.6" N, 0°00'58.5" E). Boats based at other landing sites (019 Emmanuel, 016 GreatProvider, 022 EnsoNyameYe2 never enter the radius; 014 RoyalDiadem and 017 GoodFather rarely do) are now removed by the Tema filter (1.11).

1.2 **The 1 km port radius may be too tight.** Several boats have 2-6 pings just west of Tema (about 5.61° N, 0.06° W) before being detected as in port.

1.3 **A trip's end row is the first ping back inside the port** (`detect_tema_trips`), so each kept trip includes one in-port ping at its end. The 14 single-ping trips of the full split are all dropped by the Tema filter; none remain.

1.4 **Trips are split at gaps between pings longer than `MAX_TRIP_GAP_DAYS = 4`** (`split_at_long_gaps`). The piece after a gap did not start in Tema, so the Tema filter drops it. Full split: 228 trips (177 `port`, 31 `gap`, 20 `end_of_data`); all 150 kept trips are `port`. Longest kept trips: 011 t2 31.8 days, 006 t2 22.9, 020 t4 12.7.

1.5 **Shared outage.** All trackers went silent from 11 Feb 2026 23:32:21 to 16 Apr 2026 00:00:35. No trip spans it.

1.6 **Segments are split at gaps longer than `MAX_GAP_MINUTES = 30`.** Segments are the unit for crawl and the HMM. 1,702 segments in the kept trips. 12.8% of within-segment intervals are 5.5-30 minutes and get filled by crawl (median interval 5.0 min, 90th percentile 9.3).

1.7 **Projection: UTM zone 31N (EPSG:32631) for all positions.** Measured UTM ÷ geodesic step ratio (15 Nov, all trips): 1.0069 at 6–3° W, 1.0018 at 3–0° W, 1.0007 at 0–2° E. Small for the HMM; a local equidistant projection would remove it.

1.8 **GPS spikes are removed, not moved (since 15 Nov; `drop_speed_spikes`).** The filter runs per boat on raw pings, before trips are built, and never moves a ping:
- After a step faster than `MAX_SPEED_KMH = 30`, it looks up to `MAX_SPIKE_PINGS = 3` pings ahead for a ping that (a) is reached by a fast step, (b) is reachable directly from the last good ping at ≤ 30 km/h (or ≤ half the jump speed), and (c) makes the pings between a detour at least `MIN_DETOUR_RATIO = 2` times the direct distance. The pings in between are removed.
- Pings under `MIN_PING_SECONDS = 60` apart with a fast step between are near-duplicates; the one farther from the next ping is removed. A bad first or last ping is removed when the step inward is normal.
- 100 pings removed over all boats (66 single, 8 pairs, 6 triples). In the kept trips, 21 steps inside segments are still faster than 30 km/h; they are kept because they are not clear out-and-back jumps. Check whether those are real.
- The old method (`cap_implausible_speeds`) pulled fast pings back from the previous *capped* point and dragged real fast tracks by up to 29 km (e.g. `OMI_Fisher_017_GoodFather_t4_s5`).

1.9 **Rows with missing timestamp, lat or lon are dropped.** `trip_tracks.csv` has no duplicate timestamps within a segment.

1.10 **No filtering of `on_land` pings.**

1.11 **Only Tema-to-Tema trips are modelled (since 19 Nov).** A trip is kept only if it starts by leaving Tema (`started_from_tema`: the ping just before the trip is inside the port radius) **and** ends back in Tema (`returned_to_tema`). Of 228 trips, 150 are kept (78,153 pings, 22 boats); 78 are dropped (44,465 pings, 36.3%) and written to `pipeline_outputs/gps_tracks/dropped_non_tema_trips.csv`: 35 did not end in Tema (cut off by a > 4-day gap or the end of the data), 27 did not start in Tema (mostly pieces after a long gap), 16 neither. Boats 016, 019 and 022 are excluded entirely, and 014 (99.9%) and 017 (97.6%) almost entirely, so boats based at other landing sites are not represented. A trip that leaves Tema, stops at another landing site and later returns still counts as one trip. Trip numbers come from the full split, so kept trips can skip numbers (e.g. t1, t3).

## 2. Interpolation to 5 minutes (`pipeline_scripts/interpolation_script.R`, crawl)

2.1 **Duplicate timestamps within a segment: the first row is kept.** There are none in the current data.

2.2 **Segments with fewer than `MIN_POINTS = 10` pings are skipped.** 753 of 1,702 segments skipped (949 kept).

2.3 **Near-stationary segments are skipped:** `max(sd(x), sd(y)) < STATIONARY_SD_M = 10` m. 4 skipped (2 from 023 YehowaDzi, 1 from 013 John316, 1 from 008 Malachie3). This may bias against the near-stationary state.

2.4 **Movement model: a continuous-time correlated random walk with constant parameters per segment** (`mov.model = ~1`, `time.scale = "hours"`) and **no location-error model**, so GPS error goes straight into steps and angles.

2.5 **Fits are accepted only if converged with finite standard errors.** Failing segments get up to 3 extra passes, then are dropped; their IDs are printed in the log but not saved to a file. 29 segments needed a second pass, 1 a third, 0 failed (945 fitted).

2.6 **The prediction grid starts at each segment's first ping and steps exactly 5 minutes.** Not aligned to clock time.

2.7 **The posterior mean position (`mu.x`, `mu.y`) is used, not a simulated path.** Positions inside gaps are smooth averages. 14,249 of 88,854 5-minute positions (16.0%) have no real ping within ±2.5 min.

2.8 **The original observed rows ("o") are discarded**; only grid predictions ("p") are kept.

2.9 **Crawl is seeded (since 15 Nov, `SEED = 42`).** Before, unseeded random starts made reruns differ: on identical input, 519 of 1,589 segments moved by more than 1 m and 9 by more than 1 km (worst 30 km, `OMI_Fisher_001_MessiahTanker3_t13_s16`). A seed makes reruns repeatable but does not show which optimum is better for such segments; check them.

## 3. HMM fits (`pipeline_scripts/fit_hmm.R`, with variants)

3.1 **Segments need at least `MIN_STEPS = 20` steps.** 181 of 945 crawl segments dropped (2,833 positions, 3.2%); 764 segments and 85,257 steps remain (86,021 positions).

3.2 **Coordinates are converted to km, so steps are km per 5 minutes.** `prepData(type = "UTM")`. Angle is NA at each segment's ends and after zero-length steps. Inherits 1.7.

3.3 **Zero-length steps are replaced with 1 m** (`ZERO_STEP_KM`) because the gamma cannot take 0. 48 (interpolated), 893 (no interpolation). Alternative: `zeroInflation = list(step = TRUE)`.

3.4 **Exactly 3 states.** Not tested against 2 or 4 states with AIC/BIC or pseudo-residuals.

3.5 **Gamma step lengths and von Mises turning angles** with the **mean angle fixed at 0** (`estAngleMean = FALSE`).

3.6 **Constant transition probabilities.** No covariates (time of day, distance from shore, depth, SST). The HMM uses only step length and turning angle from positions.

3.7 **Segments are independent tracks** sharing one set of parameters; no boat or trip random effects.

3.8 **Starting values and seed.** Random around the 20th/50th/90th step percentiles; concentrations U(0.5, 3) sorted upward, so state 1 (slowest) *starts* as the most tortuous. The fit makes the slow state straight instead; all starts agree, so this is minor. Seed 42; 7 starts for the main fit (10 before 19 Nov), 3 for the variants.

3.9 **A fit is "valid"** if the `nlm` code is 1 or 2, the likelihood is finite and `stateProbs` runs. Best = lowest negative log-likelihood. `retryFits = 0`. All 7 main starts were valid and reached log-likelihood 61,652.7.

3.10 **Boundary transitions.** Main fit: near-stationary→transit and transit→near-stationary are at −22.4 and −24.5 on the working scale (probability about 0). The Hessian has a negative eigenvalue (−0.016), so standard errors are not reliable. The no-interpolation fit has the same problem (−20.7, −23.0; eigenvalue −0.0075). Step-only and angle-only do not (smallest eigenvalues 30.1 and 7.2). In the 15 Nov all-trips fit the Hessian was positive definite, so this flips between runs.

3.11 **State numbers are arbitrary.** Roles are assigned by mean step length: slowest = near-stationary, **middle = fishing**, fastest = transit. Main fit: fishing 0.190 km/5 min (1.23 kn, κ 1.50), near-stationary 0.165 (1.07 kn, κ 104), transit 0.779 (5.05 kn, κ 127). Fishing and near-stationary differ mainly in turning, not speed.

3.12 **Label checks.** The fit stops if fishing is not more tortuous than transit, or not more tortuous than the slow state (a fishing/near-stationary swap). Both pass.

3.13 **"Near-stationary" is not stationary.** It averages about 1 kn with κ about 104 and never switches directly to or from transit. It looks like slow directed movement or drifting; consider renaming it.

3.14 **Viterbi vs probabilities.** The Pi_min sweeps and the track maps (`plot_hmm_tracks.py`) use the probabilities; `fishing_events.R` uses Viterbi, and `plot_fishing_min_length.py` uses Viterbi only to assign roles.

3.15 **The Pi_min sweep runs from 0.5 to 0.9.** Pi_min = 1.0 was dropped: it is reached only through numerical underflow, and only by the low-κ state (8,087 positions of the fishing state hit exactly 1; 0 for the others), so it was biased toward fishing.

### Variants

3.16 **`fit_hmm_step_only.R`:** step length only; fishing = middle mean step. Its "near-stationary" is really stationary (0.049 km/5 min; fishing 0.209, transit 0.779), so its fishing state is roughly fishing plus near-stationary from the main fit. No angle check.

3.17 **`fit_hmm_angle_only.R`:** turning angle only; fishing = lowest concentration (κ 1.07, 29.2, 405). Cannot separate stationary from transit.

3.18 **`fit_hmm_no_interp.R`:** raw pings snapped to the crawl 5-minute grid (nearest slot, ±2.5 min; the closer ping wins a shared slot; empty slots NA). Segments come from `regularised_tracks.csv`. 64,318 steps; 72,341 of 86,021 slots (84.1%) have a ping; 217 pings lost to shared slots. Fishing 0.188 km/5 min (κ 1.48), near-stationary 0.166 (κ 85.4), transit 0.781 (κ 99.5). Its log-likelihood is not comparable with the interpolated fit.

The sweep panels therefore compare different state definitions across models.

## 4. Fishing events and plots

4.1 **`fishing_events.R`: gap fill first, then minimum length**, within each segment (they cannot join across segments). `MAX_GAP_FIXES = 1`, `MIN_EVENT_FIXES = 2`. Applied to Viterbi labels and to each Pi_min.

4.2 **`plot_fishing_min_length.py`: `PI_MIN = 0.75`, n = 1-6, same gap fill.** Roles come from the empirical mean `step_km` per Viterbi state (main 0.189 / 0.165 / 0.779), which gives the same order as the fitted gamma means. The angle-only model's roles come from κ (`plot_state_sweep.R`); this script uses only the main model. At Pi_min ≥ 0.5 at most one state can qualify. Shares at n = 1: fishing 29.2%, near-stationary 31.4%, transit 34.6%, below Pi_min 4.9%; 3,250 fishing events.

4.3 **`plot_hmm_tracks.py` samples 5 trips with at least 40 steps (seed 42)**, coloured by role at Pi_min 0.75. Regenerated on 19 Nov; trip IDs changed with the Tema filter, so the sampled trips differ from earlier versions.

4.4 **`plot_state_sweep.R` finds each model's `.rds` from its states-file name by regex.** Shares are over all positions, including empty slots in the no-interpolation model.

## 5. Sea surface temperature (`pipeline_scripts/match_sst_meteosat12.py`)

5.1 **Source: OSI SAF OSI-206-b, Meteosat-12 (MTG-I1) FCI L3C sub-skin SST**, hourly, 0.05° grid (EUMETSAT Data Store collection `EO:EUM:DAT:0694`). The product is listed as pre-operational. It is **sub-skin** SST (about 1 mm depth), not bulk or depth-resolved temperature. MTG-S1 / Meteosat-13 has no SST product for these dates.

5.2 **Only hours with a ping are downloaded**, and each file is cropped to 2–7° N, 6° W–2° E before saving (`data/sst_meteosat12/`); the full file is deleted. 3,207 hours on 155 days needed; 3,186 downloaded; 21 hours have no product in the Data Store (listed in `missing_hours.csv`).

5.3 **Matching: nearest hour, nearest grid cell.** Each ping in `trip_tracks.csv` (raw pings, not the 5-minute HMM positions) gets the file whose nominal hour is closest to its timestamp (always within 30 min; each hourly file covers about ±30 min of its nominal hour) and the SST of the nearest 0.05° cell (about 5.5 km). No spatial or temporal interpolation, and no filling of cloudy pixels from nearby cells or hours. The file hour comes from the product name, and ping times are assumed to be UTC (0.2).

5.4 **Quality.** Every pixel with an SST value is kept. `low_quality = True` where GHRSST `quality_level` < 4 (`MIN_QUALITY`), `False` for levels 4–5, empty where there is no SST value. `sst_c` = `sea_surface_temperature − sses_bias` in °C (bias-corrected); `sst_raw_c` is uncorrected.

5.5 **Coverage.** 99.7% of pings have a file for their hour; 54.1% have an SST value (41.1% good quality, 13.0% low quality); the rest are mostly cloud. Good-quality coverage drops in April–May (24–30% of pings) against 38–53% in other months, likely from the rainy season. Missing SST is therefore not random in time, and possibly not in weather or behaviour; check for bias before comparing SST between states.

5.6 **Coastal pixels.** Pings near the shore (landing sites, Tema port) can fall in cells partly over land; the quality level should flag most of these, but pings close to shore deserve a check.

5.7 **Good-quality SST** (bias-corrected): mean 28.6 °C, range 24.9–32.4 °C, interquartile 27.8–29.5 °C.

## 6. Cross-cutting checks still to do

- Pseudo-residuals (`pseudoRes`) on the main fit; 2 vs 3 vs 4 states by AIC/BIC (3.4).
- Which crawl optimum is right for segments that moved by kilometres between unseeded runs (2.9).
- Boat identities for 016, 017, 021, 022 and the export behind `fisherDepth/` (0.6).
- Time zone of the raw exports (0.2), which also affects the SST hour matching.
- Whether the 21 remaining fast steps are real (1.8).
- Whether SST gaps (cloud) bias any comparison of SST between HMM states (5.5).
