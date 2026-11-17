# Assumptions to review: data prep and HMM runs

Purpose: a checklist for an independent review of this pipeline. Each item states an assumption the code makes,
where it lives, and what could go wrong. Please verify each one against the code and data, and report anything
that is wrong, risky, or undocumented. Line numbers drift; search by function or setting name.

Run R scripts from the project folder (`Rscript pipeline_scripts/<name>.R`); Python scripts resolve paths from
their own location. Use `/usr/bin/python3` (the project `.venv` lacks pandas).

**Changes on 15 Nov 2026** (after the review in "Ghana HMM pipeline verified errors"): `pars` restored in
`fit_hmm.R` (C1); depth grid fixed and `fisherDepth/` depth redone (0.4); speed capping replaced by a spike
filter (1.8); crawl seeded (2.9); fishing vs near-stationary check added (3.12); Pi_min sweep now 0.5–0.9
(3.15). The whole pipeline was rerun from step 1 on 15 Nov; all counts below are from that run unless marked otherwise.

## Pipeline order

| Step | Script | Output |
|---|---|---|
| 0 (upstream, archived) | `archive_old_code/build_fisher_depth.py` | `fisherDepth/*_depth.csv` (raw pings + depth) |
| 0b | `pipeline_scripts/update_fisher_depth.py` | redoes `depth_m` and `on_land` in `fisherDepth/` in place |
| 1 | `pipeline_scripts/build_trip_tracks.py` | `pipeline_outputs/gps_tracks/trip_tracks.csv` |
| 2 | `pipeline_scripts/interpolation_script.R` | `pipeline_outputs/gps_tracks/regularised_tracks.csv` |
| 3 | `pipeline_scripts/fit_hmm.R` (main), `fit_hmm_step_only.R`, `fit_hmm_angle_only.R`, `fit_hmm_no_interp.R` | `pipeline_outputs/hmm_results/hmm_fit*.rds`, `hmm_states*.csv`, sweep CSVs |
| 4 | `fishing_events.R`, `plot_state_sweep.R`, `plot_fishing_min_length.py`, `plot_hmm_tracks.py` | event tables, `result_figures/` |

Data files (CSV, RDS, XLSX) are git-ignored and exist only locally.

---

## 0. Upstream raw data (`archive_old_code/build_fisher_depth.py`, produces `fisherDepth/`)

0.1 **Raw exports are parsed by column position.** Columns are renamed positionally to `events, timestamp, address, lat_lng, speed, heading, ...`. A different column order silently mislabels fields; a different number of columns raises an error. Skipped files (errors, or fewer than 100 rows with a position) are now printed with the reason (15 Nov); before, they were skipped silently.

0.2 **Timestamps are naive (no time zone).** Everything downstream treats them as UTC. Ghana is UTC+0, so this is fine only if the exports log in UTC/Ghana time. This cannot be checked from the data (trip start hours spread across the whole day); check the export settings or tracker vendor.

0.3 **Speed.** Rows with a missing speed are **dropped** (`dropna(subset=["lat_lng", "speed"])`); only non-numeric speeds become 0. The unit is **knots** (tracker speed ÷ GPS-derived km/h = 0.539). It contains junk values (max 3,883). Nothing downstream uses it. The `heading` column is empty in every export and every `fisherDepth/` file.

0.4 **Depth grid (fixed 15 Nov).** `loadDepthGrid` used to rebuild latitude from a 0.015625° step anchored at row 239 = 5.0° N; that was 0.2578° (about 29 km) too far north, giving a median depth error of about 380 m. It now reads latitude from `data/YC.data` (1.008–7.008° N). `update_fisher_depth.py` redid `depth_m` and `on_land` in all 25 `fisherDepth/` files in place (positions and other columns unchanged; originals backed up). Depth changed for 120,193 of 124,513 pings; `on_land` went from 1 to 3,778 pings. Checked against Natural Earth 10m land: 3,453 of the 3,778 are on land there too (325 are coastal sea pings in a coarse 1/64°, about 1.7 km, land cell). 1,465 are within 1 km of Tema port; most others are at beach landing sites (largest: around 5.39° N, 0.49° W, 1,154 pings, mostly 020 GodFirst), plus a few truly inland pings (up to 13 km, one at 160 km). Only 1 ping the grid calls sea is more than 2 km inland, so the corrected grid lines up with the coastline. `on_land` marks pings at landing sites, not GPS errors; do not drop them as errors. No HMM script uses depth.

0.5 **`on_land` means depth == 0 in the grid.** Flagged but **not removed** anywhere downstream. 0 pings are outside the grid.

0.6 **`fisherDepth/` was not built from the current `Exported Files/`.** Boat names differ (016 GreatProvider vs Sabbath2, 017 GoodFather vs AyeYi, 021 Otsokobila vs Apostolic, 022 EnsoNyameYe2 vs VictoryNo9No10, 012 IfMenWereGod vs IfMenWereGod_1); 002, 015 and 018 have exports but no depth file; the exports end 20 Apr 2026 but `fisherDepth/` runs to 27 May 2026; and `fisherDepth/` has no `outside_grid` column, so a different script version wrote it. Step 0 cannot be reproduced from the files present. **Open question:** are 016, 017, 021 and 022 the same physical boats under new names, or were trackers moved between boats?

## 1. Trips and segments (`pipeline_scripts/build_trip_tracks.py`)

1.1 **Trips are defined only by Tema port.** A trip is a run of pings more than `TEMA_PORT_RADIUS_KM = 1.0` from Tema (5°38'14.6" N, 0°00'58.5" E). Some boats are based elsewhere: 019 Emmanuel (closest 13.7 km), 016 GreatProvider (195 km) and 022 EnsoNyameYe2 (199 km) never enter the radius; 014 RoyalDiadem and 017 GoodFather rarely do. Their "trips" span weeks (longest: 024 t2 52.5 days, 019 t1 43.8, 011 t2 31.8).

1.2 **The 1 km port radius may be too tight.** Several boats have 2-6 pings just west of Tema (about 5.61° N, 0.06° W) before being detected as in port.

1.3 **A trip's end row is the first ping back inside the port** (`detect_tema_trips`), so each returned trip includes one in-port ping. 14 single-ping trips exist, all from the 4-day split: a lone in-port ping after a long gap (labelled `port`), or a lone ping just outside the radius (`gap` or `end_of_data`). Crawl drops them (< 10 pings).

1.4 **Trips are split at gaps between pings longer than `MAX_TRIP_GAP_DAYS = 4`** (`split_at_long_gaps`). The piece after a gap is kept as its own trip even though it did not start in Tema. `trip_end_reason` is `port`, `gap` or `end_of_data`. 228 trips: 177 port, 31 gap, 20 end_of_data.

1.5 **Shared outage.** All trackers went silent from 11 Feb 2026 23:32:21 to 16 Apr 2026 00:00:35. No trip spans it.

1.6 **Segments are split at gaps longer than `MAX_GAP_MINUTES = 30`.** Segments are the unit for crawl and the HMM. About 13% of within-segment intervals are 5.5-30 minutes and get filled by crawl.

1.7 **Projection: UTM zone 31N (EPSG:32631) for all positions.** Tracks reach 5.59° W. Measured UTM ÷ geodesic step ratio: 1.0069 at 6–3° W (6.9% of steps), 1.0018 at 3–0° W, 1.0007 at 0–2° E. Small for the HMM; a local equidistant projection would remove it.

1.8 **GPS spikes are removed, not moved (changed 15 Nov; `drop_speed_spikes`).** The old `cap_implausible_speeds` pulled fast pings back along their bearing from the previous *capped* point, so a really fast boat dragged a lagging track by up to 29 km (e.g. `OMI_Fisher_017_GoodFather_t4_s5`). The new filter runs per boat on raw pings, before trips are built, and never moves a ping:
- After a step faster than `MAX_SPEED_KMH = 30`, it looks up to `MAX_SPIKE_PINGS = 3` pings ahead for a ping that (a) is reached by a fast step, (b) is reachable directly from the last good ping at ≤ 30 km/h (or ≤ half the jump speed), and (c) makes the pings between a detour at least `MIN_DETOUR_RATIO = 2` times the direct distance. The pings in between are removed.
- Pings under `MIN_PING_SECONDS = 60` apart with a fast step between are near-duplicates; the one farther from the next ping is removed. A bad first or last ping is removed when the step inward is normal.
- Result: 100 pings removed (66 single, 8 pairs, 6 triples); GoodFather's real ~30 km/h run is kept whole. 52 steps inside segments remain faster than 30 km/h (17 over 60 km/h, max 165); they are kept because they are not clear out-and-back jumps. Check whether those are real.
- Checked on the sample: earlier versions of this rule either missed multi-ping excursions or ate into real fast runs; the detour and fast-return conditions fix both.

1.9 **Rows with missing timestamp, lat or lon are dropped.** `trip_tracks.csv` has no duplicate timestamps within a segment.

1.10 **No filtering of `on_land` pings.**

1.11 **All trips are modelled, whether or not they return to Tema.** `returned_to_tema` and `trip_end_reason` are carried as flags only; no HMM or event script filters on them. In the HMM input (133,428 positions, 1,277 segments), 65.1% of positions come from trips that ended in Tema (`port`), 27.4% from trips ended by a > 4-day gap and 7.6% from `end_of_data` trips; 503 of the 1,277 segments are from trips that did not return to Tema. Boats based at other landing sites (1.1) are therefore included, but their "trips" can span several real journeys.

## 2. Interpolation to 5 minutes (`pipeline_scripts/interpolation_script.R`, crawl)

2.1 **Duplicate timestamps within a segment: the first row is kept.** There are none in the current data, so this has no effect.

2.2 **Segments with fewer than `MIN_POINTS = 10` pings are skipped.** 1,314 of 2,942 segments are skipped (1,628 kept).

2.3 **Near-stationary segments are skipped:** `max(sd(x), sd(y)) < STATIONARY_SD_M = 10` m. 33 of the 38 skipped are from `OMI_Fisher_019_Emmanuel`. This may bias against the near-stationary state.

2.4 **Movement model: a continuous-time correlated random walk with constant parameters per segment** (`mov.model = ~1`, `time.scale = "hours"`) and **no location-error model**, so GPS error goes straight into steps and angles.

2.5 **Fits are accepted only if converged with finite standard errors.** Failing segments get up to 3 extra passes, then are dropped; their IDs are printed in the log but not saved to a file. In the 15 Nov run, 57 segments needed a second pass, 5 a third, and 0 failed (1,590 fitted).

2.6 **The prediction grid starts at each segment's first ping and steps exactly 5 minutes.** Not aligned to clock time.

2.7 **The posterior mean position (`mu.x`, `mu.y`) is used, not a simulated path.** Positions inside gaps are smooth averages. 23,063 of 138,410 5-minute positions (16.7%) have no real ping within ±2.5 min.

2.8 **The original observed rows ("o") are discarded**; only grid predictions ("p") are kept.

2.9 **Crawl is seeded (since 15 Nov, `SEED = 42`).** Before, `crwMLE(attempts = 10)` used unseeded random starts: rerunning on identical input changed 519 of 1,589 segments by more than 1 m and 9 by more than 1 km (worst 30 km, `OMI_Fisher_001_MessiahTanker3_t13_s16`), and moved the main log-likelihood from 102,430 to 105,261. A seed makes reruns repeatable but does not show which optimum is better for those segments; check them.

## 3. HMM fits (`pipeline_scripts/fit_hmm.R`, with variants)

3.1 **Segments need at least `MIN_STEPS = 20` steps.** 313 of 1,590 crawl segments are dropped (4,982 positions, 3.6%); 1,277 segments and 132,151 steps remain.

3.2 **Coordinates are converted to km, so steps are km per 5 minutes.** `prepData(type = "UTM")`. Angle is NA at each segment's ends and after zero-length steps. Inherits 1.7.

3.3 **Zero-length steps are replaced with 1 m** (`ZERO_STEP_KM`) because the gamma cannot take 0. 62 (interpolated), 1,291 (no interpolation). Alternative: `zeroInflation = list(step = TRUE)`.

3.4 **Exactly 3 states.** Not tested against 2 or 4 states with AIC/BIC or pseudo-residuals.

3.5 **Gamma step lengths and von Mises turning angles** with the **mean angle fixed at 0** (`estAngleMean = FALSE`).

3.6 **Constant transition probabilities.** No covariates (time of day, distance from shore, depth). The HMM uses only step length and turning angle from positions.

3.7 **Segments are independent tracks** sharing one set of parameters; no boat or trip random effects.

3.8 **Starting values and seed.** Random around the 20th/50th/90th step percentiles; concentrations U(0.5, 3) sorted upward, so state 1 (slowest) *starts* as the most tortuous. The fit makes the slow state straight instead; all starts agree, so this is minor. Seed 42; 10 starts for the main fit, 3 for the variants.

3.9 **A fit is "valid"** if the `nlm` code is 1 or 2, the likelihood is finite and `stateProbs` runs. Best = lowest negative log-likelihood. `retryFits = 0`.

3.10 **Boundary transitions.** Main fit: near-stationary→transit and transit→near-stationary are at −23.3 on the working scale (probability about 0). The no-interpolation fit has the same problem (−21.6, −22.1). Step-only and angle-only do not. In the 15 Nov fits the Hessians are positive definite (smallest eigenvalue 0.0027 main, 0.0019 no-interpolation; before 15 Nov both had negative eigenvalues) and every standard error is finite, but standard errors for parameters at a boundary are still not meaningful.

3.11 **State numbers are arbitrary.** Roles are assigned by mean step length: slowest = near-stationary, **middle = fishing**, fastest = transit. Main fit (15 Nov): fishing 0.189 km/5 min (1.23 kn, κ 1.50), near-stationary 0.158 (1.02 kn, κ 111), transit 0.771 (5.0 kn, κ 127). Fishing and near-stationary differ mainly in turning, not speed. All 10 starts reached log-likelihood 106,291.6.

3.12 **Label checks (extended 15 Nov).** The fit stops if fishing is not more tortuous than transit, and now also if it is not more tortuous than the slow state (a fishing/near-stationary swap).

3.13 **"Near-stationary" is not stationary.** It averages about 1 kn with κ about 111 and never switches directly to or from transit. It looks like slow directed movement or drifting; consider renaming it.

3.14 **Viterbi vs probabilities.** The Pi_min sweeps and the track maps (`plot_hmm_tracks.py`) use the probabilities; `fishing_events.R` uses Viterbi, and `plot_fishing_min_length.py` uses Viterbi only to assign roles.

3.15 **The Pi_min sweep runs from 0.5 to 0.9 (changed 15 Nov).** Pi_min = 1.0 was dropped: it is reached only through numerical underflow, and only by the low-κ state (von Mises densities of high-κ states underflow to 0 at large turns), so it was biased toward fishing.

### Variants

3.16 **`fit_hmm_step_only.R`:** step length only; fishing = middle mean step. Its "near-stationary" is really stationary (0.053 km/5 min; fishing 0.202, transit 0.770), so its fishing state is roughly fishing plus near-stationary from the main fit. No angle check.

3.17 **`fit_hmm_angle_only.R`:** turning angle only; fishing = lowest concentration (κ 1.07, 33.4, 448). Cannot separate stationary from transit.

3.18 **`fit_hmm_no_interp.R`:** raw pings snapped to the crawl 5-minute grid (nearest slot, ±2.5 min; the closer ping wins a shared slot; empty slots NA). Segments come from `regularised_tracks.csv`. 98,498 steps; 111,364 of 133,428 slots (83.5%) have a ping; 317 pings lost to shared slots. Fishing 0.189 km/5 min (κ 1.48), near-stationary 0.157 (κ 90.6), transit 0.775 (κ 100). Its log-likelihood is not comparable with the interpolated fit.

The sweep panels therefore compare different state definitions across models.

## 4. Fishing events and plots

4.1 **`fishing_events.R`: gap fill first, then minimum length**, within each segment (they cannot join across segments). `MAX_GAP_FIXES = 1`, `MIN_EVENT_FIXES = 2`. Applied to Viterbi labels and to each Pi_min.

4.2 **`plot_fishing_min_length.py`: `PI_MIN = 0.75`, n = 1-6, same gap fill.** Roles come from the empirical mean `step_km` per Viterbi state; these give the same order as the fitted gamma means in the main, step-only and no-interpolation models (main 0.188 / 0.158 / 0.771). The angle-only model's roles come from κ, not step length (`plot_state_sweep.R`); this script uses only the main model. At Pi_min ≥ 0.5 at most one state can qualify.

4.3 **`plot_hmm_tracks.py` samples 5 trips with at least 40 steps (seed 42)**, coloured by role at Pi_min 0.75. Regenerated in the 15 Nov rerun, so the sampled trips may differ from earlier versions.

4.4 **`plot_state_sweep.R` finds each model's `.rds` from its states-file name by regex.** Shares are over all positions, including empty slots in the no-interpolation model.

## 5. Cross-cutting checks still to do

- Pseudo-residuals (`pseudoRes`) on the main fit; 2 vs 3 vs 4 states by AIC/BIC (3.4).
- Which crawl optimum is right for the segments that moved by kilometres between runs (2.9).
- Boat identities for 016, 017, 021, 022 and the export behind `fisherDepth/` (0.6).
- Time zone of the raw exports (0.2).
- Whether the 52 remaining fast steps are real (1.8).
