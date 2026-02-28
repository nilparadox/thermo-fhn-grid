#!/bin/bash

# -------------------------------------------------
# Find latest run automatically
# -------------------------------------------------
RUN_DIR=$(ls -1dt runs/* | head -n 1)
echo "Using RUN_DIR: $RUN_DIR"

PAPER_DIR="paper_figures"
mkdir -p "$PAPER_DIR"

# -------------------------------------------------
# Figure 1 — Example dynamics
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Fig1_example_dynamics"
cp "$RUN_DIR/examples/example_raster__spike_events.png" \
   "$PAPER_DIR/Fig1_example_dynamics/" 2>/dev/null
cp "$RUN_DIR/examples/example_timeseries__center_node.png" \
   "$PAPER_DIR/Fig1_example_dynamics/" 2>/dev/null

# -------------------------------------------------
# Figure 2 — Time-resolved coherence
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Fig2_time_coherence"
cp "$RUN_DIR/examples/derived_dynamics_from_raw/dynamics_kuramoto_R_vs_time.png" \
   "$PAPER_DIR/Fig2_time_coherence/" 2>/dev/null
cp "$RUN_DIR/examples/derived_dynamics_from_raw/dynamics_mean_field_xbar_vs_time.png" \
   "$PAPER_DIR/Fig2_time_coherence/" 2>/dev/null

# -------------------------------------------------
# Figure 3 — Phase locking
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Fig3_phase_locking"
cp "$RUN_DIR/examples/derived_dynamics_from_raw/dynamics_phase_difference_dphi_vs_time__node_3_3__node_21_21.png" \
   "$PAPER_DIR/Fig3_phase_locking/" 2>/dev/null

# -------------------------------------------------
# Figure 4 — Spatial structure
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Fig4_spatial_structure"
cp "$RUN_DIR/examples/derived_dynamics_from_raw/spatial_firing_rate_map__steady_window.png" \
   "$PAPER_DIR/Fig4_spatial_structure/" 2>/dev/null
cp "$RUN_DIR/examples/derived_dynamics_from_raw/spatial_phase_snapshot_map__late_time.png" \
   "$PAPER_DIR/Fig4_spatial_structure/" 2>/dev/null

# -------------------------------------------------
# Figure 5 — Phase maps
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Fig5_phase_maps"
cp "$RUN_DIR/maps_em_f/phase_map_em_f__kuramoto_R__mean.png" \
   "$PAPER_DIR/Fig5_phase_maps/" 2>/dev/null
cp "$RUN_DIR/maps_em_f/phase_map_em_f__cv_active_mean__mean.png" \
   "$PAPER_DIR/Fig5_phase_maps/" 2>/dev/null

# -------------------------------------------------
# Figure 6 — Temperature sweep
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Fig6_temperature_control"
cp "$RUN_DIR/temperature_sweep/temperature_sweep__R_and_CV_active__mean_std.png" \
   "$PAPER_DIR/Fig6_temperature_control/" 2>/dev/null

# -------------------------------------------------
# Supplementary figures
# -------------------------------------------------
mkdir -p "$PAPER_DIR/Supplement"

cp "$RUN_DIR/maps_em_f/phase_map_em_f__kuramoto_R__std.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null
cp "$RUN_DIR/maps_em_f/phase_map_em_f__cv_active_mean__std.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null
cp "$RUN_DIR/maps_em_f/phase_map_em_f__active_fraction__mean.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null
cp "$RUN_DIR/maps_em_f/phase_map_em_f__active_fraction__std.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null

cp "$RUN_DIR/temperature_sweep/temperature_sweep__active_fraction__mean_std.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null

cp "$RUN_DIR/slices_frequency/slice_R_vs_f__multiple_Em__mean_std.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null
cp "$RUN_DIR/slices_frequency/slice_active_fraction_vs_f__multiple_Em__mean_std.png" \
   "$PAPER_DIR/Supplement/" 2>/dev/null

# -------------------------------------------------
# Create captions file
# -------------------------------------------------
cat > "$PAPER_DIR/captions.txt" << 'CAP'

Figure 1. Example collective dynamics.
(a) Raster plot of spike events across the 25×25 network.
(b) Representative time series of a single neuron.

Figure 2. Time-resolved collective coherence.
(a) Mean-field activity x̄(t).
(b) Kuramoto order parameter R(t) computed from spike-defined phases.

Figure 3. Phase coordination between distant neurons.
Wrapped phase difference Δφ(t) between nodes (3,3) and (21,21).

Figure 4. Spatial organization in the steady regime.
(a) Firing-rate map over the grid.
(b) Spatial phase snapshot at late time.

Figure 5. Phase diagrams in modulation parameter space.
(a) Mean Kuramoto coherence R(Em, f).
(b) Mean coefficient of variation CV(Em, f).

Figure 6. Thermal control.
Mean and standard deviation of R and CV as a function of baseline temperature T.

Supplementary Figures.
Robustness across seeds (standard deviations), active fraction controls, and frequency slices.

CAP

echo ""
echo "Paper figures collected in: $PAPER_DIR"
echo "Structure:"
find "$PAPER_DIR" -maxdepth 2 -type f | sort
