#!/usr/bin/env python3
"""Plot whitened strain with glitch reconstruction and residual from GL1_result.json."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import bilby.gw.detector
import bilby.gw.waveform_generator
from bilby.core.utils import infft
from bilby.gw.transdimensional_source_models import make_glitch_signal_model

parser = argparse.ArgumentParser()
parser.add_argument('--ci', action='store_true',
                    help='Compute and plot the 90%% posterior predictive CI (slow)')
parser.add_argument('--qscan', action='store_true',
                    help='Generate a Q-scan of the glitched and deglitched data '
                         '(saved to glitch_qscan.png)')
args = parser.parse_args()

STRAIN_FILE = "H1_SL1.gwf"
RESULT_FILE = "GL1_result.json"

SAMPLE_RATE = 2048.0
GPS_START = 100.0
DURATION = 8
N_MAX = 8
MIN_FREQ = 20
MAX_FREQ = 1024

# ── Load posterior samples ────────────────────────────────────────────────────
with open(RESULT_FILE) as f:
    result = json.load(f)

param_keys = result['search_parameter_keys']
samples = np.array(result['samples']['content'])
log_likes = np.array(result['log_likelihood_evaluations']['content'])
map_idx = np.argmax(log_likes)
map_params = dict(zip(param_keys, samples[map_idx]))

print(f"Loaded {len(samples)} posterior samples")
print(f"MAP sample: index={map_idx}, n={int(map_params['H1_n'])}")

# ── Set up IFO from actual .gwf file ─────────────────────────────────────────
ifo = bilby.gw.detector.get_empty_interferometer('H1')
ifo.set_strain_data_from_frame_file(
    frame_file=STRAIN_FILE,
    sampling_frequency=SAMPLE_RATE,
    duration=DURATION,
    start_time=GPS_START,
    channel='H1:STRAIN',
)
t_rel = ifo.time_array - GPS_START

# ── Whitened strain ───────────────────────────────────────────────────────────
whitened_data = ifo.whitened_time_domain_strain

# ── Waveform generator (same setup as glitch_reconstruction.py) ──────────────
glitch_model = make_glitch_signal_model(N_MAX, ifo)
wfgen = bilby.gw.waveform_generator.WaveformGenerator(
    duration=DURATION,
    sampling_frequency=SAMPLE_RATE,
    frequency_domain_source_model=glitch_model,
    parameter_conversion=None,
    waveform_arguments=dict(minimum_frequency=MIN_FREQ, maximum_frequency=MAX_FREQ),
)
wfgen.start_time = GPS_START


def whiten_recon(fd):
    """Whiten a reconstruction frequency series using the IFO's noise model."""
    whitened_fd = ifo.whiten_frequency_series(fd)
    return ifo.get_whitened_time_series_from_whitened_frequency_series(whitened_fd)


# ── MAP reconstruction ────────────────────────────────────────────────────────
glitch_params_map = {k[3:]: v for k, v in map_params.items() if k.startswith('H1_')}
fd_map = wfgen.frequency_domain_strain(glitch_params_map)
whitened_recon_map = whiten_recon(fd_map['plus'])
# Unwhitened reconstruction (for Q-scan subtraction)
recon_map_td = infft(fd_map['plus'], SAMPLE_RATE)

# ── Posterior predictive band (90% CI, optional) ─────────────────────────────
if args.ci:
    print("Computing posterior predictive band...")
    recon_all = np.zeros((len(samples), len(t_rel)))
    for i, row in enumerate(samples):
        params = dict(zip(param_keys, row))
        gp = {k[3:]: v for k, v in params.items() if k.startswith('H1_')}
        fd = wfgen.frequency_domain_strain(gp)
        recon_all[i] = whiten_recon(fd['plus'])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(samples)}")
    recon_median = np.median(recon_all, axis=0)
    recon_lo = np.percentile(recon_all, 5, axis=0)
    recon_hi = np.percentile(recon_all, 95, axis=0)

whitened_residual = whitened_data - whitened_recon_map

# ── Main plot: whitened strain + reconstruction + residual ────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
ax.plot(t_rel, whitened_data, color='0.4', lw=0.6, alpha=0.8, label='Whitened strain (data)')
if args.ci:
    ax.fill_between(t_rel, recon_lo, recon_hi, color='firebrick', alpha=0.35,
                    label='Reconstruction 90% CI')
    ax.plot(t_rel, recon_median, color='firebrick', lw=1.0, label='Reconstruction median')
ax.plot(t_rel, whitened_recon_map, color='darkorange', lw=1.2, ls='--',
        label=f'Reconstruction MAP (n={int(map_params["H1_n"])})')
ax.set_ylabel('Whitened strain [1/√Hz]')
ax.set_title('Whitened Strain with Glitch Reconstruction')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_rel, whitened_residual, color='steelblue', lw=0.7, label='Residual (data − MAP)')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_ylabel('Whitened residual [1/√Hz]')
ax.set_xlabel('Time from GPS start [s]')
ax.set_title('Residual (Whitened Data − MAP Reconstruction)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
outfile = 'glitch_comparison.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"Saved {outfile}")

# ── Q-scan plot (optional) ────────────────────────────────────────────────────
if args.qscan:
    from gwpy.timeseries import TimeSeries

    raw_strain = ifo.time_domain_strain
    ts_glitched = TimeSeries(raw_strain, sample_rate=SAMPLE_RATE, t0=GPS_START)
    ts_deglitched = TimeSeries(raw_strain - recon_map_td, sample_rate=SAMPLE_RATE, t0=GPS_START)

    # Build the shared ASD from the IFO's theoretical noise model so both
    # transforms are whitened with the same stable floor.  Estimating the ASD
    # from the data (whiten=True) is unreliable here: the deglitched signal has
    # reduced power at glitch frequencies, biasing its own ASD estimate and
    # producing vertical stripe artefacts.
    from gwpy.frequencyseries import FrequencySeries
    freq_arr = ifo.strain_data.frequency_array
    asd_arr = ifo.amplitude_spectral_density_array
    finite = np.isfinite(asd_arr) & (asd_arr > 0)
    shared_asd = FrequencySeries(asd_arr[finite], frequencies=freq_arr[finite])

    qt_kw = dict(
        qrange=(4, 64),
        frange=(MIN_FREQ, MAX_FREQ // 2),
        logf=True,
        norm='median',
        whiten=shared_asd,
    )
    print("Computing Q-transforms...")
    qt_glitched = ts_glitched.q_transform(**qt_kw)
    qt_deglitched = ts_deglitched.q_transform(**qt_kw)

    times = qt_glitched.times.value - GPS_START
    freqs = qt_glitched.frequencies.value

    vmax = np.percentile(qt_glitched.value, 99)

    fig_q, axes_q = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    for ax, qt, title in [
        (axes_q[0], qt_glitched,   'Glitched (original data)'),
        (axes_q[1], qt_deglitched, 'Deglitched (data − MAP reconstruction)'),
    ]:
        pcm = ax.pcolormesh(
            times, freqs, qt.value.T,
            cmap='viridis', vmin=0, vmax=vmax, shading='nearest',
        )
        ax.set_yscale('log')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_title(title)
        fig_q.colorbar(pcm, ax=ax, label='Normalised energy')

    axes_q[1].set_xlabel('Time from GPS start [s]')
    fig_q.tight_layout()
    qscan_outfile = 'glitch_qscan.png'
    fig_q.savefig(qscan_outfile, dpi=150, bbox_inches='tight')
    print(f"Saved {qscan_outfile}")

plt.show()
