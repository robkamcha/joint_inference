#!/usr/bin/env python3
"""Write a Scattered_Light GAN glitch to a .gwf frame file."""

import argparse
import os

import numpy as np
import lal
import bilby
from gwpy.timeseries import TimeSeries

GLITCH_DATA_DIR = "/home/robinc/et-tgr/poisson_glitches/glitch_data"
SAMPLE_RATE = 2048  # Hz — matches glitch_reconsruction.py
IFO_NAME = "H1"
GPS_DEFAULT = 100 
GLITCH_SAMPLING_FREQUENCY = 4096.0

def main():
    parser = argparse.ArgumentParser(
        description="Generate a .gwf frame file containing a Scattered_Light glitch."
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Index into the Scattered_Light subset (default: 0)",
    )
    parser.add_argument(
        "--gps-start", type=int, default=GPS_DEFAULT,
        help=f"GPS start time written into the frame (default: {GPS_DEFAULT})",
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Directory for the output .gwf file (default: current directory)",
    )
    args = parser.parse_args()

    samples = np.load(os.path.join(GLITCH_DATA_DIR, "glitch_GAN_samples_scaled_balanced.npy"))
    labels = np.load(os.path.join(GLITCH_DATA_DIR, "glitch_GAN_labels_balanced.npy"))

    # order = ['Blip', 'Fast_Scattering', 'Koi_Fish', 'Low_Frequency_Burst',
    #          'Scattered_Light', 'Tomte', 'Whistle']
    scattered = samples[labels[:, 4] == 1]
    n_available = len(scattered)
    if args.index >= n_available:
        raise ValueError(f"--index {args.index} out of range; {n_available} Scattered_Light samples available")

    glitch = scattered[args.index]  # shape (8192,)

    dt = 1.0 / SAMPLE_RATE
    n_samples = len(glitch)
    duration = 8.0  # seconds

    ifo = bilby.gw.detector.get_empty_interferometer(IFO_NAME)
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=SAMPLE_RATE, duration=duration, start_time=args.gps_start
    )

    glitch_parameters = dict(
        onset_time=args.gps_start + 2.0, 
        snr=40.0
    )
    glitch_sample_times = np.arange(n_samples) / GLITCH_SAMPLING_FREQUENCY

    ifo.inject_glitch(
        glitch_parameters=glitch_parameters,
        glitch_sample_times=glitch_sample_times,
        glitch_time_domain_strain=glitch,
    )

    # Save time series
    ts = TimeSeries(
        data=ifo.time_domain_strain,
        sample_rate=ifo.sampling_frequency,
        t0=ifo.start_time,
        channel=f'{ifo.name}:STRAIN',
    )
    main_dir = '/home/robinc/projects/joint_inference/inference'
    ts.write(os.path.join(main_dir, f'{ifo.name}_SL1.gwf'))

    
    print(f"Wrote {ifo.name}_SL1.gwf  (index={args.index}, GPS={args.gps_start}, duration={duration:.1f}s)")


if __name__ == "__main__":
    main()
