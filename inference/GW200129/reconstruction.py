import tbilby
import bilby
import numpy as np

from tbilby.core.prior.order_stats import TransdimensionalConditionalDescendingOrderStatPrior
from gwpy.timeseries import TimeSeries
from bilby.gw.transdimensional_source_models import make_glitch_signal_model
from bilby.gw.source import lal_binary_black_hole

logger = bilby.core.utils.logger
outdir = "GW200129_transdimensional"
label = "GW200129"
sampling_frequency = 2048.0
trigger_time = 1264316116.4
detectors = ["H1", "L1", "V1"]
maximum_frequency = 1024
minimum_frequency = 20
roll_off = 0.4
duration = 2
post_trigger_duration = 1
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

N = 5  # Max number of sine-Gaussian glitch components per IFO

# ── Data loading ───────────────────────────────────────────────────────────────
ifo_list = bilby.gw.detector.InterferometerList([])
# ifo_list.set_strain_data_from_power_spectral_densities(
#     sampling_frequency=sampling_frequency,
#     duration=duration,
#     start_time=start_time,
    # psd_start_time=psd_start_time,
    # psd_end_time=psd_end_time,
    # roll_off=roll_off,
    # minimum_frequency=minimum_frequency,
    # maximum_frequency=maximum_frequency,
# )

for det in detectors:
    logger.info("Downloading analysis data for ifo {}".format(det))
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    data = TimeSeries.fetch_open_data(det, start_time, end_time)
    ifo.strain_data.set_from_gwpy_timeseries(data)

    logger.info("Downloading psd data for ifo {}".format(det))
    psd_data = TimeSeries.fetch_open_data(det, psd_start_time, psd_end_time)
    psd_alpha = 2 * roll_off / duration
    psd = psd_data.psd(
        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
    )
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
        frequency_array=psd.frequencies.value, psd_array=psd.value
    )
    ifo.maximum_frequency = maximum_frequency
    ifo.minimum_frequency = minimum_frequency
    ifo_list.append(ifo)

# ── Waveform generators ────────────────────────────────────────────────────────
waveform_arguments = dict(
    minimum_frequency=minimum_frequency,
    maximum_frequency=maximum_frequency,
    reference_frequency=minimum_frequency,
    waveform_approximant="IMRPhenomXPHM",
)
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# ifo_list[0] = H1, ifo_list[1] = L1, ifo_list[2] = V1 — order matches the detectors loop above.
glitch_generators = {
    'H1': bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=make_glitch_signal_model(N, ifo_list[0]),
        parameter_conversion=None,
    ),
    'L1': bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=make_glitch_signal_model(N, ifo_list[1]),
        parameter_conversion=None,
    ),
    'V1': bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=make_glitch_signal_model(N, ifo_list[2]),
        parameter_conversion=None,
    ),
}

# ── Likelihood ─────────────────────────────────────────────────────────────────
likelihood = bilby.gw.likelihood.GlitchGravitationalWaveTransient(
    interferometers=ifo_list,
    waveform_generator=waveform_generator,
    glitch_waveform_generators=glitch_generators,
)

# ── Priors ─────────────────────────────────────────────────────────────────────
# Separate SNR prior classes per IFO — each references its own prefix attributes
# (H1_SNR/H1_n vs L1_SNR/L1_n) as set by tBilby's conditional machinery.

class H1TransdimensionalSNRPrior(TransdimensionalConditionalDescendingOrderStatPrior):
    def transdimensional_condition_function(self, **required_variables):
        if len(self.H1_SNR) > 0:
            self._prev_val = self.H1_SNR[-1]
            self._this_order_num = self.H1_SNR.shape[0] + 1
        else:
            self.this_order_num = 1
            if isinstance(self.H1_n, np.ndarray):
                self._prev_val = self.minimum * np.ones(self.H1_n.shape)
        try:
            self._tot_order_num = self.H1_n.astype(int)
        except Exception:
            self._tot_order_num = int(self.H1_n)
        return dict(
            _prev_val=self._prev_val,
            _this_order_num=self._this_order_num,
            _tot_order_num=self._tot_order_num,
        )


class L1TransdimensionalSNRPrior(TransdimensionalConditionalDescendingOrderStatPrior):
    def transdimensional_condition_function(self, **required_variables):
        if len(self.L1_SNR) > 0:
            self._prev_val = self.L1_SNR[-1]
            self._this_order_num = self.L1_SNR.shape[0] + 1
        else:
            self.this_order_num = 1
            if isinstance(self.L1_n, np.ndarray):
                self._prev_val = self.minimum * np.ones(self.L1_n.shape)
        try:
            self._tot_order_num = self.L1_n.astype(int)
        except Exception:
            self._tot_order_num = int(self.L1_n)
        return dict(
            _prev_val=self._prev_val,
            _this_order_num=self._this_order_num,
            _tot_order_num=self._tot_order_num,
        )


class V1TransdimensionalSNRPrior(TransdimensionalConditionalDescendingOrderStatPrior):
    def transdimensional_condition_function(self, **required_variables):
        if len(self.V1_SNR) > 0:
            self._prev_val = self.V1_SNR[-1]
            self._this_order_num = self.V1_SNR.shape[0] + 1
        else:
            self.this_order_num = 1
            if isinstance(self.V1_n, np.ndarray):
                self._prev_val = self.minimum * np.ones(self.V1_n.shape)
        try:
            self._tot_order_num = self.V1_n.astype(int)
        except Exception:
            self._tot_order_num = int(self.V1_n)
        return dict(
            _prev_val=self._prev_val,
            _this_order_num=self._this_order_num,
            _tot_order_num=self._tot_order_num,
        )

priors = bilby.core.prior.dict.ConditionalPriorDict()

priors = tbilby.core.base.create_transdimensional_priors(
    transdimensional_prior_class=H1TransdimensionalSNRPrior,
    param_name='H1_SNR',
    nmax=N,
    nested_conditional_transdimensional_params=['H1_SNR'],
    conditional_transdimensional_params=[],
    conditional_params=['H1_n'],
    prior_dict_to_add=priors,
    SaveConditionFunctionsToFile=False,
    minimum=0, maximum=20, prev_val=20, this_order_num=1,
)

priors = tbilby.core.base.create_transdimensional_priors(
    transdimensional_prior_class=L1TransdimensionalSNRPrior,
    param_name='L1_SNR',
    nmax=N,
    nested_conditional_transdimensional_params=['L1_SNR'],
    conditional_transdimensional_params=[],
    conditional_params=['L1_n'],
    prior_dict_to_add=priors,
    SaveConditionFunctionsToFile=False,
    minimum=0, maximum=20, prev_val=20, this_order_num=1,
)

priors = tbilby.core.base.create_transdimensional_priors(
    transdimensional_prior_class=V1TransdimensionalSNRPrior,
    param_name='V1_SNR',
    nmax=N,
    nested_conditional_transdimensional_params=['V1_SNR'],
    conditional_transdimensional_params=[],
    conditional_params=['V1_n'],
    prior_dict_to_add=priors,
    SaveConditionFunctionsToFile=False,
    minimum=0, maximum=20, prev_val=20, this_order_num=1,
)

priors['H1_n'] = tbilby.core.prior.DiscreteUniform(0, N, 'H1_n_dimension')
priors['L1_n'] = tbilby.core.prior.DiscreteUniform(0, N, 'L1_n_dimension')
priors['V1_n'] = tbilby.core.prior.DiscreteUniform(0, N, 'V1_n_dimension')

for i in range(N):
    for prefix in ('H1', 'L1', 'V1'):
        priors[f'{prefix}_dt{i}'] = bilby.core.prior.Uniform(
           trigger_time-0.2, trigger_time+0.2, name=f'{prefix}_dt{i}')
        priors[f'{prefix}_f{i}'] = bilby.core.prior.Uniform(
            20, 50, name=f'{prefix}_f{i}')
        priors[f'{prefix}_Q{i}'] = bilby.core.prior.Uniform(
            1, 20, name=f'{prefix}_Q{i}')
        priors[f'{prefix}_phi{i}'] = bilby.core.prior.Uniform(
            0, 2 * np.pi, name=f'{prefix}_phi{i}', boundary='periodic')

# BBH signal priors — sample in chirp_mass + mass_ratio to avoid needing a
# Constraint prior (which has array-vs-scalar shape issues in ConditionalPriorDict).
# convert_to_lal_binary_black_hole_parameters derives mass_1 and mass_2 from these.
# chirp_mass range covers the GW231123 posterior (m1~120, m2~112 → Mc~101 Msun).
# mass_ratio bounded at 0.25 (NRSur7dq4 lower limit).
priors['chirp_mass'] = bilby.core.prior.Uniform(
    minimum=26, maximum=38, name='chirp_mass', latex_label='$\\mathcal{M}$', unit='$M_\\odot$')
priors['mass_ratio'] = bilby.core.prior.Uniform(
    minimum=0.167, maximum=1.0, name='mass_ratio', latex_label='$q$')
priors['luminosity_distance'] = bilby.core.prior.PowerLaw(
    alpha=0.2, minimum=500, maximum=1500,
    name='luminosity_distance', latex_label='$d_L$', unit='Mpc')
priors['geocent_time'] = bilby.core.prior.Uniform(
    trigger_time - 0.1, trigger_time + 0.1, name='geocent_time')
priors['ra'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, name='ra', boundary='periodic')
priors['dec'] = bilby.core.prior.Cosine(name='dec')
priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
priors['psi'] = bilby.core.prior.Uniform(
    0, np.pi, name='psi', boundary='periodic')
priors['phase'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, name='phase', boundary='periodic')
priors['a_1'] = bilby.core.prior.Uniform(0, 0.99, name='a_1')
priors['a_2'] = bilby.core.prior.Uniform(0, 0.99, name='a_2')
priors['tilt_1'] = bilby.core.prior.Sine(name='tilt_1')
priors['tilt_2'] = bilby.core.prior.Sine(name='tilt_2')
priors['phi_12'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, name='phi_12', boundary='periodic')
priors['phi_jl'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, name='phi_jl', boundary='periodic')

print("Sampling...")
# ── Sampling ───────────────────────────────────────────────────────────────────
result = bilby.core.sampler.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample="rwalk",
    nlive=3000,
    nact=80,
    outdir=outdir,
    label=label,
    resume=True,
    npool=64,
)
