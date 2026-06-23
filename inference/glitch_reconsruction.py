import tbilby
import bilby.core.utils
import bilby.core.prior
import bilby.core.prior.dict
import bilby.core.sampler
import bilby.gw.detector
import bilby.gw.waveform_generator
import numpy as np

from tbilby.core.prior.order_stats import TransdimensionalConditionalDescendingOrderStatPrior
from bilby.gw.transdimensional_source_models import make_glitch_signal_model
from bilby.gw.likelihood import GlitchGravitationalWaveTransient


logger = bilby.core.utils.logger
outdir = "glitch_reconstruction"
label = "GL1"
sampling_frequency = 2048.0
trigger_time = 300.0  # Arbitrary
maximum_frequency = 1024
minimum_frequency = 20
roll_off = 0.4
duration = 16
post_trigger_duration = 2
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

N = 8  # Max number of sine-Gaussian glitch components

# ── Data loading ───────────────────────────────────────────────────────────────
ifo_h1 = bilby.gw.detector.get_empty_interferometer('H1')
# TODO: replace with actual frame file once available:
# ifo_h1.set_strain_data_from_frame_file(
#     frame_file='path/to/H1.gwf',
#     sampling_frequency=sampling_frequency,
#     duration=duration,
#     start_time=start_time,
#     channel='H1:GDS-CALIB_STRAIN',
# )
ifo_h1.set_strain_data_from_power_spectral_density(
    sampling_frequency=sampling_frequency, duration=duration, start_time=start_time
)
ifo_list = bilby.gw.detector.InterferometerList([ifo_h1])

# ── Waveform generators ────────────────────────────────────────────────────────
# Glitch model for H1: SNR→amplitude conversion uses H1's PSD directly,
# no antenna-response weighting.
glitch_model_h1 = make_glitch_signal_model(N, ifo_h1)
glitch_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=glitch_model_h1,
    parameter_conversion=None,
    waveform_arguments=dict(
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    ),
)

# Zero-contribution GW generator: GlitchGravitationalWaveTransient requires a
# GW waveform generator; returning zeros makes this a glitch-only analysis.
# get_detector_response is still called internally and needs geocent_time, ra,
# dec, psi — those are fixed via DeltaFunction priors below.
def _zero_gw(frequency_array, **kwargs):
    return {
        'plus':  np.zeros_like(frequency_array, dtype=complex),
        'cross': np.zeros_like(frequency_array, dtype=complex),
    }

gw_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=_zero_gw,
    parameter_conversion=None,
)

# ── Likelihood ─────────────────────────────────────────────────────────────────
likelihood = GlitchGravitationalWaveTransient(
    interferometers=ifo_list,
    waveform_generator=gw_waveform_generator,
    glitch_waveform_generators={'H1': glitch_waveform_generator},
)

# ── Priors ─────────────────────────────────────────────────────────────────────
# Transdimensional descending-order-statistic prior on per-IFO SNR.
# Parameters are prefixed 'H1_' so the likelihood can unpack them for H1's
# glitch waveform generator via _glitch_parameters().
class TransdimensionalConditionalDescendingOrderStatPriorSNR(
        TransdimensionalConditionalDescendingOrderStatPrior):

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
        except:
            self._tot_order_num = int(self.H1_n)
        return dict(
            _prev_val=self._prev_val,
            _this_order_num=self._this_order_num,
            _tot_order_num=self._tot_order_num,
        )


priors = bilby.core.prior.dict.ConditionalPriorDict()
priors = tbilby.core.base.create_transdimensional_priors(
    transdimensional_prior_class=TransdimensionalConditionalDescendingOrderStatPriorSNR,
    param_name='H1_SNR',
    nmax=N,
    nested_conditional_transdimensional_params=['H1_SNR'],
    conditional_transdimensional_params=[],
    conditional_params=['H1_n'],
    prior_dict_to_add=priors,
    SaveConditionFunctionsToFile=False,
    minimum=0, maximum=30, prev_val=30, this_order_num=1,
)

priors['H1_n'] = tbilby.core.prior.DiscreteUniform(1, N, 'H1_n_dimension')

for i in range(N):
    priors[f'H1_dt{i}']  = bilby.core.prior.Uniform(-0.3, 0.2, name=f'H1_dt{i}')
    priors[f'H1_f{i}']   = bilby.core.prior.Uniform(
        minimum_frequency, maximum_frequency / 2, name=f'H1_f{i}')
    priors[f'H1_Q{i}']   = bilby.core.prior.Uniform(0.1, 40, name=f'H1_Q{i}')
    priors[f'H1_phi{i}']  = bilby.core.prior.Uniform(
        0, 2 * np.pi, name=f'H1_phi{i}', boundary='periodic')

# Sky parameters are fixed: the zero GW model ignores them, but
# get_detector_response still requires them in the parameter dict.
priors['geocent_time'] = bilby.core.prior.DeltaFunction(
    peak=trigger_time, name='geocent_time')
priors['ra']  = bilby.core.prior.DeltaFunction(peak=0.0, name='ra')
priors['dec'] = bilby.core.prior.DeltaFunction(peak=0.0, name='dec')
priors['psi'] = bilby.core.prior.DeltaFunction(peak=0.0, name='psi')

# ── Sampling ───────────────────────────────────────────────────────────────────
result = bilby.core.sampler.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    sample="rwalk",
    nlive=2000,
    nact=80,
    outdir=outdir,
    label=label,
    resume=True,
    npool=16,
)
