import numpy as np
import matplotlib.pyplot as plt
import copy

from bilby.gw.source import lal_binary_neutron_star
from bilby.gw.transdimensional_source_models import bns_with_resonances_factory
from bilby.gw.likelihood.relative import RelativeBinningGravitationalWaveTransient
import bilby

from tbilby.core.prior.order_stats import TransdimensionalConditionalAscendingOrderStatPrior
import tbilby


logger = bilby.core.utils.logger
outdir = "NS_resonance_transdimensional"
label = "NS_resonance"
sampling_frequency = 8192.0
trigger_time = 1384782888.6
detectors = ["ET"]
maximum_frequency = sampling_frequency // 2
minimum_frequency = 5 
roll_off = 0.4
duration = 1500 # s
post_trigger_duration = 5 #s
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time

N = 2
bns_with_resonances = bns_with_resonances_factory(N)

parameter_dict = {
    'mass_1': 1.4,
    'mass_2': 1.0,
    'a_1': 0.02,
    'a_2': 0.01,
    'lambda_1': 400.0,
    'lambda_2': 600.0,
    'tilt_1': 0.0,
    'tilt_2': 0.0,
    'phi_12': 0.0,
    'phi_jl': 0.0,
    'luminosity_distance': 40,  # Mpc
    'theta_jn': np.pi / 3.0,
    'phase': 0.0,
    'ra': 0.0,
    'dec': 0.0,
    'geocent_time': trigger_time,
    'psi': 0.0,
    'n': N,
    'f00': 40.0,
    'f01': 200.0,
    'dphi0': 1e-2,
    'dphi1': 1.0,
}

mc_injected = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1=parameter_dict['mass_1'], mass_2=parameter_dict['mass_2'])
q_injected = bilby.gw.conversion.component_masses_to_mass_ratio(mass_1=parameter_dict['mass_1'], mass_2=parameter_dict['mass_2'])

parameter_dict['chirp_mass'] = mc_injected
parameter_dict['mass_ratio'] = q_injected

# ── Waveform generators ────────────────────────────────────────────────────────
waveform_arguments = dict(
    minimum_frequency=minimum_frequency,
    maximum_frequency=maximum_frequency,
    reference_frequency=minimum_frequency,
    waveform_approximant="IMRPhenomXAS_NRTidalv3",
)
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bns_with_resonances,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)


# ── Setup ifos ───────────────────────────────────────────────────────────────
ifo_list = bilby.gw.detector.InterferometerList(detectors)
ifo_list.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)

print("Injecting signal...")
ifo_list.inject_signal(
    waveform_generator=waveform_generator, 
    parameters=parameter_dict,
    earth_rotation=True
)

def get_network_snr(ifos, wf_gen, parameters):
    ifos = copy.deepcopy(ifos) # Don't modify the original ifos
    network_snr = 0.0
    injection_polarisations = wf_gen.frequency_domain_strain(parameters)

    for ifo in ifos:
        ifo_polarisations = ifo.get_detector_response(injection_polarisations, parameters, earth_rotation=True)
        ifo_snr_squared = ifo.optimal_snr_squared(ifo_polarisations)
        network_snr += ifo_snr_squared

    return np.sqrt(network_snr)

network_snr = get_network_snr(ifo_list, waveform_generator, parameter_dict)
print(f"Network SNR: {network_snr:.2f}")


# ── Likelihood ─────────────────────────────────────────────────────────────────
print("Setting up likelihood...")
likelihood = RelativeBinningGravitationalWaveTransient(
    interferometers=ifo_list,
    waveform_generator=waveform_generator,
    fiducial_parameters=parameter_dict, # for now we pass the injection parameters as the fiducial parameters, in reality we don't necessarily know them a priori
)


# ── Priors ─────────────────────────────────────────────────────────────────────
print("Setting up transdimensional priors...")
# Define the transdimensional prior dictionary for the resonance parameters
class ResonanceTransdimensionalfrequencyPrior(TransdimensionalConditionalAscendingOrderStatPrior):
    def transdimensional_condition_function(self, **required_variables):
        if len(self.f0) > 0:
            self._prev_val = self.f0[-1]
            self._this_order_num = self.f0.shape[0] + 1
        else:
            self._this_order_num = 1
            if isinstance(self.n, np.ndarray):
                self._prev_val = self.minimum * np.ones(self.n.shape)
        try:
            self._tot_order_num = self.n.astype(int)
        except Exception:
            self._tot_order_num = int(self.n)
        return dict(
            _prev_val=self._prev_val,
            _this_order_num=self._this_order_num,
            _tot_order_num=self._tot_order_num,
        )

priors = bilby.core.prior.dict.ConditionalPriorDict()

### Set up resonance priors
priors = tbilby.core.base.create_transdimensional_priors(
    transdimensional_prior_class=ResonanceTransdimensionalfrequencyPrior,
    param_name='f0',
    nmax=N,
    nested_conditional_transdimensional_params=['f0'],
    conditional_transdimensional_params=[],
    conditional_params=['n'],
    prior_dict_to_add=priors,
    SaveConditionFunctionsToFile=False,
    minimum=minimum_frequency, maximum=350.0, prev_val=minimum_frequency, this_order_num=1,
)

priors['n'] = tbilby.core.prior.DiscreteUniform(0, N, 'N_resonances')
for i in range(N):
    priors[f'dphi{i}'] = bilby.core.prior.LogUniform(1e-4, 10, f'dphi{i}')


print("Setting up bilby priors...")
### Set up BNS priors
    # Use tight prior on chirp mass as this is well-measured
    # Fix sky location and tc as earth rotation will constrain these well

priors['chirp_mass'] = bilby.core.prior.Gaussian(
    mu=mc_injected, sigma=0.1, name='chirp_mass', latex_label='$\\mathcal{M}$', unit='$M_\\odot$')
priors['mass_ratio'] = bilby.core.prior.Uniform(
    minimum=0.5, maximum=1.0, name='mass_ratio', latex_label='$q$')
priors['luminosity_distance'] = bilby.core.prior.PowerLaw(
    alpha=0.2, minimum=40, maximum=1000,
    name='luminosity_distance', latex_label='$d_L$', unit='Mpc')
priors['geocent_time'] = bilby.core.prior.DeltaFunction(
    peak=trigger_time, name='geocent_time')
priors['ra'] = bilby.core.prior.DeltaFunction(
    peak=0, name='ra')
priors['dec'] = bilby.core.prior.DeltaFunction(
    peak=0, name='dec')
priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
priors['psi'] = bilby.core.prior.Uniform(
    0, np.pi, name='psi', boundary='periodic')
priors['phase'] = bilby.core.prior.Uniform(
    0, 2 * np.pi, name='phase', boundary='periodic')
priors['a_1'] = bilby.core.prior.Uniform(0, 0.05, name='a_1') # NSs are expected to have low spins, so we restrict the prior to a small range
priors['a_2'] = bilby.core.prior.Uniform(0, 0.05, name='a_2')


# ── Sampling ───────────────────────────────────────────────────────────────────
print("Sampling...")
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
    npool=64,
)
