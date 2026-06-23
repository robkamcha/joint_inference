"""
Tests for GlitchGravitationalWaveTransient.

Strategy: inject signals into zero-noise data so that the injected parameters
are the exact maximum-likelihood point. Each test class covers one of the five
scenarios described in the test plan.
"""

import unittest
import numpy as np

import bilby.gw.detector as det
import bilby.gw.waveform_generator as wfg
import bilby.gw.likelihood as lk
import bilby.gw.source as src
from bilby.gw.transdimensional_source_models import make_glitch_signal_model


# ── shared constants ──────────────────────────────────────────────────────────

DURATION = 4
SAMPLING_FREQUENCY = 512
START_TIME = 0.0

# sinegaussian is used as a lightweight GW proxy (no LAL required)
GW_PARAMS = dict(
    hrss=1e-22, Q=5.0, frequency=80.0,
    ra=0.5, dec=0.3, geocent_time=0.0, psi=0.5,
)

# Single-component glitch for H1
H1_GLITCH_1 = dict(n=1, SNR0=10.0, f0=100.0, Q0=5.0, phi0=0.0, dt0=0.0)

# Two-component glitches for the network test
H1_GLITCH_2 = dict(
    n=2,
    SNR0=10.0, f0=100.0, Q0=5.0, phi0=0.0, dt0=0.0,
    SNR1=7.0,  f1=150.0, Q1=3.0, phi1=1.0, dt1=0.1,
)
L1_GLITCH_2 = dict(
    n=2,
    SNR0=8.0,  f0=60.0,  Q0=4.0, phi0=0.5, dt0=0.0,
    SNR1=5.0,  f1=200.0, Q1=6.0, phi1=2.0, dt1=-0.05,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _prefix(ifo_name, params):
    """Return a copy of params with every key prefixed by '<ifo_name>_'."""
    return {f'{ifo_name}_{k}': v for k, v in params.items()}


def _make_ifo(name):
    """Zero-noise IFO with PSD initialised (needed for inner products)."""
    ifo = det.get_empty_interferometer(name)
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=SAMPLING_FREQUENCY,
        duration=DURATION,
        start_time=START_TIME,
    )
    ifo.strain_data.frequency_domain_strain[:] = 0.0
    return ifo


def _make_gw_wfg():
    return wfg.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=src.sinegaussian,
        parameter_conversion=None,
    )


def _make_glitch_wfg(ifo, n_max):
    return wfg.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=make_glitch_signal_model(n_max, ifo),
        parameter_conversion=None,
    )


def _inject_gw(ifo_list, gw_wfg, gw_params):
    """Inject GW polarisations into every IFO via the antenna response."""
    pols = gw_wfg.frequency_domain_strain(gw_params)
    for ifo in ifo_list:
        ifo.inject_signal_from_waveform_polarizations(gw_params, pols)


def _inject_glitch(ifo, glitch_wfg, glitch_params):
    """Add glitch strain directly to IFO data (no antenna projection)."""
    pols = glitch_wfg.frequency_domain_strain(glitch_params)
    ifo.strain_data.frequency_domain_strain += pols['plus']


# ── test cases ────────────────────────────────────────────────────────────────

class TestNoGlitchMatchesStandard(unittest.TestCase):
    """
    With no glitch generators, GlitchGravitationalWaveTransient must produce
    exactly the same log-likelihood ratio as GravitationalWaveTransient.
    """

    def setUp(self):
        np.random.seed(0)
        self.ifo = _make_ifo('H1')
        self.ifo_list = det.InterferometerList([self.ifo])
        self.gw_wfg = _make_gw_wfg()
        _inject_gw(self.ifo_list, self.gw_wfg, GW_PARAMS)

        self.standard = lk.GravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
        )
        self.glitch_lk = lk.GlitchGravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
        )

    def test_llr_matches_at_injection(self):
        self.assertAlmostEqual(
            self.standard.log_likelihood_ratio(GW_PARAMS),
            self.glitch_lk.log_likelihood_ratio(GW_PARAMS),
            places=10,
        )

    def test_llr_matches_at_off_target_frequency(self):
        params = {**GW_PARAMS, 'frequency': 150.0}
        self.assertAlmostEqual(
            self.standard.log_likelihood_ratio(params),
            self.glitch_lk.log_likelihood_ratio(params),
            places=10,
        )


class TestSingleIFOGlitch(unittest.TestCase):
    """
    Single H1 detector with a one-component sine-Gaussian glitch and a GW signal.
    The injected (GW + glitch) parameters should be the maximum-likelihood point.
    """

    def setUp(self):
        np.random.seed(1)
        self.ifo = _make_ifo('H1')
        self.ifo_list = det.InterferometerList([self.ifo])
        self.gw_wfg = _make_gw_wfg()
        self.g_wfg = _make_glitch_wfg(self.ifo, n_max=1)

        _inject_gw(self.ifo_list, self.gw_wfg, GW_PARAMS)
        _inject_glitch(self.ifo, self.g_wfg, H1_GLITCH_1)

        self.likelihood = lk.GlitchGravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
            glitch_waveform_generators={'H1': self.g_wfg},
        )
        self.injection_params = {**GW_PARAMS, **_prefix('H1', H1_GLITCH_1)}

    def test_injection_gives_positive_llr(self):
        """In zero noise, signal + glitch at injection parameters must beat the noise hypothesis."""
        self.assertGreater(self.likelihood.log_likelihood_ratio(self.injection_params), 0.0)

    def test_injection_beats_wrong_glitch_frequency(self):
        perturbed = {**self.injection_params, 'H1_f0': 250.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_injection_beats_wrong_glitch_snr(self):
        perturbed = {**self.injection_params, 'H1_SNR0': 0.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_injection_beats_wrong_gw_frequency(self):
        perturbed = {**self.injection_params, 'frequency': 200.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )


class TestNetworkSingleIFOGlitch(unittest.TestCase):
    """
    H1 + L1 network; GW signal in both detectors, one-component glitch in H1 only.
    Tests that:
    - injection params give positive LLR
    - injection beats wrong glitch or GW params
    - the glitch likelihood beats the standard likelihood on the same glitchy data
    """

    def setUp(self):
        np.random.seed(2)
        self.h1 = _make_ifo('H1')
        self.l1 = _make_ifo('L1')
        self.ifo_list = det.InterferometerList([self.h1, self.l1])
        self.gw_wfg = _make_gw_wfg()
        self.g_wfg_h1 = _make_glitch_wfg(self.h1, n_max=1)

        _inject_gw(self.ifo_list, self.gw_wfg, GW_PARAMS)
        _inject_glitch(self.h1, self.g_wfg_h1, H1_GLITCH_1)

        self.likelihood = lk.GlitchGravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
            glitch_waveform_generators={'H1': self.g_wfg_h1},
        )
        self.injection_params = {**GW_PARAMS, **_prefix('H1', H1_GLITCH_1)}

    def test_injection_gives_positive_llr(self):
        self.assertGreater(self.likelihood.log_likelihood_ratio(self.injection_params), 0.0)

    def test_injection_beats_wrong_glitch_frequency(self):
        perturbed = {**self.injection_params, 'H1_f0': 250.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_injection_beats_wrong_gw_frequency(self):
        perturbed = {**self.injection_params, 'frequency': 200.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_glitch_model_beats_standard_on_glitchy_data(self):
        """
        The standard GWT evaluated on the GW-only params sees the H1 glitch as
        unexplained power in the data; the glitch likelihood should give a
        higher LLR by accounting for it.
        """
        standard = lk.GravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
        )
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            standard.log_likelihood_ratio(GW_PARAMS),
        )


class TestNetworkMultiGlitch(unittest.TestCase):
    """
    H1 + L1 network; GW signal in both detectors, two independent glitch
    components in each detector.
    """

    def setUp(self):
        np.random.seed(3)
        self.h1 = _make_ifo('H1')
        self.l1 = _make_ifo('L1')
        self.ifo_list = det.InterferometerList([self.h1, self.l1])
        self.gw_wfg = _make_gw_wfg()
        self.g_wfg_h1 = _make_glitch_wfg(self.h1, n_max=2)
        self.g_wfg_l1 = _make_glitch_wfg(self.l1, n_max=2)

        _inject_gw(self.ifo_list, self.gw_wfg, GW_PARAMS)
        _inject_glitch(self.h1, self.g_wfg_h1, H1_GLITCH_2)
        _inject_glitch(self.l1, self.g_wfg_l1, L1_GLITCH_2)

        self.likelihood = lk.GlitchGravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
            glitch_waveform_generators={
                'H1': self.g_wfg_h1,
                'L1': self.g_wfg_l1,
            },
        )
        self.injection_params = {
            **GW_PARAMS,
            **_prefix('H1', H1_GLITCH_2),
            **_prefix('L1', L1_GLITCH_2),
        }

    def test_injection_gives_positive_llr(self):
        self.assertGreater(self.likelihood.log_likelihood_ratio(self.injection_params), 0.0)

    def test_injection_beats_wrong_h1_glitch_frequencies(self):
        perturbed = {**self.injection_params, 'H1_f0': 250.0, 'H1_f1': 300.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_injection_beats_wrong_l1_glitch_frequencies(self):
        perturbed = {**self.injection_params, 'L1_f0': 250.0, 'L1_f1': 300.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_injection_beats_wrong_gw_frequency(self):
        perturbed = {**self.injection_params, 'frequency': 200.0}
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            self.likelihood.log_likelihood_ratio(perturbed),
        )

    def test_glitch_model_beats_standard_on_glitchy_data(self):
        standard = lk.GravitationalWaveTransient(
            interferometers=self.ifo_list,
            waveform_generator=self.gw_wfg,
        )
        self.assertGreater(
            self.likelihood.log_likelihood_ratio(self.injection_params),
            standard.log_likelihood_ratio(GW_PARAMS),
        )


if __name__ == '__main__':
    unittest.main()
