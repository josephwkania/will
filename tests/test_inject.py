#!/usr/bin/env python3
"""
Test Pulse injection routines.
"""

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201

# pylint: disable=redefined-outer-name
# The pytest.fixture needs to be redefined

import numpy as np
import pytest
from your import Your
from your.formats.filwriter import make_sigproc_object

from will import create, inject

DM = 20
NUM_SAMP = 2**10
NCHANS = 512
CHAN_FREQS = np.linspace(1100, 1000, NCHANS)
TSAMP = 0.000256


@pytest.fixture(scope="session")
def create_fil(tmpdir_factory):
    """
    Create a filter bank to test
    """

    out_name = tmpdir_factory.mktemp("data").join("test.fil")
    out_name = str(out_name)

    medians = 15 * np.ones(NCHANS)
    medians[10:15] += 10
    stds = 5 * np.ones(NCHANS)
    stds[60:70] += 14
    dynamic = create.dynamic_from_statistics(
        medians, stds, dtype=np.uint8, nsamps=NUM_SAMP
    )

    sigproc_object = make_sigproc_object(
        rawdatafile=out_name,
        source_name="fake",
        nchans=NCHANS,
        foff=-np.abs(CHAN_FREQS[0] - CHAN_FREQS[1]),  # MHz
        fch1=CHAN_FREQS.max(),  # MHz
        tsamp=TSAMP,  # seconds
        tstart=59246,  # MJD
        src_raj=112233.44,  # HHMMSS.SS
        src_dej=112233.44,  # DDMMSS.SS
        machine_id=0,
        nbeams=0,
        ibeam=0,
        nbits=8,
        nifs=1,
        barycentric=0,
        pulsarcentric=0,
        telescope_id=6,
        data_type=0,
        az_start=-1,
        za_start=-1,
    )

    sigproc_object.write_header(out_name)
    sigproc_object.append_spectra(dynamic, out_name)
    return out_name, dynamic


class TestInjectConstant:
    """
    Inject constant into file
    """

    def setup_class(self):
        """
        Make the pulse
        """
        simple_pulse = create.SimpleGaussPulse(
            0.01,
            dm=DM,
            tau=5,
            chan_freqs=CHAN_FREQS,
            sigma_freq=100,
            center_freq=1050,
            tsamp=TSAMP,
            spectral_index_alpha=1,
            nscint=1,
            phi=1,
        )
        self.pulse = simple_pulse.sample_pulse(int(4e5))

    def test_inject_constant(self, create_fil):
        """
        Inject a constant pulse into the fake fil
        """
        yr_obj = Your(create_fil[0])
        inj_pulse = inject.inject_constant_into_file(  # noqa: max-complexity:30
            yr_obj,
            self.pulse,
            start=0,
            period=100,
            gulp=NUM_SAMP,
            out_fil=None,
            clip_powers=True,
        )

        assert inj_pulse.dtype == yr_obj.your_header.dtype
        assert inj_pulse.shape == (NUM_SAMP, NCHANS)

    def test_chan_error(self, create_fil):
        """
        If this pulse is the wrong size raise an error
        """
        yr_obj = Your(create_fil[0])
        with pytest.raises(RuntimeError):
            inject.inject_constant_into_file(  # noqa: max-complexity:30
                yr_obj,
                self.pulse[:, :10],
                start=0,
                period=100,
                gulp=NUM_SAMP,
                out_fil=None,
                clip_powers=True,
            )

    def test_gulp_error(self, create_fil):
        """
        If this pulse is the wrong size raise an error
        """
        yr_obj = Your(create_fil[0])
        with pytest.raises(RuntimeError):
            inject.inject_constant_into_file(  # noqa: max-complexity:30
                yr_obj,
                self.pulse[:, :10],
                start=0,
                period=100,
                gulp=1,
                out_fil=None,
                clip_powers=True,
            )


def test_num_pulses():
    """
    Test if the number of pulses is reasonable
    """
    period = 100
    npulse = inject.num_pulses(period, NUM_SAMP)
    assert npulse > 0
    assert isinstance(npulse, np.integer)


class TestInjectDistro:
    """
    Test injecting a distributution
    """

    def setup_class(self):
        """
        Create the pulse
        """
        self.simple_pulse = create.SimpleGaussPulse(
            0.01,
            dm=DM,
            tau=5,
            chan_freqs=CHAN_FREQS,
            sigma_freq=100,
            center_freq=1050,
            tsamp=TSAMP,
            spectral_index_alpha=1,
            nscint=1,
            phi=1,
        )

        self.period = 100
        npulse = inject.num_pulses(self.period, NUM_SAMP)
        powers = 10e2 + 10 * np.random.normal(size=npulse)
        np.around(powers, out=powers)
        self.powers = powers.astype(int)

    def test_inject_distro(self, create_fil):
        """
        Inject a distrbution into the file
        """
        yr_obj = Your(create_fil[0])
        pulse_distro = inject.inject_distribution_into_file(
            yr_obj,
            self.simple_pulse,
            pulse_counts=self.powers,
            start=0,
            period=self.period,
            gulp=NUM_SAMP,
            out_fil=None,
            clip_powers=True,
        )

        assert pulse_distro.dtype == yr_obj.your_header.dtype
        assert pulse_distro.shape == (NUM_SAMP, NCHANS)

    def test_gulp_error(self, create_fil):
        """
        If the gulp is too small, raise an error
        """
        yr_obj = Your(create_fil[0])
        with pytest.raises(RuntimeError):
            inject.inject_distribution_into_file(
                yr_obj,
                self.simple_pulse,
                pulse_counts=self.powers,
                start=0,
                period=self.period,
                gulp=1,
                out_fil=None,
                clip_powers=True,
            )
