#!/usr/bin/env python3
"""
Test Pulse detection routines.
"""

# pylint: disable=redefined-outer-name
# The pytest.fixture needs to be redefined

from unittest import mock

import numpy as np
import pytest
from your.formats.filwriter import make_sigproc_object

from will import create, detect

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
    pulse = simple_pulse.sample_pulse(int(4e5))
    dynamic[64 : 64 + len(pulse)] += pulse
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


def test_find_first_pulse(create_fil):
    """
    Test if find_first_pulse calls matplotlib
    """
    with mock.patch("matplotlib.pyplot.show") as show:
        detect.find_first_pulse(create_fil[0], start=0, dm=DM, gulp=NUM_SAMP)
        show.assert_called_once()

    with mock.patch("matplotlib.pyplot.show") as show:
        detect.find_first_pulse(
            create_fil[0], start=0, dm=DM, gulp=NUM_SAMP, box_car_length=4
        )
        show.assert_called_once()


# tried to combine theses three info class but unsure how
# to handle the pytest.fixture argument
def test_dedisped_time_series(create_fil):
    """
    Test the time series maker
    """
    time_series = detect.dedisped_time_series(
        create_fil[1], dm=DM, tsamp=TSAMP, chan_freqs=CHAN_FREQS
    )
    assert time_series.size == NUM_SAMP


def test_find_all_pulses(create_fil):
    """
    Test find all pusles
    """
    time_series = detect.dedisped_time_series(
        create_fil[1], dm=DM, tsamp=TSAMP, chan_freqs=CHAN_FREQS
    )
    all_pulses = detect.detect_all_pulses(time_series=time_series, box_car_length=111)

    assert len(all_pulses.locations) > 0
    assert len(all_pulses.snrs) == len(all_pulses.locations)
    assert all_pulses.std > 0

    max_pulse = detect.find_max_pulse(all_pulses)
    assert NUM_SAMP > max_pulse.location > 0
    assert max_pulse.snr > 0


def test_find_max_pulse(create_fil):
    """
    Find the max Pulse
    """
    time_series = detect.dedisped_time_series(
        create_fil[1], dm=DM, tsamp=TSAMP, chan_freqs=CHAN_FREQS
    )
    all_pulses = detect.detect_max_pulse(time_series=time_series, box_car_length=111)

    assert NUM_SAMP > all_pulses.locations > 0
    assert all_pulses.snrs > 0
    assert all_pulses.std > 0


def test_location_pules(create_fil):
    """
    Tests location of pulses
    """
    offset = 164  # = simple_pulse.pulse_center
    pulse_params = detect.PulseSearchParamters(
        create_fil[0],
        first_pulse=offset,
        period=128 * TSAMP,  # in seconds
        dm=DM,
        box_car_length=111,
        samples_around_pulse=32,
    )
    pulse_locations = detect.locations_of_pulses(pulse_params)
    assert len(pulse_locations) > 0
    # 1st instance is too close to the edge, to period gets added
    assert pulse_locations[0] == offset
    np.testing.assert_almost_equal(offset + 128, pulse_locations[1])


def test_search_file(create_fil):
    """
    Test search file and PulseSNRs class
    """
    offset = 164  # = simple_pulse.pulse_center
    pulse_params = detect.PulseSearchParamters(
        create_fil[0],
        first_pulse=offset + 64,
        period=128 * TSAMP,  # in seconds
        dm=DM,
        box_car_length=111,
        samples_around_pulse=128,
    )
    pulse_locations = detect.locations_of_pulses(pulse_params)
    pulses = detect.search_file(pulse_params, pulse_locations)

    assert len(pulses.snrs) > 0
    assert pulses.stds.shape == pulses.snrs.shape
    assert pulses.folded.shape == (111 + 128 * 2, NCHANS)

    assert 100 >= pulses.percent_with_pulses >= 0
    folded = pulses.folded_properties
    assert folded.locations > 0
    assert folded.snrs > 0
    assert folded.std > 0

    # Not sure why this test is failing. The plot shows if not Mocked
    # Maybe because it is a property?
    # put the show to make flake8 happy
    with mock.patch("matplotlib.pyplot.show") as show:
        pulses.plot_snrs()
        show.assert_called_once()

    with mock.patch("matplotlib.pyplot.show") as show:
        pulses.plot_stds()
        show.assert_called_once()

    with mock.patch("matplotlib.pyplot.show") as show:
        pulses.plot_folded_profile()
        show.assert_called_once()

    with mock.patch("matplotlib.pyplot.show") as show:
        pulses.plot_folded_dynamic()
        show.assert_called_once()


def test_process_dynamic_spectra():
    """
    Test the cadidate the preprocessor
    """
    size = 10
    dtype = np.float32
    fake_cand = np.ones((size, size))
    fake_cand += 10 * np.linspace(0, size, num=size)
    fake_cand = fake_cand.astype(np.int8)
    processed, centering_mean = detect.process_dynamic_spectra(
        fake_cand, sigma=1, dtype=dtype
    )
    np.testing.assert_allclose(np.median(processed, axis=0), np.zeros(size))
    # bandpass gets subtracted off my median removal
    # mean should be close to zero
    np.testing.assert_almost_equal(processed.mean(), 0)
    np.testing.assert_almost_equal(centering_mean, 0)
    assert processed.dtype == dtype


def test_extract_pulses(create_fil):
    """
    Test the extract pulses
    """
    offset = 164  # = simple_pulse.pulse_center
    pulse_params = detect.PulseSearchParamters(
        create_fil[0],
        first_pulse=offset + 64,
        period=128 * TSAMP,  # in seconds
        dm=DM,
        box_car_length=111,
        samples_around_pulse=128,
    )
    pulse_locations = detect.locations_of_pulses(pulse_params)
    pulses = detect.extract_pulses(pulse_params, pulse_locations)

    assert len(pulses.dynamic_spectra) == len(pulse_locations)
    assert len(pulses.times) == len(pulse_locations)
    assert len(pulses.bandpass_labels) > 0
    assert len(pulses.dynamic_spectra) == len(pulses.centering_means)
