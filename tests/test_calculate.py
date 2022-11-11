#!/usr/bin/env python3
"""
Test will.calculate
"""
from unittest import mock

import numpy as np
import pytest
from your.formats.filwriter import make_sigproc_object

from will import calculate, create

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201

# pylint: disable=redefined-outer-name
# The pytest.fixture needs to be redefined


rng = np.random.default_rng(2022)


def test_std_min_func():
    """
    Test the std minimization function
    We know where it should give zeros
    """
    # if sigma is zero, std must also be zero
    assert calculate.std_min_func(0, 1, 0) == 0
    # calculate the expected variance for mu=sigma=1
    assert calculate.std_min_func(1, 1, np.sqrt((np.exp(1) - 1) * np.exp(2 + 1))) == 0


def test_log_normal_from_stats():
    """
    Test if the properties of the log normal distro are
    expected
    """
    median = 5e5
    std = 0.1 * median
    num_samples = 100
    distro = calculate.log_normal_from_stats(median, std, num_samples)

    # 17% seems generous enough
    assert np.std(distro) / std - 1 < 0.17
    assert np.median(distro) / median - 1 < 0.17
    assert len(distro) == num_samples


class TestQuickSort:
    """
    Test the quicksort
    """

    def setup_class(self):
        """
        Random array to sort
        """
        self.rands = rng.normal(size=20)

    def test_sort_assent(self):
        """
        Test quicksort ascending
        """
        rands_copy = self.rands.copy()
        calculate.quicksort(rands_copy, sort_ascend=True)
        np.testing.assert_allclose(rands_copy, np.sort(self.rands))

    def test_sort_descend(self):
        """
        Test quicksort descending
        """
        rands_copy = self.rands.copy()
        calculate.quicksort(rands_copy, sort_ascend=False)
        np.testing.assert_allclose(rands_copy, np.sort(self.rands)[::-1])

    def test_subsort_array(self):
        """
        Split an array in two and split
        """
        split = len(self.rands) // 2
        array_split = np.concatenate(
            [np.sort(self.rands[:split]), np.sort(self.rands[split:])[::-1]]
        )
        quicksort_split = self.rands.copy()
        quicksort_split = calculate.sort_subarrays(quicksort_split, num_subarrays=2)
        np.testing.assert_array_equal(quicksort_split, array_split)


class TestCalculateDMWidths:
    """
    Test the DM width calculator.
    """

    def setup_class(self):
        """
        Set up constants, use GREENBURST numbers
        """
        self.nchans = 4096
        self.chan_freqs = np.linspace(1919.8828125, 960.1171875, 4096)
        self.chan_width = 0.234375

    def test_zero_dm(self):
        """
        Should return zeros
        """
        assert not np.any(
            calculate.calculate_dm_widths(0, self.chan_width, self.chan_freqs)
        )

    def test_ends(self):
        """
        Test the end values, lengths, and all positive.
        """
        widths = calculate.calculate_dm_widths(100, self.chan_width, self.chan_freqs)
        assert self.chan_freqs.shape == widths.shape
        assert widths[0] == calculate.calculate_dm_widths(
            100, self.chan_width, self.chan_freqs[0]
        )
        assert widths[-1] == calculate.calculate_dm_widths(
            100, self.chan_width, self.chan_freqs[-1]
        )
        assert np.all(widths > 0)


class TestCalculateDMBoxcarWidths:
    """
    Test the DM width calculator.
    """

    def setup_class(self):
        """
        Set up constants, use GREENBURST numbers
        """
        self.nchans = 4096
        self.sampling_time = 0.000256
        self.chan_freqs = np.linspace(1919.8828125, 960.1171875, 4096)
        self.chan_width = 0.234375

    def test_zero_dm(self):
        """
        Should return ones.
        """
        widths = calculate.calculate_dm_boxcar_widths(
            0, self.sampling_time, self.chan_freqs
        )
        assert np.all(widths == 1)

    def test_high_dm(self):
        """
        Test 10000 DM.
        """
        dm = 10000
        widths = calculate.calculate_dm_boxcar_widths(
            dm, self.sampling_time, self.chan_freqs
        )
        assert widths[0] == 1
        sec_width = calculate.calculate_dm_widths(
            dm, self.chan_width, self.chan_freqs[-1]
        )
        width_samples = sec_width / self.sampling_time
        assert widths[-1] == np.around(width_samples)
        assert self.chan_freqs.shape == widths.shape


class TestGenerateBoxcarArray:
    """
    Test the boxcar array generator.
    """

    def setup_class(self):
        """
        Boxcar widths
        """
        self.num_boxcars = 8
        self.boxcar_widths = 2 ** np.arange(0, self.num_boxcars)
        self.sqrt_array, self.max_boxcar = calculate.generate_boxcar_array(
            self.boxcar_widths, return_max=True
        )

    def test_size(self):
        """
        Test the size of the array.
        """
        num_rows, num_cols = self.sqrt_array.shape
        assert num_rows == self.boxcar_widths.max()
        assert num_cols == self.num_boxcars
        assert self.boxcar_widths.max() == self.max_boxcar

    def test_normalization_sqrt(self):
        """
        Test the normalization of the boxcars
        for Gaussian noise (scale sqrt)
        """
        np.testing.assert_almost_equal(
            self.sqrt_array.sum(axis=0), np.sqrt(self.boxcar_widths)
        )

    def test_normalization_unity(self):
        """
        Test the normalization of the boxcars
        for power preserving boxcar.
        """
        unity_array = calculate.generate_boxcar_array(self.boxcar_widths, lambda x: x)
        np.testing.assert_almost_equal(
            unity_array.sum(axis=0), np.ones(self.num_boxcars)
        )


class TestConvolveMultiBoxcar:
    """
    Test the multi boxcar convolver
    """

    def setup_class(self):
        """
        Boxcar widths
        """
        self.num_boxcars = 3
        self.boxcar_widths = 2 ** np.arange(0, self.num_boxcars)
        self.max_boxcar = self.boxcar_widths.max()
        self.sqrt_array = calculate.generate_boxcar_array(self.boxcar_widths)
        self.profile = np.zeros(2 * self.max_boxcar)
        self.profile[self.max_boxcar] = 1

    def test_single_profile(self):
        """
        Test a single profile, sqrt normalization.
        """

        convolved = calculate.convolve_multi_boxcar(self.profile, self.sqrt_array)
        num_rows, num_cols = convolved.shape
        assert self.num_boxcars == num_cols
        assert len(self.profile) == num_rows

        above_zero = convolved > 0.1
        np.testing.assert_equal(above_zero.sum(axis=0), self.boxcar_widths)
        np.testing.assert_almost_equal(
            convolved.sum(axis=0), np.sqrt(self.boxcar_widths)
        )

    def test_single_profile_unity(self):
        """
        Test a single profile, unity normalization.
        """
        unity_array = calculate.generate_boxcar_array(self.boxcar_widths, lambda x: x)
        convolved = calculate.convolve_multi_boxcar(self.profile, unity_array)
        total_power = convolved.sum(axis=0)
        np.testing.assert_allclose(total_power, np.ones_like(total_power))

    def test_high_dimension_error(self):
        """
        If profile is above two, raise NotImplementedError
        """
        with pytest.raises(NotImplementedError):
            calculate.convolve_multi_boxcar(
                self.profile[:, None, None], self.sqrt_array
            )


def test_boxcar_convolved():
    """
    Test the boxcar convolver
    """
    time_profile = np.zeros(10)
    # len 2 boxcar
    widths = np.arange(1, 4)
    time_profile[4:6] = 1
    convolved = calculate.boxcar_convolved(time_profile, widths)

    assert convolved.argmax() == 1
    assert convolved[0] < convolved.max()
    assert convolved[2] < convolved.max()


class TestMedianDetrend:
    """
    Test the median_line_detrend.
    """

    @staticmethod
    def test_base():
        """
        Test no padding
        """
        num_samps = 2**14
        noise = rng.normal(size=num_samps)
        times = np.linspace(0, 1, num_samps)
        noise_with_tend = noise + 5 * times + 4
        detrend = calculate.median_line_detrend(noise_with_tend, num_sample=1024)
        assert detrend.size == num_samps
        fit = np.polyfit(times, detrend, deg=1)
        assert (np.abs(fit) < np.array((0.1, 0.1))).all()

    @staticmethod
    def test_pad():
        """
        Test padding.
        """
        num_samps = 2**14 - 1
        noise = rng.normal(size=num_samps)
        times = np.linspace(0, 1, num_samps)
        noise_with_tend = noise + 5 * times + 4
        detrend = calculate.median_line_detrend(noise_with_tend, num_sample=1024)
        assert detrend.size == num_samps
        fit = np.polyfit(times, detrend, deg=1)
        assert (np.abs(fit) < np.array((0.1, 0.1))).all()


def test_calculate_noises_multi():
    """
    Make a Gaussian time series, check if
    it integrates down as expected.
    """

    time_series = rng.normal(size=2**14)
    boxcar_lengths = np.array([2**x for x in range(0, 8)])
    boxcar_array, max_boxcar = calculate.generate_boxcar_array(
        boxcar_lengths, return_max=True, normalization_func=lambda x: x
    )
    noises = calculate.calculate_noises_multi(
        time_series, boxcar_array=boxcar_array, max_boxcar=max_boxcar
    )
    np.testing.assert_allclose(noises, 1 / np.sqrt(boxcar_lengths), rtol=0.3)


def test_NoiseInfoResult():
    """
    Test the NoiseInfoResult dataclass
    """
    num_samps = 10
    num_boxcars = 8
    noises = rng.integers(10, size=num_samps * num_boxcars).reshape(
        num_samps, num_boxcars
    )
    noise_info = calculate.NoiseInfoResult(
        noises, np.array([2**x for x in range(0, num_boxcars)]), 4096
    )
    np.testing.assert_allclose(noise_info.mean_noise_levels, noises.mean(axis=0))
    np.testing.assert_allclose(
        noise_info.median_noise_levels, np.median(noises, axis=0)
    )

    with mock.patch("matplotlib.pyplot.show") as show:
        noise_info.plot_noise()
        show.assert_called_once()


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


class TestNoiseInfo:
    """
    Test noise_info on a fake filterbank.
    """

    def setup_class(self):
        """
        Use same boxcar lengths, test locations
        """
        self.boxcar_lengths = np.array([2**x for x in range(0, 4)])
        self.num_locations = 10

    def test_base(self, create_fil):
        """
        Test the base case.
        """

        noise_info = calculate.noise_info(
            create_fil[0],
            dm=0,
            boxcar_lengths=self.boxcar_lengths,
            num_locations=self.num_locations,
            num_samples=2**6,
        )
        np.testing.assert_allclose(self.boxcar_lengths, noise_info.boxcars_lengths)
        assert NCHANS == noise_info.num_chans

    def test_dm(self, create_fil):
        """
        Test the DM != 0.
        """
        noise_info = calculate.noise_info(
            create_fil[0],
            dm=10,
            boxcar_lengths=self.boxcar_lengths,
            num_locations=self.num_locations,
            num_samples=2**6,
        )
        assert len(noise_info.noise_levels[0]) == len(self.boxcar_lengths)

    def test_mask(self, create_fil):
        """
        Test with mask.
        """
        mask = np.zeros(NCHANS, dtype=bool)
        mask[10:20] = True
        noise_info = calculate.noise_info(
            create_fil[0],
            dm=10,
            boxcar_lengths=self.boxcar_lengths,
            num_locations=self.num_locations,
            num_samples=2**6,
            chan_mask=mask,
        )
        assert len(noise_info.mean_noise_levels) == len(self.boxcar_lengths)

    def test_raise_error(self, create_fil):
        """
        Should raise error is noise sections overlap.
        """
        with pytest.raises(ValueError):
            calculate.noise_info(
                create_fil[0],
                dm=10,
                boxcar_lengths=self.boxcar_lengths,
                num_locations=self.num_locations,
                num_samples=2**10,
            )
