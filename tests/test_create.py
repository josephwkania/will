#!/usr/bin/env python3
"""
Test Pulse creation routines.
"""

# Can't use inits with pytest, this error is unavoidable
# pylint: disable=W0201

import numpy as np
import pytest

from will import create

rng = np.random.default_rng(2022)


def test_std_min_func():
    """
    Test the std minimization function
    We know where it should give zeros
    """
    # if sigma is zero, std must also be zero
    assert create.std_min_func(0, 1, 0) == 0
    # calculate the expected varience for mu=sigma=1
    assert create.std_min_func(1, 1, np.sqrt((np.exp(1) - 1) * np.exp(2 + 1))) == 0


def test_log_normal_from_stats():
    """
    Test if the properties of the log normal distro are
    expected
    """
    median = 5e5
    std = 0.1 * median
    num_samples = 100
    distro = create.log_normal_from_stats(median, std, num_samples)

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
        Test quicksort assending
        """
        rands_copy = self.rands.copy()
        create.quicksort(rands_copy, sort_assend=True)
        np.testing.assert_allclose(rands_copy, np.sort(self.rands))

    def test_sort_desend(self):
        """
        Test quicksort desending
        """
        rands_copy = self.rands.copy()
        create.quicksort(rands_copy, sort_assend=False)
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
        quicksort_split = create.sort_subarrays(quicksort_split, num_subarrays=2)
        np.testing.assert_array_equal(quicksort_split, array_split)


def test_gaussian():
    """
    Test Gauss function
    """
    size = 10
    gauss = create.gaussian(np.linspace(0, 10, size), mu=10, sig=10)
    assert gauss.size == size
    np.testing.assert_allclose(gauss.max(), 1)


def test_skew_pulse():
    """
    Test the skewed pulse.
    """
    size = 5
    time_indices = np.arange(0, size)
    chan_indices = np.arange(0, size)
    chan_indices, time_indices = np.meshgrid(chan_indices, time_indices)
    gauss = create.skewed_gauss(
        x=chan_indices,
        y=time_indices,
        x_mu=size / 2,
        y_mu=size / 2,
        x_sig=1,
        y_sig=1,
        theta=np.pi / 5,
    )
    assert gauss.size == size**2


def test_pulse_with_tail():
    """
    Test making a pulse with exponential tail
    """
    num_samples = 10

    times = np.linspace(1, 256, num_samples)
    pulse = create.pulse_with_tail(times, 10)

    assert pulse.size == num_samples


def test_uniform_locations():
    """
    Test the uniform location generator.
    It should give integer locations withim a range.
    """
    start = -1
    stop = 10
    num_locations = 10
    uniform = create.uniform_locations(start, stop, num_locations)

    assert isinstance(uniform[0], np.integer)
    assert uniform.max() <= stop
    assert uniform.min() >= start
    assert uniform.size == num_locations


class TestGaussWithTailLocations:
    """
    Test gauss_with_tail_locations with both backends.
    """

    @staticmethod
    def test_rvs():
        """
        Test rvs backend
        """
        num_samples = 10
        locations = create.gauss_with_tail_locations(
            0.1, 1024, 20, num_samples, back_end="rvs"
        )
        assert isinstance(locations[0], np.integer)
        assert locations.size == num_samples

    @staticmethod
    def test_cdf():
        """
        Test rvs backend
        """
        num_samples = 10
        locations = create.gauss_with_tail_locations(
            0.1, 1024, 20, num_samples, back_end="cdf"
        )
        assert isinstance(locations[0], np.integer)
        assert locations.size == num_samples

    @staticmethod
    def test_not_implemented():
        """
        Test that not implemented is called
        """
        with pytest.raises(NotImplementedError):
            num_samples = 10
            create.gauss_with_tail_locations(0.1, 1024, 20, num_samples, back_end="joe")


class TestBuildPulse:
    """
    Test that the location arrays correctly build the pulse
    """

    @staticmethod
    def test_total_power():
        """
        Should through an arrow is the total power of not correct
        Three dimensional array does not make sense here, so it
        should through an error
        """
        with pytest.raises(RuntimeError):
            create.build_pulse(10, 10, np.asarray([[2, 2], [2, 2], [1, 1]]))

    @staticmethod
    def test_location():
        """
        Test if locations are in the correct place
        """
        dynamic_spectra = create.build_pulse(10, 10, ((2, 2, 2), (2, 0, 2)))
        assert dynamic_spectra[2, 2] == 2
        assert dynamic_spectra[2, 0] == 1
        assert dynamic_spectra[3, 3] == 0


def test_spectral_index():
    """
    Test spectral_index, if alpha=0, should be ones
    """
    nchans = 10
    chans = np.linspace(960, 1400, nchans)
    ref = chans[len(chans) // 2]
    spec_array = create.spectral_index(chans, ref, 0)
    np.testing.assert_allclose(spec_array, np.ones(nchans))


def test_scintillation():
    """
    Test the scintillation. If nscint=0, nothing should happen
    """
    nchans = 10
    chans = np.linspace(960, 1400, nchans)
    ref = chans[len(chans) // 2]
    scint_array = create.scintillation(chans, ref, 0)
    np.testing.assert_allclose(scint_array, np.ones(nchans))


def test_scatter_profile():
    """
    Scatter profile should be 1 at the top frequency.
    """
    nchans = 10
    chans = np.linspace(960, 1400, nchans)
    ref = chans[len(chans) // 2]
    scatter = create.scatter_profile(chans, ref, 1)

    assert scatter[0] == 1
    np.testing.assert_almost_equal(scatter[-1], 0, decimal=3)
    assert scatter.size == nchans


def test_boxcar_convolved():
    """
    Test the boxcar convolver
    """
    time_profile = np.zeros(10)
    # len 2 boxcar
    widths = np.arange(1, 4)
    time_profile[4:6] = 1
    convolved = create.boxcar_convolved(time_profile, widths)

    assert convolved.argmax() == 1
    assert convolved[0] < convolved.max()
    assert convolved[2] < convolved.max()


class TestOptimalBoxcarWidth:
    """
    Test the optimal boxcar width against square and
    Gaussian pulse.
    """

    @staticmethod
    def test_square():
        """
        Test if the best boxcar width is returned for square
        pulse
        """
        time_profile = np.zeros(70)
        time_profile[20:40] += 1
        widths = np.arange(2, 60)
        opt = create.optimal_boxcar_width(time_profile, widths)
        assert opt == 20

    @staticmethod
    def test_gauss():
        """
        Test if the best boxcarwidth is returned for a
        Gauss pulse
        """
        time_profile = np.linspace(0, 100, 100)
        time_profile = create.gaussian(time_profile, mu=50, sig=10)
        widths = np.arange(2, 60)
        opt = create.optimal_boxcar_width(time_profile, widths)
        assert 15 < opt < 30


class TestSimplePulse:
    """
    Test the simple pulse
    """

    def setup_class(self):
        """
        Create the pulse
        """
        nchans = 4096
        bandpass = np.ones(nchans)
        self.splice = slice(2000, 2048)
        bandpass[self.splice] = 0
        self.simple_pulse = create.SimpleGaussPulse(
            0.01,
            dm=10,
            tau=5,
            chan_freqs=np.linspace(1919, 960, nchans),
            sigma_freq=600,
            center_freq=1500,
            tsamp=0.000256,
            spectral_index_alpha=1,
            nscint=1,
            phi=1,
            bandpass=bandpass,
        )
        self.num_samples = 10000
        self.pulse = self.simple_pulse.sample_pulse(self.num_samples)

    def test_power(self):
        """
        Test the powser levels
        """
        assert self.pulse.sum() == self.num_samples

    def test_bandpass(self):
        """
        Zero weights at the slice, these should be zero
        """
        assert self.pulse[:, self.splice].sum() < 10

    def test_optimal_width(self):
        """
        Test optimal location
        """
        optimal_width = self.simple_pulse.optimal_boxcar_width
        assert optimal_width > 0
        assert optimal_width < self.pulse.shape[0]

    def test_center(self):
        """
        Test optimal location
        """
        center = self.simple_pulse.pulse_center
        assert center > 0
        assert center < self.pulse.shape[0]


class TestTwoDimensionalPulse:
    """
    Test the two dimensional pdf.
    """

    def setup_class(self):
        """
        Create the pdf and pulse object
        """
        size_pdf = 100
        self.num_samples = 10000
        pulse_pdf = np.zeros((size_pdf, size_pdf))
        self.sin = np.sin(np.linspace(0, np.pi, size_pdf))
        self.sin /= self.sin.sum()
        pulse_pdf += self.sin[:, None]
        self.twod_pulse = create.TwoDimensionalPulse(
            pulse_pdf=pulse_pdf,
            chan_freqs=np.linspace(1919, 960, size_pdf),
            tsamp=0.000256,
            dm=155,
        )
        self.pulse = self.twod_pulse.sample_pulse(nsamp=self.num_samples)

    def test_bandpass(self):
        """
        Pulse pdf should be what we put it in.
        """
        assert np.allclose(self.twod_pulse.pulse_pdf.mean(axis=1), self.sin)

    def test_optimal_width(self):
        """
        Test optimal location
        """
        optimal_width = self.twod_pulse.optimal_boxcar_width
        assert optimal_width > 0
        assert optimal_width < self.pulse.shape[0]

    def test_center(self):
        """
        Test optimal location
        """
        center = self.twod_pulse.pulse_center
        assert center > 0
        assert center < self.pulse.shape[0]


class TestGaussPulse:
    """
    Test the multicomponent pulse
    """

    def setup_class(self):
        """
        Create the pulse
        """
        nchans = 4096
        bandpass = np.ones(nchans)
        self.splice = slice(2000, 2048)
        bandpass[self.splice] = 0

        self.num_samples = 10000
        self.complex_pulse = create.GaussPulse(
            relative_intensities=(1, 0.8, 0.8, 0.8),
            sigma_times=(0.005, 0.001, 0.001, 0.006),
            sigma_freqs=(150, 120, 120, 90),
            pulse_thetas=(0, 0, 0, -np.pi / 60),
            center_freqs=(1500, 1400, 1350, 1200),
            dm=155,
            tau=25,
            offsets=(0, 0.01536, 0.02304, 0.03968),  # all from start of window
            chan_freqs=np.linspace(1919, 960, nchans),
            tsamp=0.000256,
            spectral_index_alpha=0,
            nscint=2,
            phi=0,
            bandpass=bandpass,
        )
        # pulse with 3e5 samples
        self.pulse = self.complex_pulse.sample_pulse(nsamp=self.num_samples)

    def test_power(self):
        """
        Test the powser levels
        """
        assert self.pulse.sum() == self.num_samples

    def test_bandpass(self):
        """
        Zero weights at the slice, these should be zero
        but might be small.
        """
        assert self.pulse[:, self.splice].sum() < 10

    def test_optimal_width(self):
        """
        Test optimal location
        """
        optimal_width = self.complex_pulse.optimal_boxcar_width
        assert optimal_width > 0
        assert optimal_width < self.pulse.shape[0]

    def test_center(self):
        """
        Test optimal location
        """
        center = self.complex_pulse.pulse_center
        assert center > 0
        assert center < self.pulse.shape[0]


def test_filter_weights():
    """
    Test the filter_weights by creating a dynamic
    spectra with a bandstop filter
    """
    nchans = 512
    nsamps = 16
    dynamic = 150 * np.ones(nchans)
    diff = 150
    start = 200
    dynamic[start : start + diff] = 0
    dynamic = dynamic + np.random.normal(size=nchans * nsamps).reshape(nsamps, nchans)
    weights = create.filter_weights(dynamic, smooth_sigma=5)

    np.testing.assert_allclose(weights[start + diff // 2], 0)
    np.testing.assert_allclose(weights[start // 2], 1)

    assert len(create.filter_weights(dynamic, smooth_sigma=0))


class TestDynamicCreator:
    """
    Test dynamic spectra creators
    """

    def setup_class(self):
        """
        Create dynamic spectra
        """
        self.nchans = 128
        self.nsamps = 2**8
        self.medians = 15 * np.ones(self.nchans)
        self.medians[10:15] += 10
        self.stds = 5 * np.ones(self.nchans)
        self.stds[60:70] += 14
        self.dynamic = create.dynamic_from_statistics(
            self.medians, self.stds, dtype=np.uint8, nsamps=self.nsamps
        )

    def test_dynamic_from_statistics(self):
        """
        Test dynamic_from_statistics returns resonoable results
        """
        assert (self.nsamps, self.nchans) == (self.dynamic.shape)
        np.testing.assert_array_almost_equal(
            self.medians, np.median(self.dynamic, axis=0), decimal=-1
        )
        np.testing.assert_array_almost_equal(
            self.stds, self.dynamic.std(axis=0), decimal=-1
        )

    def test_clone_spectra(self):
        """
        Test cloning the above dynamic spectra
        """
        dynamic_clone = create.clone_spectra(self.dynamic)
        np.testing.assert_array_almost_equal(
            self.medians, np.median(dynamic_clone, axis=0), decimal=-1
        )
        np.testing.assert_array_almost_equal(
            self.stds, dynamic_clone.std(axis=0), decimal=-1
        )

        clone_shape = create.clone_spectra(self.dynamic, median_filter_length=2).shape
        assert clone_shape == self.dynamic.shape
