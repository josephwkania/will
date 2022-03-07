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

    # 15% seems generous enough
    assert np.std(distro) / std - 1 < 0.15
    assert np.median(distro) / median - 1 < 0.15
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
    time_series = np.zeros(10)
    time_series[4:6] = 1
    convolved = create.boxcar_convolved(time_series, [1, 2])

    np.testing.assert_allclose(convolved[0], 1)
    np.testing.assert_allclose(convolved[1], np.sqrt(2))
