#!/usr/bin/env python3
"""
Test will.calculate
"""
import numpy as np

from will import calculate

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
        Test quicksort assending
        """
        rands_copy = self.rands.copy()
        calculate.quicksort(rands_copy, sort_assend=True)
        np.testing.assert_allclose(rands_copy, np.sort(self.rands))

    def test_sort_desend(self):
        """
        Test quicksort desending
        """
        rands_copy = self.rands.copy()
        calculate.quicksort(rands_copy, sort_assend=False)
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
            0, self.chan_width, self.sampling_time, self.chan_freqs
        )
        assert np.all(widths == 1)

    def test_high_dm(self):
        """
        Test 10000 DM.
        """
        dm = 10000
        widths = calculate.calculate_dm_boxcar_widths(
            dm, self.chan_width, self.sampling_time, self.chan_freqs
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
        self.boxcarwidths = 2 ** np.arange(0, self.num_boxcars)
        self.sqrt_array = calculate.generate_boxcar_array(self.boxcarwidths)

    def test_size(self):
        """
        Test the size of the array.
        """
        num_rows, num_cols = self.sqrt_array.shape
        assert num_rows == self.boxcarwidths.max()
        assert num_cols == self.num_boxcars

    def test_normalization_sqrt(self):
        """
        Test the normalization of the boxcars
        for Gaussian noise (scale sqrt)
        """
        np.testing.assert_almost_equal(
            self.sqrt_array.sum(axis=0), np.sqrt(self.boxcarwidths)
        )

    def test_normalization_unity(self):
        """
        Test the normalization of the boxcars
        for power preserving boxcar.
        """
        unity_array = calculate.generate_boxcar_array(self.boxcarwidths, lambda x: x)
        np.testing.assert_almost_equal(
            unity_array.sum(axis=0), np.ones(self.num_boxcars)
        )


class TestConvolveMutliBoxcar:
    """
    Test the multi boxcar convolver
    """

    def setup_class(self):
        """
        Boxcar widths
        """
        self.num_boxcars = 3
        self.boxcarwidths = 2 ** np.arange(0, self.num_boxcars)
        self.max_boxcar = self.boxcarwidths.max()
        self.sqrt_array = calculate.generate_boxcar_array(self.boxcarwidths)
        self.profile = np.zeros(2 * self.max_boxcar)
        self.profile[self.max_boxcar] = 1

    def test_single_profile(self):
        """
        Test a single profile, sqrt normalization.
        """

        convolved = calculate.convolve_mutli_boxcar(self.profile, self.sqrt_array)
        num_rows, num_cols = convolved.shape
        assert self.num_boxcars == num_cols
        assert len(self.profile) == num_rows

        above_zero = convolved > 0.1
        np.testing.assert_equal(above_zero.sum(axis=0), self.boxcarwidths)
        np.testing.assert_almost_equal(
            convolved.sum(axis=0), np.sqrt(self.boxcarwidths)
        )

    def test_single_profile_unity(self):
        """
        Test a single profile, unity normalization.
        """
        unity_array = calculate.generate_boxcar_array(self.boxcarwidths, lambda x: x)
        convolved = calculate.convolve_mutli_boxcar(self.profile, unity_array)
        total_power = convolved.sum(axis=0)
        np.testing.assert_allclose(total_power, np.ones_like(total_power))


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
