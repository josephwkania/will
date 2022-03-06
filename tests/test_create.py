#!/usr/bin/env python3
"""
Test Pulse creation routines.
"""

import numpy as np
import pytest

from will import create


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
