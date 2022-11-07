#!/usr/bin/env python3
"""
Math functions.
"""
import logging
import operator
import warnings
from typing import Callable

import numpy as np
from scipy import optimize, signal


# pylint: disable=invalid-name
def std_min_func(sigma: float, mu: float, std: float) -> float:
    """
    The zeros of this function are the log normal sigma that will
    have standard deviation `std`

    Args:
        sigma - lognormal sigma

        mu - lognormal mu

        std - The desired standard deviation

    Returns:
        desired variance - Actual variance(sigma, mu)
    """
    return std**2 - (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)


def log_normal_from_stats(median: float, std: float, size: int) -> np.ndarray:
    """
    Make a lognormal distribution that has has a given median
    and standard deviation.

    Args:
        median - median of resulting distribution

        std - Standard deviation of returned samples

    Returns:
        `size` numbers from the sampled from lognormal distribution
    """
    mu = np.log(median)
    sigma_guess = np.sqrt(mu - 0.5 * np.log(std**2))
    if sigma_guess == 0:
        sigma_guess = 0.01
    logging.debug("sigma_guess=%.2f", sigma_guess)

    # if the fitting does not converge,something went wrong
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sigma = optimize.fsolve(std_min_func, sigma_guess, args=(mu, std))[0]
    logging.debug("mu=%.2f, sigma=%.2f", mu, sigma)
    normal_random = np.random.normal(size=size)
    return np.exp(mu + sigma * normal_random)


def quicksort(
    array: np.ndarray,
    left: int = None,
    right: int = None,
    sort_fraction: float = 1.0,
    sort_ascend: bool = True,
) -> None:
    """
    Quicksort in place.

    Args:
        array - Array to be sorted

        left - Left most element of sort

        right - Right most element of sort

        sort_fraction - The fraction of the array to
                        stop sorting.

        sort_ascend - Sort increasing

    Returns:
        None - Sort in place
    """
    if left is None:
        left = 0
    if right is None:
        right = len(array) - 1

    if left >= sort_fraction * right:
        return

    pivot = array[right]
    divider = left

    if sort_ascend:
        compare = operator.lt
    else:
        compare = operator.gt

    for j in range(left, right):
        if compare(array[j], pivot):
            array[divider], array[j] = array[j], array[divider]
            divider += 1

    array[divider], array[right] = array[right], array[divider]
    pivot_idx = divider

    quicksort(
        array, left, pivot_idx - 1, sort_fraction=sort_fraction, sort_ascend=sort_ascend
    )
    quicksort(
        array,
        pivot_idx + 1,
        right,
        sort_fraction=sort_fraction,
        sort_ascend=sort_ascend,
    )


def sort_subarrays(
    array: np.ndarray, num_subarrays: int, sort_fraction: float = 1
) -> np.ndarray:
    """
    Sort subband of array. Flips the sorts back and forth to get a sine wave
    type shape.

    Args:
        array - array to sort

        num_subarrays - Number of subarrays to sort

        sort_fraction - The fraction of the subarray to
                stop sorting.

    Returns:
        Subarray sorted (also is sorted in place)
    """
    splits = np.array_split(array, num_subarrays)
    for j, split in enumerate(splits):
        is_even = j % 2 == 0
        quicksort(split, sort_ascend=is_even, sort_fraction=sort_fraction)
    return array


def calculate_dm_widths(
    dm: float, channel_width: float, chan_freqs: np.ndarray
) -> np.ndarray:
    """
    Calculate the inter-channel DM smearing. This implements
    Burke-Spolaor & Bannister 2018 (https://arxiv.org/pdf/1407.0400.pdf)
    Eqn 2 in seconds.

    Args:
        dm - The Dispersion Measure in pc/cm^3.

        channel_width - The channel width in MHz.

        chan_freqs - The channel frequencies in MHz.

    Returns:
        inter-channel dm smearing in seconds
    """
    return 8.3 / 10e6 * dm * channel_width * (1000 / chan_freqs) ** 3


def calculate_dm_boxcar_widths(
    dm: float,
    sampling_time: float,
    chan_freqs: np.ndarray,
) -> np.ndarray:
    """
     This calculates the boxcar widths in samples that correspond to the
    inter-channel dispersion delays in seconds. If delay is less than
     than the sample time of one, return a boxcar width of one.

     Args:
         dm - Dispersion Measure in pc/cm^3.

         sampling_time - Sampling time of data.

         chan_freqs - The channel frequencies in MHz.

     Returns:
         Array with boxcar lengths in samples.
    """
    channel_width = np.abs(chan_freqs[1] - chan_freqs[0])
    dm_widths_sec = calculate_dm_widths(dm, channel_width, chan_freqs)
    dm_widths_samples = np.around(dm_widths_sec / sampling_time).astype(int)
    dm_widths_samples[1 > dm_widths_samples] = 1
    return dm_widths_samples


def generate_boxcar_array(
    boxcar_lengths: np.ndarray,
    normalization_func: Callable = np.sqrt,
    return_max: bool = False,
) -> np.ndarray:
    """
    Make a 2D array of boxcars for the given boxcar_lengths.

    Args:
        boxcar_lengths - Array of boxcar lengths.

        normalization_func - Function to normalize the boxcar, default
                             is to use sqrt.

        return_max - Return max boxcar length

    Returns:
        Array of boxcars, with the boxcars stacked horizontally.
        (Optional) max boxcar length
    """
    max_boxcar = boxcar_lengths.max()
    boxcars = np.zeros((max_boxcar, len(boxcar_lengths)))
    for j, boxcar_length in enumerate(boxcar_lengths):
        offset = (max_boxcar - boxcar_length) // 2
        boxcars[offset : offset + boxcar_length, j] = 1 / normalization_func(
            boxcar_length
        )
    if return_max:
        return boxcars, max_boxcar
    return boxcars


def convolve_multi_boxcar(profile: np.ndarray, boxcar_array: np.ndarray) -> np.ndarray:
    """
    Convolve an profile array with an array that contains multiple boxcars.

    Args:
        profile - 1D or 2D array that contains the profile. For 1D array,
                  will be convolved with all boxcars. If 2D, time should
                  be on the vertical axis and number of profiles and
                  boxcars should match.

        boxcar_array - Array containing boxcars, boxcars should be stacked along
                       axis=1, see `generate_boxcar_array`.

    Returns:
        Profile convolved with the boxcar_array.
    """
    max_boxcar, num_boxcar = boxcar_array.shape
    if profile.ndim == 1:
        num_profiles = len(profile)
        logging.debug(
            "1D profile given, broadcasting to (%i, %i).", num_profiles, num_boxcar
        )
        profile = np.broadcast_to(profile[:, None], (num_profiles, num_boxcar))
    elif profile.ndim > 2:
        raise NotImplementedError(f"Cannot convolve {profile.ndim} dimension array.")
    convolved_profile = signal.fftconvolve(
        boxcar_array,
        profile,
        "full",
        axes=0,
    )
    return convolved_profile[max_boxcar // 2 - 1 : -max_boxcar // 2]


def boxcar_convolved(time_profile: np.ndarray, boxcar_widths: np.ndarray) -> np.ndarray:
    """
    Calculate the pulse profile convolved with a boxcar width.

    Args:
        time_profile - Time profile of the pulse

        boxcar_widths - Array of boxcar widths

    Returns:
        Pulse profile convolved with a boxcar
    """
    boxcar_widths = np.array(boxcar_widths, ndmin=1)
    powers = np.zeros(boxcar_widths.shape, dtype=np.float64)
    for j, width in enumerate(boxcar_widths):
        if width > 1:
            window = signal.boxcar(width) / np.sqrt(width)
            convolved_profile = signal.fftconvolve(window, time_profile, "full")
            convolved_profile = convolved_profile[width // 2 - 1 : -width // 2]
        else:
            convolved_profile = time_profile
        powers[j] = convolved_profile.max()
    return powers
