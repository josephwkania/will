#!/usr/bin/env python3
"""
Math functions.
"""
import logging
import operator
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from jess.dispersion import dedisperse, delay_lost
from rich.progress import track
from scipy import interpolate, optimize, signal, stats
from your import Your


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
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
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


def median_line_detrend(array: np.ndarray, num_sample: int) -> np.ndarray:
    """
    Detrend an array by finding medians of sections and fitting a line
    to these medians, then subtracting this line.

    Args:
        array - Array to detrend.

        num_samples - Number of samples per median block.

    Return:
        array with trend subtracted.

    Inspired by 'Advanced Architectures for Astrophysical Supercomputing',
    section 4.2.3
    """
    len_array = len(array)
    num_medians = np.ceil(len_array / num_sample).astype(int)
    desired_length = num_medians * num_sample
    to_length = desired_length - len_array

    if to_length > 0:
        half_length = to_length / 2
        correct_array = np.pad(
            array,
            (np.floor(half_length).astype(int), np.ceil(half_length).astype(int)),
            mode="reflect",
        )
    else:
        correct_array = array

    reshaped_array = correct_array.reshape(num_medians, -1)
    medians = np.median(reshaped_array, axis=1)

    offset = np.around(len_array / (2 * num_medians))
    middles = np.linspace(offset, len_array - offset, num_medians, dtype=int)
    interp_func = interpolate.interp1d(
        middles, medians, fill_value="extrapolate", kind="linear"
    )

    full_samples = np.linspace(0, len_array, len_array, dtype=int)
    return array - interp_func(full_samples)


def calculate_noises_multi(
    time_series: np.ndarray,
    boxcar_array: np.ndarray,
    max_boxcar: int,
    smoothing_factor: int = 4,
    scale_func: Callable = partial(stats.median_abs_deviation, scale="normal"),
) -> np.ndarray:
    """
    Calculate the noise level of a time series over a set
    of boxcars. Detrend the time series using median_line_detrend.
    Convolve the detrened time series with the boxcar array,
    Then use `scale_func` to calculate the noise level.

    Args:
        time_series - The dedispersed time series.

        box_car_length - Length of the boxcars.

        sigma - Return pulses with significance above
                this

        smoothing_factor - Detrend blocks are
                           smoothing_factor*box_car_length long.

        scale_func - The function used to calculate the measure of scale.
                     must accept `axis=0` argument.

    Returns:
        Measures of scale for each boxcar in `boxcar array`.
    """

    flattened_times_series = median_line_detrend(
        time_series, max_boxcar * smoothing_factor
    )
    convolved_profiles = convolve_multi_boxcar(
        flattened_times_series, boxcar_array=boxcar_array
    )
    stds = scale_func(convolved_profiles, axis=0)
    return stds


@dataclass
class NoiseInfoResult:
    """
    The noise levels for a file.

    noise_levels - The noise level at each location (row) and
                   each boxcar, column.

    boxcar_lengths - Length of the boxcars.

    num_chans - Number of channels.
    """

    noise_levels: np.ndarray
    boxcars_lengths: np.ndarray
    num_chans: int

    @property
    def mean_noise_levels(self):
        """
        The mean noise levels across the file.
        """
        return self.noise_levels.mean(axis=0)

    @property
    def median_noise_levels(self):
        """
        The mean noise levels across the file.
        """
        return np.median(self.noise_levels, axis=0)

    def plot_noise(self):
        """
        Plot the mean noise levels for a file.
        """
        plt.plot(self.boxcars_lengths, self.mean_noise_levels, label="Mean")
        plt.plot(self.boxcars_lengths, self.median_noise_levels, label="Median")
        plt.xlabel("Boxcar Lengths [Samples]", size=25)
        plt.ylabel("Noise Level", size=25)
        plt.tick_params(axis="both", which="both", labelsize=15)
        plt.tick_params(width=5)
        plt.legend(prop={"size": 15})
        plt.title("Noise Levels", size=30)
        plt.show()

    @property
    def mean_onesigma_boxcar_volume(self):
        """
        The mean volume of a one sigma boxcar.
        """
        return self.num_chans * self.boxcars_lengths * self.mean_noise_levels


def noise_info(
    file_path: str,
    dm: float,
    boxcar_lengths: np.ndarray,
    num_locations: int = 16,
    num_samples: int = 2**14,
    chan_mask: Union[np.ndarray, None] = None,
):
    """
    Get noise levels for a file and set of boxcar lengths. This is done by
    selecting `num_locations` evenly spaced throughout the file. `num_samples` are
    extracted from each of these locations. If a Dispersion Measure is given, The
    the data is dedispersed to that DM. If a channel mask is provided, these channels
    are not included in the time series. The resulting time series are detrended by
    calculating medians of blocks 4 times the largest boxcar size, and
    interpolating a linear fit between these points. The detrened time series are
    convolved with boxcars of the given length. The noise is then calculated my MAD.

    Args:
        file_path - String that is the path to a file.

        dm - Dispersion Measure in pc/cm^3.

        boxcar_lengths - Numpy array of boxcar lengths to consider.

        num_locations - Number of locations in the file to consider.

        num_samples - Number of sample to consider, must be 4 times the
                      largest boxcar.

        chan_mask - Mask these channels if given. True=value to mask

    Returns:
        NoiseInfoResult
    """
    yr_obj = Your(file_path)
    samples_lost = delay_lost(
        dm=dm,
        chan_freqs=yr_obj.chan_freqs,
        tsamp=yr_obj.your_header.tsamp,
    )
    if num_samples * num_locations > yr_obj.your_header.nspectra:
        raise ValueError("Too many or too long Boxcars.")
    start_locations = np.linspace(
        0, yr_obj.your_header.nspectra - num_samples - samples_lost, num_locations
    )
    nsamp = num_samples + samples_lost
    boxcar_array, max_boxcar = generate_boxcar_array(
        boxcar_lengths, return_max=True, normalization_func=lambda x: x
    )
    if 4 * max_boxcar > num_samples:
        raise ValueError(f"{4*max_boxcar=} is larger than {num_samples=}")

    noises = np.zeros((num_locations, len(boxcar_lengths)))
    for j, location in enumerate(
        track(
            start_locations,
            description="Processing Noises",
            transient=True,
            refresh_per_second=1,
        )
    ):
        dynamic_spectra = yr_obj.get_data(location, nsamp)
        if dm > 0:
            dynamic_spectra_dispered = dedisperse(
                dynamic_spectra,
                dm=dm,
                tsamp=yr_obj.your_header.tsamp,
                chan_freqs=yr_obj.chan_freqs,
            )
            # cut the rolled part
            dynamic_spectra = dynamic_spectra_dispered[:-samples_lost]
        if chan_mask is not None:
            dynamic_spectra = dynamic_spectra[:, ~chan_mask]

        time_series = dynamic_spectra.mean(axis=1)
        noises[j] = calculate_noises_multi(
            time_series=time_series, boxcar_array=boxcar_array, max_boxcar=max_boxcar
        )
    return NoiseInfoResult(noises, boxcar_lengths, yr_obj.your_header.nchans)
