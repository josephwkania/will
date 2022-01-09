#!/usr/bin/env python3
"""
Pulse Creation routines
"""

import logging

import numpy as np
from jess.dispersion import dedisperse, delay_lost
from scipy import integrate, interpolate, signal, stats


def gaussian(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """
    A Gaussian

    Args:
        x - Domain to calculate the Gaussian

        mu - Start location

        sig - Pulse width

    Returns:
        Gaussian evaluated over x
    """
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def pulse_with_tail(times: np.ndarray, tau: float = 50) -> np.ndarray:
    """
    Create a Gaussian Pulse with a scattering tail

    Args:
        Times - Time array

        tau - With paramater

    Returns:
        pulse profile

    Notes:
    Based on
    Ian P. Williamson, Pulse Broadening due to Multiple Scattering
    in the Interstellar Medium
    https://academic.oup.com/mnras/article/157/1/55/2604596

    I tried moving the center to match the other Gaussian,
    but this slows down the rvs sampler by a factor of ~4
    """
    # center: int = 0
    # times -= center
    logging.debug("Creating pulse with exponential tail, tau: %f", tau)
    # return np.where(
    #     times < 0,
    #     0,
    #     np.sqrt(np.pi * tau / (4 * times ** 3))
    #     * np.exp(-np.pi ** 2 * tau / (16 * times)),
    # )
    return np.sqrt(np.pi * tau / (4 * times ** 3)) * np.exp(
        -np.pi ** 2 * tau / (16 * times)
    )


def uniform_locations(
    start: int,
    stop: int,
    num_locations: int,
) -> np.ndarray:
    """
    Locations based on uniform sampling

    Args:
        start - Start index

        stop - Stop index

        num_locations - The number of locations to generate

    returns:
        location indices for one axis
    """
    logging.debug("Creating %i uniform pulse locations", num_locations)

    # The 0.4 makes the rounds on the final sections
    locations = np.random.uniform(start - 0.4, stop - 0.4, size=num_locations)
    np.around(locations, out=locations)
    np.clip(locations, start, stop - 1, out=locations)
    return locations.astype(int)


def _normalize_pulse_with_tail(times: np.ndarray, tau: float) -> float:
    """
    Find the normalization constant for pulse with scatter tail

    Args:
        times - Time sample locations

        tau - pulse width

    returns:
        normalization constant

    Note:
        based on Harry45
    """
    # center: int
    norm_constant = integrate.simps(pulse_with_tail(times, tau=tau), times)
    logging.debug("Normalization constant is %f", norm_constant)
    return norm_constant


class pulse_with_tail_dist(stats.rv_continuous):
    """
    Args:
        rv_continuous class

    Thanks to
    https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    """

    def _pdf(self, times: np.ndarray, tau: float, norm_const: float):
        """
        The pdf is the pulse_with_tail function

        Args:
            times - Time index

            tau - Pulse width

            norm_constant - Normalization Constant

        Returns:
            Values sampled from pulse with tail distrution
        """
        # center: int
        return (1.0 / norm_const) * pulse_with_tail(times, tau=tau)


def gauss_with_tail_cdf(times: np.ndarray, tau: float) -> np.ndarray:
    """
    Calculate the time locations for a Gaussian pulse with exponenital tail

    Args:
        times - Array with times to consider

        tau - Pulse width


    Returns:
        Values sampled from pulse with tail distrution

    Notes:
        based on
        https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    """
    #  center:int
    logging.debug("Sampling Gauss with tail, tau: %f", tau)
    pulse_cdf = np.cumsum(pulse_with_tail(times, tau=tau))
    pulse_cdf /= np.max(pulse_cdf)
    cdf_interp = interpolate.interp1d(pulse_cdf, times)

    # need some epsilon b/c 0 in not interpolated
    uniform_rand = np.random.uniform(0.001, 1, len(times))
    return cdf_interp(uniform_rand)


def arbitrary_array_cdf(
    array: np.ndarray, locations: float, num_samples: int
) -> np.ndarray:
    """
    Calculate the time locations from a given array

    Args:
        times - Array with times to consider

        tau - Pulse width


    Returns:
        Values sampled from pulse with tail distrution

    Notes:
        based on
        https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    """
    logging.debug("Sampling given array")
    pulse_cdf = np.cumsum(array)
    pulse_cdf /= np.max(pulse_cdf)
    cdf_interp = interpolate.interp1d(pulse_cdf, locations)

    # need some epsilon b/c 0 in not interpolated
    uniform_rand = np.random.uniform(0.001, 1, num_samples)
    return cdf_interp(uniform_rand)


def gauss_with_tail_locations(
    start: int,
    stop: int,
    tau: int,
    num_locations: int,
    back_end: str = "cdf",
) -> np.ndarray:
    """
    Locations based on Gaussian location

    Args:
        start - Start index

        stop - Stop index

        sigma - Gaussian sigma

        num_locations - The number of locations to generate

        back_end - How the samples are calculated [rvs] uses the rv_continous class,
                   cdf creates a CDF an linearly interpolates it. The latter is
                   much faster.

    Returns:
        Location indices for one axis

    """
    # center: int = None,
    # if center is None:
    #     center = (stop - start) // 2
    #     logging.debug("Center not given, will set center to %i")

    time_indices = np.linspace(start, stop, num_locations)
    back_end = back_end.casefold()
    if back_end == "cdf":
        locations = gauss_with_tail_cdf(time_indices, tau=tau)
    elif back_end == "rvs":
        pulse_distribution = pulse_with_tail_dist(
            name="pulse_distribution", a=start, b=stop
        )
        norm_const = _normalize_pulse_with_tail(time_indices, tau=tau)
        locations = pulse_distribution.rvs(
            tau=tau, norm_const=norm_const, size=num_locations
        )
    else:
        raise NotImplementedError(f"{back_end} is not avaliable")

    np.around(locations, out=locations)
    np.clip(locations, start, stop - 1, out=locations)
    return locations.astype(int)


def build_pulse(
    num_times: int, time_locs: np.ndarray, num_chans: int, chan_locs: np.ndarray
) -> np.ndarray:
    """
    Build the pulse from the time and freq locations.

    Args:
        time_locs - Time locations

        chan_locs - Channel locations

    Returns:
        2D float array with the pulse, time on the ventricle
        axis
    """
    ntimeloc = len(time_locs)
    nchanloc = len(chan_locs)
    if ntimeloc != nchanloc:
        raise RuntimeError(f"Number time locs: {ntimeloc} not equal to chan {nchanloc}")

    array = np.zeros((num_times, num_chans), dtype=np.uint32)
    # https://stackoverflow.com/a/45711530
    np.add.at(array, (time_locs, chan_locs), 1)

    total_power = array.sum()
    if total_power != len(time_locs):
        raise RuntimeError("Total power of the pulse is incorrect!")

    logging.debug("Created pulse with total counts %f", total_power)

    return array


def spectral_index(
    chan_freqs: np.ndarray, freq_ref: float, spectral_index: float
) -> np.ndarray:
    """
    Add spectral index to pulse profile

    Args:
        frequencies - Frequencies of the bandpass

        freq_ref - Reference frequency

        spectral index - Spectral index

    Returns:
        pulse profile modulated by spectral index
    """
    return (chan_freqs / freq_ref) ** spectral_index


def scintillation(
    chan_freqs: np.ndarray, freq_ref: float, nscint: int = 3, phi: float = 0
) -> np.ndarray:
    """
    Add Scintillation that is abs(cos(band))
    """
    # nscint = np.random.randint(0, 5)
    logging.debug("Scintillating with nscint %i and phi %f", nscint, phi)

    envelope = np.abs(np.cos(2 * np.pi * nscint * (chan_freqs / freq_ref) ** 2 + phi))
    return envelope


def scatter_profile(freqs: np.ndarray, ref_freq: float, tau: float = 1.0) -> np.ndarray:
    """
    Create exponential scattering profile.

    Args:
        freq - Frequencies array

        ref_freq - Reference Frequency

        tau - Scattering parameter

    Return:
        Exponential scattering profile
    """
    num_times = len(freqs)
    tau_nu = tau * (freqs / ref_freq) ** -4.0
    times = np.linspace(0.0, num_times // 2, num_times)
    prof = (1 / tau_nu) * np.exp(-times / tau_nu)
    return prof / prof.max()


def apply_scatter_profile(
    time_profile: np.ndarray, chan_freqs: np.ndarray, ref_freq: float, tau: float = 1.0
) -> np.ndarray:
    """
    Create exponential scattering profile.

    Args:
        freq - Frequencies array

        ref_freq - Reference Frequency

        tau - Scattering parameter

    Return:
        Exponential scattering profile
    """
    scatter = scatter_profile(chan_freqs, ref_freq, tau)
    scattered = signal.fftconvolve(time_profile, scatter, "full")[: len(time_profile)]
    return scattered / scattered.max()


def create_gauss_pulse(
    nsamp: int,
    sigma_time: float,
    dm: float,
    tau: float,
    sigma_freq: float,
    center_freq: float,
    chan_freqs: np.ndarray,
    tsamp: float,
    spectral_index: float,
    nscint: int,
    phi: float,
    bandpass: np.ndarray = None,
) -> np.ndarray:
    """
    Create a pulse from Gaussians in time and frequency

    """

    logging.debug("Creating time profile.")

    sigma_time_samples = np.around(sigma_time / tsamp)
    pulse_width = int(7 * sigma_time_samples + np.around(tau))
    time_indices = np.arange(0, pulse_width)
    pulse_time_profile = gaussian(
        time_indices, mu=pulse_width // 2, sig=sigma_time_samples
    )
    if tau > 0:
        pulse_time_profile = apply_scatter_profile(
            pulse_time_profile,
            chan_freqs=chan_freqs,
            ref_freq=chan_freqs[len(chan_freqs) // 2],
            tau=tau,
        )
    channel_bw = np.abs(chan_freqs[0] - chan_freqs[1])
    sigma_freq_samples = np.around(sigma_freq / channel_bw)
    freq_center_index = np.abs(chan_freqs - center_freq).argmin()
    nchans = len(chan_freqs)
    chan_indices = np.arange(0, nchans)
    pulse_freq_profile = gaussian(
        chan_indices,
        mu=freq_center_index,
        sig=sigma_freq_samples,
    )
    if spectral_index != 0:
        pulse_freq_profile *= spectral_index(
            chan_freqs=chan_freqs,
            freq_ref=center_freq,
            spectral_index=spectral_index,
        )
    if nscint != 0:
        pulse_freq_profile *= scintillation(
            chan_freqs=chan_freqs, freq_ref=center_freq, nscint=nscint, phi=phi
        )
    if bandpass is not None:
        pulse_freq_profile *= bandpass

    logging.debug("Calculating %i locations.", nsamp)
    time_locations = arbitrary_array_cdf(
        pulse_time_profile, locations=time_indices, num_samples=nsamp
    )
    np.round(time_locations, out=time_locations)
    time_locations = time_locations.astype(int)

    freq_locations = arbitrary_array_cdf(
        pulse_freq_profile, locations=chan_indices, num_samples=nsamp
    )
    np.around(freq_locations, out=freq_locations)
    freq_locations = freq_locations.astype(int)

    pulse_array = build_pulse(pulse_width, time_locations, nchans, freq_locations)

    delay = delay_lost(dm=dm, chan_freqs=chan_freqs, tsamp=tsamp)
    pulse_array_pad = np.zeros((pulse_width + delay, nchans), dtype=np.uint32)
    print(pulse_array.shape, pulse_array_pad.shape)
    pulse_array_pad[:pulse_width] = pulse_array
    pulse_dispersed = dedisperse(
        pulse_array_pad, dm=-dm, tsamp=tsamp, chan_freqs=chan_freqs
    )

    return pulse_dispersed
