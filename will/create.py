#!/usr/bin/env python3
"""
Pulse creation routines.
"""

import functools
import logging
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Union

import numpy as np
from jess.calculators import median_abs_deviation_med, to_dtype
from jess.dispersion import dedisperse, delay_lost
from jess.fitters import median_fitter
from scipy import integrate, interpolate, ndimage, signal, stats

from . import calculate


# pylint: disable=invalid-name
def gaussian(domain: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """
    A Gaussian

    Args:
        domain - Domain to calculate the Gaussian

        mu - Center location

        sig - Pulse width

    Returns:
        Gaussian evaluated over x
    """
    return np.exp(-np.power(domain - mu, 2.0) / (2 * np.power(sig, 2.0)))


def skewed_gauss(
    x: np.ndarray,
    y: np.ndarray,
    x_mu: float,
    y_mu: float,
    x_sig: float,
    y_sig: float,
    theta: float,
) -> np.ndarray:
    """
    Two dimensional Gaussian with an angle theta.

    Args:
        x - Horizontal Domain from np.meshgrid

        y - Vertical Domain from np.meshgrid

        x_mu - Horizontal Location

        y_mu - Vertical Distance

        x_sig - Horizontal sigma

        y_sig - Vertical Sigma

        theta - Rotation angle increasing counterclockwise [radians]

    Returns:
        2D gaussian with amplitude one.

    Notes:
        Based on
        https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Gaussian2D.html
    """

    cos_t_2 = np.cos(theta) ** 2
    sin_t_2 = np.sin(theta) ** 2
    sin_2t = np.sin(2.0 * theta)
    xstd_2 = x_sig**2
    ystd_2 = y_sig**2
    xdiff = x - x_mu
    ydiff = y - y_mu

    a = 0.5 * ((cos_t_2 / xstd_2) + (sin_t_2 / ystd_2))
    b = 0.5 * ((sin_2t / xstd_2) - (sin_2t / ystd_2))
    c = 0.5 * ((sin_t_2 / xstd_2) + (cos_t_2 / ystd_2))

    return np.exp(-((a * xdiff**2) + (b * xdiff * ydiff) + (c * ydiff**2)))


def pulse_with_tail(times: np.ndarray, tau: float = 50) -> np.ndarray:
    """
    Create a Gaussian Pulse with a scattering tail

    Args:
        Times - Time array

        tau - With parameter

    Returns:
        pulse profile

    Notes:
    Based on
    Ian P. Williamson, Pulse Broadening due to Multiple Scattering
    in the Interstellar Medium
    https://academic.oup.com/mnras/article/157/1/55/2604596

    I tried moving the center to match the other Gaussian,
    but this slows down the rvs sampler by a factor of ~4

    Example:
    times = np.linspace(1, 256, 256)
    pulse = create.pulse_with_tail(times, 50)
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
    return np.sqrt(np.pi * tau / (4 * times**3)) * np.exp(
        -np.pi**2 * tau / (16 * times)
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
            Values sampled from pulse with tail distribution
        """
        # center: int
        return (1.0 / norm_const) * pulse_with_tail(times, tau=tau)


def gauss_with_tail_cdf(times: np.ndarray, tau: float) -> np.ndarray:
    """
    Calculate the time locations for a Gaussian pulse with exponential tail

    Args:
        times - Array with times to consider

        tau - Pulse width

    Returns:
        Values sampled from pulse with tail distribution

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
        Values sampled from pulse with tail distribution

    Notes:
        based on
        https://harry45.github.io/blog/2016/10/Sampling-From-Any-Distribution
    """
    logging.debug("Sampling given array")
    # if array has negatives, pulse_cdf will not be monotonic
    # and interpolation is difficult
    assert array.min() >= 0, "Probability array must be non-negative!"
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
    Locations based on Gaussian with an exponential tail.
    You probably want to use GaussPulse or Simple Gauss pulse.
    This provides a comparison between the cdf and rvs samplers.

    Args:
        start - Start index

        stop - Stop index

        sigma - Gaussian sigma

        num_locations - The number of locations to generate

        back_end - How the samples are calculated [rvs] uses the rv_continuous class,
                   cdf creates a CDF an linearly interpolates it. The latter is
                   much faster.

    Returns:
        Location indices for one axis

    Example:
        gauss_with_tail_locations(0.1, 1024, 20, int(5e2), back_end="rvs")
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
    num_times: int,
    num_chans: int,
    locations: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Build the pulse from locations tuple.

    Args:
        num_times - Length of the time samples axis

        num_chans - Length of channel axis

        locations - Locations of the points to increase the energy
                    given as two arrays

    Returns:
        2D float array with the pulse, time on the ventricle
        axis

    Example:
        pulse = build_pulse(10, 10, [[2, 2, 2], [2, 0, 2]]) will
        make a point value 2 at pulse[2,2] and one at pulse[2,0]
    """

    array = np.zeros((num_times, num_chans), dtype=np.uint32)
    # https://stackoverflow.com/a/45711530
    np.add.at(array, locations, 1)

    total_power = array.sum()
    if total_power != len(locations[0]):
        raise RuntimeError("Total power of the pulse is incorrect!")

    logging.debug("Created pulse with total counts %f", total_power)

    return array


def spectral_index(
    chan_freqs: np.ndarray, freq_ref: float, spectral_index_alpha: float
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
    return (chan_freqs / freq_ref) ** spectral_index_alpha


def scintillation(
    chan_freqs: np.ndarray, freq_ref: float, nscint: int = 3, phi: float = 0
) -> np.ndarray:
    """
    Adds Scintillation that is abs(cos(band))

    Args:
        chan_freqs - Array of channel Frequencies

        freq_freq - Reference Frequency

        nscint - number of scintills

        phi - phase of of scintillation

    Returns:
        scintillation intensities

    Notes:
        Similar to https://arxiv.org/abs/2003.14272
        and
        https://github.com/liamconnor/injectfrb/blob/a4dd5f22438ba7bdfaa2eb792eb54736cde53fed/injectfrb/simulate_frb.py#L77
        added abs as it seems more relativistic to me
    """
    logging.debug("Scintillating with nscint %i and phi %f", nscint, phi)

    envelope = np.abs(np.cos(2 * np.pi * nscint * (chan_freqs / freq_ref) ** 2 + phi))
    return envelope


def scatter_profile(
    chan_freqs: np.ndarray, ref_freq: float, tau: float = 1.0
) -> np.ndarray:
    """
    Create exponential scattering profile.

    Args:
        freq - Frequencies array

        ref_freq - Reference Frequency

        tau - Scattering parameter

    Return:
        Exponential scattering profile

    Notes:
        Bases on
        https://github.com/liamconnor/injectfrb/blob/a4dd5f22438ba7bdfaa2eb792eb54736cde53fed/injectfrb/simulate_frb.py#L111
    """
    num_times = len(chan_freqs)
    tau_nu = tau * (chan_freqs / ref_freq) ** -4.0
    times = np.linspace(0.0, num_times // 2, num_times)
    prof = (1 / tau_nu) * np.exp(-times / tau_nu)
    return prof / prof.max()


def apply_scatter_profile(
    time_profile: np.ndarray,
    chan_freqs: np.ndarray,
    ref_freq: float,
    tau: float = 1.0,
    axis: Union[None, int] = None,
) -> np.ndarray:
    """
    Create exponential scattering profile.

    Args:
        freq - Frequencies array

        ref_freq - Reference Frequency

        tau - Scattering parameter

        axis - Axis to perform convolution

    Return:
        Exponential scattering profile
    """
    if axis is None:
        scatter = scatter_profile(chan_freqs, ref_freq, tau)
    elif axis == 0:
        scatter = scatter_profile(chan_freqs, ref_freq, tau)[:, None]
    else:
        raise NotImplementedError(f"{axis=} is not implemented!")

    scattered = signal.fftconvolve(time_profile, scatter, "full", axes=axis)[
        : len(time_profile)
    ]
    return scattered / scattered.max()


def optimal_boxcar_width(
    time_profile: np.ndarray, boxcar_widths: np.ndarray
) -> np.int64:
    """
    Find the best boxcar width for a given time profile and array
    of boxcar widths.

    Args:
        time_profile - The time profile of the pulse

        boxcar_widths - Array of boxcar widths

    Returns:
        Length of the optimal boxcar
    """
    powers = calculate.boxcar_convolved(
        time_profile=time_profile, boxcar_widths=boxcar_widths
    )
    # Get the last index of the larget value, this will be the one most robust to Gauss
    # noise
    max_idx = len(powers) - np.argmax(powers[::-1]) - 1
    return boxcar_widths[max_idx]


@dataclass
class SimpleGaussPulse:
    """
    Create a pulse from Gaussians in time and frequency.
    The time and frequency profiles are created with the
    object. To sample use .sample_pulse(nsamps)
    to sample the pulse.

    Args:

        sigma_time - time sigma in seconds

        dm - Dispersion measure

        tau - scatter

        sigma_freq - Frequency Sigma in MHz

        center_freq - Center Frequency in MHz

        chan_freq - Array of channel frequencies in MHz

        tsamp - sampling time of dynamic spectra in second

        spectra_index_alpha - spectral index around center_freq

        nscint - number of scintills

        phi - phase of of scintillation

        bandpass - scale frequency structure with bandpass if
                   not None
    """

    sigma_time: float
    sigma_freq: float
    center_freq: float
    dm: float
    tau: float
    chan_freqs: np.ndarray
    tsamp: float
    spectral_index_alpha: float
    nscint: int
    phi: float
    bandpass: Union[np.ndarray, None] = None

    def __post_init__(self):
        """
        Create the pulse time and frequency profiles when
        the instance is created
        """
        self.create_pulse()

    @property
    def pulse_center(self) -> np.int64:
        """
        The location of the pulse maximum in time samples
        """
        return self.pulse_time_profile.argmax()

    @functools.cached_property
    def optimal_boxcar_width(self) -> np.int64:
        """
        Find the optimal boxcar width
        """
        boxcar_widths = np.arange(1, self.pulse_width)
        return optimal_boxcar_width(
            time_profile=self.pulse_time_profile, boxcar_widths=boxcar_widths
        )

    def create_pulse(self) -> None:
        """
        Create the pulse
        """
        logging.debug("Creating time profile.")

        sigma_time_samples = np.around(self.sigma_time / self.tsamp)
        self.gauss_width = int(8 * sigma_time_samples)
        self.pulse_width = int(self.gauss_width + np.around(8 * self.tau))
        self.time_indices = np.arange(0, self.pulse_width)
        self.pulse_time_profile = gaussian(
            self.time_indices, mu=self.gauss_width // 2, sig=sigma_time_samples
        )
        if self.tau > 0:
            self.pulse_time_profile = apply_scatter_profile(
                self.pulse_time_profile,
                chan_freqs=self.chan_freqs,
                ref_freq=self.chan_freqs[len(self.chan_freqs) // 2],
                tau=self.tau,
            )
        # offset_to_pulse_max = pulse_time_profile.argmax()

        channel_bw = np.abs(self.chan_freqs[0] - self.chan_freqs[1])
        sigma_freq_samples = np.around(self.sigma_freq / channel_bw)
        freq_center_index = np.around(
            (self.center_freq - self.chan_freqs.min()) / channel_bw
        ).astype(int)
        self.nchans = self.chan_freqs.size
        self.chan_indices = np.arange(0, self.nchans)
        self.pulse_freq_profile = gaussian(
            self.chan_indices,
            mu=freq_center_index,
            sig=sigma_freq_samples,
        )
        if self.spectral_index_alpha != 0:
            self.pulse_freq_profile *= spectral_index(
                chan_freqs=self.chan_freqs,
                freq_ref=self.center_freq,
                spectral_index_alpha=self.spectral_index_alpha,
            )
        if self.nscint != 0:
            self.pulse_freq_profile *= scintillation(
                chan_freqs=self.chan_freqs,
                freq_ref=self.center_freq,
                nscint=self.nscint,
                phi=self.phi,
            )
        if self.bandpass is not None:
            self.pulse_freq_profile *= self.bandpass

    def sample_pulse(self, nsamp: int, dtype: type = np.uint32) -> np.ndarray:
        """
        Sample the pulse with `nsamp` samples

        Args:
            nsamp - Number of samples in the pulse

            dtype - Data type of the pulse

        Returns:
            2D ndarray with disperesed pulse
        """

        logging.debug("Calculating %i locations.", nsamp)
        time_locations = arbitrary_array_cdf(
            self.pulse_time_profile, locations=self.time_indices, num_samples=nsamp
        )
        time_locations = to_dtype(time_locations, int)

        freq_locations = arbitrary_array_cdf(
            self.pulse_freq_profile, locations=self.chan_indices, num_samples=nsamp
        )
        freq_locations = to_dtype(freq_locations, dtype=int)

        pulse_array = build_pulse(
            self.pulse_width, self.nchans, (time_locations, freq_locations)
        )

        delay = delay_lost(dm=self.dm, chan_freqs=self.chan_freqs, tsamp=self.tsamp)
        pulse_array_pad = np.zeros((self.pulse_width + delay, self.nchans), dtype=dtype)
        pulse_array_pad[: self.pulse_width] = pulse_array
        pulse_dispersed = dedisperse(
            pulse_array_pad, dm=-self.dm, tsamp=self.tsamp, chan_freqs=self.chan_freqs
        )

        return pulse_dispersed


@dataclass
class TwoDimensionalPulse:
    """
    Create a pulse from a 2D pulse Probability Distribution Function (PDF).

    Args:
        pulse_pdf - The 2D array containing pulse the pulse profile at
                    0 DM.

        chan_freq - Array of channel frequencies in MHz

        tsamp - sampling time of dynamic spectra in second

        dm - Dispersion Measure

    """

    pulse_pdf: np.ndarray
    chan_freqs: np.ndarray
    tsamp: float
    dm: float

    def __post_init__(self):
        self.pulse_width = len(self.pulse_pdf)
        self.nchans = self.chan_freqs.size

    @property
    def pulse_center(self) -> int:
        """
        The location of the pulse maximum in time samples
        """
        return self.pulse_pdf.mean(axis=1).argmax()

    @functools.cached_property
    def optimal_boxcar_width(self) -> np.int64:
        """
        Find the optimal boxcar width
        """
        boxcar_widths = np.arange(1, self.pulse_width)
        return optimal_boxcar_width(
            time_profile=self.pulse_pdf.mean(axis=1), boxcar_widths=boxcar_widths
        )

    def sample_pulse(self, nsamp: int, dtype: type = np.uint32) -> np.ndarray:
        """
        Sample the pulse with `nsamp` samples

        Args:
            nsamp - Number of samples in the pulse

            dtype - Data type of the pulse

        Returns:
            2D ndarray with disperesed pulse
        """
        logging.debug("Calculating %i locations.", nsamp)

        pulse_pdf_flat = self.pulse_pdf.flatten()
        locations = arbitrary_array_cdf(
            pulse_pdf_flat,
            locations=np.arange(0, len(pulse_pdf_flat)),
            num_samples=nsamp,
        )
        locations = to_dtype(locations, dtype=int)
        locations = np.unravel_index(locations, self.pulse_pdf.shape)

        pulse_array = build_pulse(self.pulse_width, self.nchans, locations)

        delay = delay_lost(dm=self.dm, chan_freqs=self.chan_freqs, tsamp=self.tsamp)
        pulse_array_pad = np.zeros((self.pulse_width + delay, self.nchans), dtype=dtype)
        pulse_array_pad[: self.pulse_width] = pulse_array
        pulse_dispersed = dedisperse(
            pulse_array_pad, dm=-self.dm, tsamp=self.tsamp, chan_freqs=self.chan_freqs
        )

        return pulse_dispersed


@dataclass
class GaussPulse:
    """
    Create a pulse from a 2D Gaussian.
    This function can handle

    The PDF is created ith the object.
    To sample use sample_pulse(nsamps).

    Args:
        relative_intensities - The relative intensities of the
                               pule components.

        sigma_time - time sigma in seconds

        sigma_freq - Frequency Sigma in MHz

        chan_freq - Array of channel frequencies in MHz

        tsamp - sampling time of dynamic spectra in second

        pulse_theta - Angle of pulse components

        nsamp - Number of samples to add

        dm - Dispersion Measure

        tau - Scatter parameter

        tsamp - Sampling time of dynamic spectra in second

        spectra_index_alpha - Spectral index power around center_freq

        nscint - Number of frequency scintills

        phi - Phase of frequency scintillation

        bandpass - Scale frequency structure with bandpass if
                   not None

        dm_interchan_smear - Interchannel DM smearing simulated by
                             boxcar convolution
    """

    relative_intensities: Union[Sequence, float]
    sigma_times: Union[Sequence, float]
    sigma_freqs: Union[Sequence, float]
    center_freqs: Union[Sequence, float]
    pulse_thetas: Union[Sequence, float]
    offsets: Union[Sequence, float]
    dm: float
    tau: float
    chan_freqs: np.ndarray
    tsamp: float
    spectral_index_alpha: float
    nscint: int
    phi: float
    bandpass: Union[np.ndarray, None] = None
    dm_interchan_smear: bool = False

    def __post_init__(self):
        """
        Convert sequencies to ndarrays

        Create the pulse when the object is created
        """
        self.relative_intensities = np.array(self.relative_intensities, ndmin=1)
        self.sigma_times = np.array(self.sigma_times, ndmin=1)
        self.sigma_freqs = np.array(self.sigma_freqs, ndmin=1)
        self.center_freqs = np.array(self.center_freqs, ndmin=1)
        self.pulse_thetas = np.array(self.pulse_thetas, ndmin=1)
        self.offsets = np.array(self.offsets, ndmin=1)

        # set will be unique
        lengths_set = {
            self.relative_intensities.size,
            self.sigma_times.size,
            self.sigma_freqs.size,
            self.center_freqs.size,
            self.pulse_thetas.size,
            self.offsets.size,
        }

        if len(lengths_set) > 1:
            raise ValueError("Didn't provide info for all of the pulses")

        self.create_pulse()

    @property
    def pulse_center(self) -> int:
        """
        The location of the pulse maximum in time samples
        """
        return self.pulse_pdf.mean(axis=1).argmax()

    @functools.cached_property
    def optimal_boxcar_width(self) -> np.int64:
        """
        Find the optimal boxcar width
        """
        boxcar_widths = np.arange(1, self.pulse_width)
        return optimal_boxcar_width(
            time_profile=self.pulse_pdf.mean(axis=1), boxcar_widths=boxcar_widths
        )

    def create_pulse(self) -> None:
        """
        Create the pulse
        """
        logging.debug("Creating pulse profile.")

        channel_bw = np.abs(self.chan_freqs[0] - self.chan_freqs[1])
        sigmas_freq_samples = np.around(self.sigma_freqs / channel_bw)

        freq_center_indices = np.around(
            (self.center_freqs - self.chan_freqs.min()) / channel_bw
        ).astype(int)
        self.nchans = self.chan_freqs.size
        chan_indices = np.arange(self.nchans, 0, step=-1)

        if self.dm_interchan_smear:
            boxcar_widths = calculate.calculate_dm_boxcar_widths(
                self.dm, self.tsamp, self.chan_freqs
            )
            dm_smear_boxcars, max_dm_boxcar = calculate.generate_boxcar_array(
                boxcar_widths, return_max=True
            )
        else:
            max_dm_boxcar = 0

        sigmas_time_samples = np.around(self.sigma_times / self.tsamp)
        self.offsets = np.around(self.offsets / self.tsamp)
        gauss_widths = 8 * sigmas_time_samples
        self.pulse_width = int(
            gauss_widths[0]
            + gauss_widths[-1]
            + self.offsets.sum()
            + np.around(8 * self.tau)
            + 3 * (np.sin(self.pulse_thetas) * sigmas_freq_samples).argmax()
            + max_dm_boxcar,
        )
        time_indices = np.arange(0, self.pulse_width)
        chan_indices, time_indices = np.meshgrid(chan_indices, time_indices)

        self.pulse_pdf = np.zeros((self.pulse_width, self.nchans))
        for j in range(self.relative_intensities.size):
            logging.debug("Adding pulse #%i", j)
            self.pulse_pdf += self.relative_intensities[j] * skewed_gauss(
                x=chan_indices,
                y=time_indices,
                x_mu=freq_center_indices[j],
                y_mu=self.offsets[j] + gauss_widths[0] // 2,
                x_sig=sigmas_freq_samples[j],
                y_sig=sigmas_time_samples[j],
                theta=self.pulse_thetas[j],
            )

        if self.tau > 0:
            self.pulse_pdf = apply_scatter_profile(
                self.pulse_pdf,
                chan_freqs=self.chan_freqs,
                ref_freq=self.chan_freqs[len(self.chan_freqs) // 2],
                tau=self.tau,
                axis=0,
            )

        if self.nscint != 0:
            self.pulse_pdf *= scintillation(
                chan_freqs=self.chan_freqs,
                freq_ref=np.median(self.center_freqs),
                nscint=self.nscint,
                phi=self.phi,
            )

        if self.spectral_index_alpha != 0:
            self.pulse_pdf *= spectral_index(
                chan_freqs=self.chan_freqs,
                freq_ref=np.median(self.center_freqs),
                spectral_index_alpha=self.spectral_index_alpha,
            )

        if self.bandpass is not None:
            self.pulse_pdf *= self.bandpass

        if self.dm_interchan_smear and max_dm_boxcar > 1:
            # only do the convoulution if the boxcar is bigger than one,
            # otherwise convolvue_multi_boxcar cuts returns any empty
            # because it trimes off excess, which there is none.
            self.pulse_pdf = calculate.convolve_multi_boxcar(
                self.pulse_pdf, dm_smear_boxcars
            )

    def sample_pulse(self, nsamp: int, dtype: type = np.uint32) -> np.ndarray:
        """
        Sample the pulse with `nsamp` samples

        Args:
            nsamp - Number of samples in the pulse

            dtype - Data type of the pulse

        Returns:
            2D ndarray with disperesed pulse
        """
        logging.debug("Calculating %i locations.", nsamp)

        pulse_pdf_flat = self.pulse_pdf.flatten()
        locations = arbitrary_array_cdf(
            pulse_pdf_flat,
            locations=np.arange(0, len(pulse_pdf_flat)),
            num_samples=nsamp,
        )
        locations = to_dtype(locations, dtype=int)
        locations = np.unravel_index(locations, self.pulse_pdf.shape)

        pulse_array = build_pulse(self.pulse_width, self.nchans, locations)

        delay = delay_lost(dm=self.dm, chan_freqs=self.chan_freqs, tsamp=self.tsamp)
        pulse_array_pad = np.zeros((self.pulse_width + delay, self.nchans), dtype=dtype)
        pulse_array_pad[: self.pulse_width] = pulse_array
        pulse_dispersed = dedisperse(
            pulse_array_pad, dm=-self.dm, tsamp=self.tsamp, chan_freqs=self.chan_freqs
        )

        return pulse_dispersed


def filter_weights(
    dynamic_spectra: np.ndarray,
    metric: Callable = np.median,
    bandpass_smooth_length: int = 50,
    cut_sigma: float = 2 / 3,
    smooth_sigma: int = 30,
) -> np.ndarray:
    """
    Makes weights based on low values of some meteric.
    This is designed to ignore bandpass filters or tapers
    at the end of the bandpass.

    Args:
        dynamic_spectra - 2D dynamic spectra with time on the
                          vertical axis

        metric - The statistic to sample.

        bandpass_smooth_length - length of the median filter to
                                 smooth the bandpass

        sigma_cut - Cut values below (standard deviation)*(sigma cut)

        smooth_sigma - Gaussian filter smoothing sigma. If =0, return
                       the mask where True=good channels

    Returns:
        Bandpass weights for sections of spectra with low values.
        0 where the value is below the threshold and 1 elsewhere,
        with a Gaussian taper.
    """
    bandpass = metric(dynamic_spectra, axis=0)
    bandpass_std = stats.median_abs_deviation(bandpass, scale="normal")
    threshold = bandpass_std * cut_sigma
    if bandpass_smooth_length > 1:
        bandpass = median_fitter(bandpass, chans_per_fit=bandpass_smooth_length)
    mask = bandpass > threshold

    if smooth_sigma > 0:
        return ndimage.gaussian_filter1d((mask).astype(float), sigma=smooth_sigma)
    return mask


def dynamic_from_statistics(
    medians: np.ndarray, stds: np.ndarray, dtype: np.dtype, nsamps: int = 2**16
) -> np.ndarray:
    """
    Make a dynamic spectra from statistics.

    Args:
        medians - Bandpass medians

        stds - Standard deviation of the bandpass

        dtype - data type of fake file

        nsamps - Number of time samples

    Returns:
        2D random arrays
    """
    nchans = medians.shape[0]
    clone = np.random.normal(size=nchans * nsamps)
    clone = clone.reshape(nsamps, nchans)
    clone *= stds
    clone += medians
    return to_dtype(clone, dtype=dtype)


def clone_spectra(
    dynamic_spectra: np.ndarray, median_filter_length: int = 0
) -> np.ndarray:
    """
    Clone a section of dynamic spectra using Gaussian random numbers.

    Args:
        2D array of dynamic spectra

    Returns:
        Dynamic spectra that has simlar statstics as the given dynamic
        spectra.
    """
    dtype = dynamic_spectra.dtype
    nsamps, _ = dynamic_spectra.shape
    stds, medians = median_abs_deviation_med(dynamic_spectra, axis=0, scale="normal")

    if median_filter_length > 0:
        stds = median_fitter(stds, chans_per_fit=median_filter_length)
        medians = median_fitter(medians, chans_per_fit=median_filter_length)
    return dynamic_from_statistics(
        medians=medians, stds=stds, dtype=dtype, nsamps=nsamps
    )
