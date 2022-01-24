#!/usr/bin/env python3
"""
Pulse analysis routine
"""
import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
from jess.dispersion import dedisperse, delay_lost
from rich.progress import track
from scipy import ndimage, signal, stats
from your import Your


def dedisped_time_series(
    dynamic_spectra: np.ndarray,
    dm: float,  # pylint: disable=invalid-name
    tsamp: float,
    chan_freqs: np.ndarray,
) -> np.ndarray:
    """
    Get the dedispered time series from a chunk of dynamic spectra.

    Args:
        dynamic_spectra - 2D spectra with time on the vertical axis

        dm - The dispersion measure

        tsamp - Time sample of the data

        chan_freqs - The channel frequencies

    Returns:
        Time series at a the given DM
    """
    dynamic_spectra_dispered = dedisperse(
        dynamic_spectra, dm=dm, tsamp=tsamp, chan_freqs=chan_freqs
    )
    return dynamic_spectra_dispered.mean(axis=1)


@dataclass
class PulseInfo:
    """
    Pulse info result

    locations - sample location of the pulses

    snrs - Signal to noise of pulses

    std - Standard devivation of time series
    """

    locations: np.ndarray
    snrs: np.ndarray
    std: np.float64


def detect_pulses(
    time_series: np.ndarray,
    box_car_length: int,
    sigma: float = 6,
    smoothing_factor: int = 4,
) -> PulseInfo:
    """
    Detect pulses in a dedisperesed serries.

    Args:
        time_series - The dedispersed time series

        box_car_length - Length of the boxcar

        sigma - Return pulses with significance above
                this

        smoothing_factor - Median filter is smoothing_factor*box_car_length

    Returns:
        dataclass[Locations, SNRs]

    Deterned the time series by subtracting off the running median
    Thesis described in Bardell Thesis, but Heimdall uses a different
    method

    Scale the SNR with 1/sqrt(boxcar length)
    as described in https://arxiv.org/pdf/2011.10191.pdf
    """
    flattened_times_series = time_series - ndimage.median_filter(
        time_series, box_car_length * smoothing_factor
    ).astype(float)

    # this follows https://arxiv.org/pdf/2011.10191.pdf
    # std = stats.median_abs_deviation(flattened_times_series, scale="normal")
    # normatlized_time_series = flattened_times_series / std

    if box_car_length > 1:
        window = signal.boxcar(box_car_length) / np.sqrt(box_car_length)
        flattened_times_series = signal.fftconvolve(
            window, flattened_times_series, "full"
        )
        flattened_times_series = flattened_times_series[
            box_car_length // 2 - 1 : -box_car_length // 2
        ]

    std = stats.median_abs_deviation(flattened_times_series, scale="normal")
    normatlized_time_series = flattened_times_series / std

    locations = np.argwhere(normatlized_time_series > sigma)
    return PulseInfo(locations, normatlized_time_series[locations], std)


@dataclass
class MaxPulse:
    """
    Max pulse location

    location - Sample location of max pulse

    snr - Signal to noise ratio

    If `None`, no pulse fitting requirements found.
    """

    location: Union[np.int64, float]
    snr: Union[np.float64, float]


@dataclass
class PulseSearchParamters:
    """
    The parameters to use for the single pulse search.

    file - Fits for Fil contaning the pulse.

    first_pulse - Location of the first pulse in seconds.

    period - Period of the pulse in seconds.

    dm - Dispersion Measure of the pulse

    box_car_length - Length of the boxcar for the matched filter.

    sigma - Pulse threshold

    start - Start sample to process (default to first sample)

    stop - Final sample to process (default is EOF)
    """

    # pylint: disable=invalid-name
    file: str
    first_pulse: int
    period: int
    dm: float
    box_car_length: int
    samples_around_pulse: int
    sigma: float = 6
    start: int = 0
    stop: int = -1

    def __post_init__(self):

        self.yr_obj: Your = Your(self.file)

        self.samples_lost: int = delay_lost(
            dm=self.dm,
            chan_freqs=self.yr_obj.chan_freqs,
            tsamp=self.yr_obj.your_header.tsamp,
        )

        self.period = self.period / self.yr_obj.native_tsamp

        if self.stop == -1:
            self.stop = self.yr_obj.your_header.nspectra


def find_max_pulse(
    pulses: PulseInfo, start_idx: int = 0, end_idx: int = -1
) -> MaxPulse:
    """
    Find the maximum pulse between two indices.

    Args:
        pulses - The dataclass from detected pulses

        start_idx - Start index of the the range

        end_idx - End index of range

    Returns:
        dataclass(location index, SNR)
        if no pulse in range, returns (None, None)
    """
    if end_idx == -1:
        mask = pulses.locations >= start_idx
    else:
        mask = (pulses.locations >= start_idx) & (pulses.locations <= end_idx)

    snrs = pulses.snrs[mask]

    npulse = len(snrs)
    if npulse > 0:
        logging.debug("Found %i pulses", npulse)
        max_pulse_location = np.argmax(snrs)
        return MaxPulse(
            pulses.locations[mask][max_pulse_location], snrs[max_pulse_location]
        )
    logging.debug("No suitable pulses!")
    # No suitable pulses
    return MaxPulse(np.nan, np.nan)


def locations_of_pulses(pulse_search_params: PulseSearchParamters) -> np.ndarray:
    """
    Make an array of pulse locations from the start location and period.

    Args:
        pulse_search_params - The dataclass that has the search parameters

    Returns:
        Location of the pulses in samples
    """
    first_pulse = pulse_search_params.first_pulse
    while (
        first_pulse
        < pulse_search_params.start
        + pulse_search_params.box_car_length // 2
        + pulse_search_params.samples_around_pulse
        + pulse_search_params.samples_lost
    ):
        logging.warning(
            "First pulse (%i) does not give enough padding, increasing by %i",
            first_pulse,
            pulse_search_params.period,
        )
        first_pulse += pulse_search_params.period

    padded_end = (
        pulse_search_params.stop
        - pulse_search_params.box_car_length // 2
        - pulse_search_params.samples_around_pulse
        - pulse_search_params.samples_lost
    )

    locations = np.arange(first_pulse, padded_end, pulse_search_params.period)
    return np.around(locations).astype(int)


@dataclass
class PulseSNRs:
    """
    Results of pulse search

    snrs - Pulse Signal to Noise Ratio

    stds - Standard Deviations of pulse block
           Computed via Median Abs Deviation

    folded - Folded dynamic spectra
    """

    snrs: np.ndarray
    stds: np.ndarray
    folded: np.ndarray


def search_file(
    pulse_search_params: PulseSearchParamters,
    pulse_locations: np.ndarray,
) -> PulseSNRs:
    """
    Search a Fil or Fits file for pulses.

    Args:
        pulse_search_params - dataclass with the pulse search paramters.

        pulse_locations - Array with the locations of the center of the pulse.

    Returns:
        PulseSNRs - Dataclass that has the pulse snrs, standard deviations, and
                    the folded profile.
    """

    offset = (
        pulse_search_params.box_car_length // 2
        + pulse_search_params.samples_around_pulse
        + pulse_search_params.samples_lost
    )
    nsamp = (
        pulse_search_params.box_car_length
        + 2 * pulse_search_params.samples_around_pulse
        + 2 * pulse_search_params.samples_lost
    )
    snrs = np.zeros(pulse_locations.shape, dtype=np.float64)
    stds = np.zeros(pulse_locations.shape, dtype=np.float64)
    folded = np.zeros(
        (
            pulse_search_params.box_car_length
            + 2 * pulse_search_params.samples_around_pulse,
            pulse_search_params.yr_obj.your_header.nchans,
        ),
        dtype=np.float64,
    )

    for j, location in enumerate(track(pulse_locations)):
        dynamic_spectra = pulse_search_params.yr_obj.get_data(
            int(location) - offset, nsamp
        )
        dynamic_spectra_dispered = dedisperse(
            dynamic_spectra,
            dm=pulse_search_params.dm,
            tsamp=pulse_search_params.yr_obj.your_header.tsamp,
            chan_freqs=pulse_search_params.yr_obj.chan_freqs,
        )
        dynamic_spectra_trim = dynamic_spectra_dispered[
            pulse_search_params.samples_lost : -pulse_search_params.samples_lost
        ]
        folded += dynamic_spectra_trim
        time_series = dynamic_spectra_trim.mean(axis=1)

        # cut the rolled part
        pulse = detect_pulses(
            time_series,
            box_car_length=pulse_search_params.box_car_length,
            sigma=pulse_search_params.sigma,
        )
        # print(pulse)

        stds[j] = pulse.std

        # pulse_window = offset // 2
        snrs[j] = find_max_pulse(pulse, 0, -1).snr

    return PulseSNRs(snrs, stds, folded)
