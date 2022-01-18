#!/usr/bin/env python3
"""
Pulse analysis routine
"""
from dataclasses import dataclass
from typing import Union

import numpy as np
from jess.dispersion import dedisperse
from scipy import ndimage, signal, stats


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

    std = stats.median_abs_deviation(flattened_times_series, scale="normal")
    normatlized_time_series = flattened_times_series / std
    if box_car_length > 1:
        window = signal.boxcar(box_car_length) / np.sqrt(box_car_length)
        normatlized_time_series = signal.fftconvolve(
            window, normatlized_time_series, "full"
        )
        normatlized_time_series = normatlized_time_series[
            box_car_length // 2 - 1 : -box_car_length // 2
        ]
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

    location: Union[np.int64, None]
    snr: Union[np.float64, None]


def find_max_pulse(pulses: PulseInfo, start_idx: int, end_idx: int) -> MaxPulse:
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
    mask = (pulses.locations >= start_idx) & (pulses.locations <= end_idx)
    snrs = pulses.snrs[mask]

    if len(snrs) > 0:
        max_pulse_location = np.argmax(snrs)
        return MaxPulse(
            pulses.locations[mask][max_pulse_location], snrs[max_pulse_location]
        )

    # No suitable pulses
    return MaxPulse(None, None)
