#!/usr/bin/env python3
"""
Pulse analysis routine
"""
import logging
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
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

    locations: Union[int, float, np.ndarray]
    snrs: float
    std: np.float64


def detect_all_pulses(
    time_series: np.ndarray,
    box_car_length: int,
    sigma: float = 6,
    smoothing_factor: int = 4,
) -> PulseInfo:
    """
    Detect pulses in a dedisperesed series.

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

    Don't scale SNR as described in https://arxiv.org/pdf/2011.10191.pdf
    because we want the actual SNR of the time series.
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


def detect_max_pulse(
    time_series: np.ndarray,
    box_car_length: int,
    # sigma: float = 6,
    smoothing_factor: int = 4,
) -> PulseInfo:
    """
    Detect the largest pulse in a dedisperesed series.
    reports back the location, pulse SNR and time series
    standard deviation as computed by Median Absolute Deviation.

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

    Don't scale SNR as described in https://arxiv.org/pdf/2011.10191.pdf
    because we want the actual SNR of the time series.
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

    # Find max value
    max_index = np.argmax(flattened_times_series)
    max_value = flattened_times_series[max_index]

    # don't use pulse for calculating SNR
    mask = np.zeros(flattened_times_series.shape, dtype=bool)
    mask[max_index] = True
    ndimage.binary_dilation(
        mask, iterations=box_car_length * smoothing_factor // 2, output=mask
    )

    std = stats.median_abs_deviation(flattened_times_series[~mask], scale="normal")
    snr = max_value / std
    # if snr < sigma:
    #     return PulseInfo(np.nan, np.nan, std)
    return PulseInfo(max_index, snr, std)


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
    pulse_search_params: PulseSearchParamters
    pulse_locations: np.ndarray

    @property
    def percent_with_pulses(self):
        mask = self.snrs >= self.pulse_search_params.sigma
        return 100 * mask.mean()

    def plot_snrs(self) -> None:
        """
        Plot Signal to Noise Ratios as a function of time.
        """
        snr_mask = self.snrs > self.pulse_search_params.sigma
        # pulse locations in seconds
        locs = (
            self.pulse_locations[snr_mask]
            * self.pulse_search_params.yr_obj.your_header.tsamp
        )
        plt.plot(locs, self.snrs[snr_mask])
        plt.xlabel("Time [Seconds]")
        plt.ylabel("SNR")
        plt.title("SNR vs. Time")
        plt.show()

    def plot_stds(self) -> None:
        """
        Plot Standard Deviation (As calculated via Median Absolute Deviation)
        as a function of time.
        """
        locs = self.pulse_locations * self.pulse_search_params.yr_obj.your_header.tsamp
        plt.plot(locs, self.stds)
        plt.xlabel("Time [Seconds]")
        plt.ylabel("Standard Deviation")
        plt.title("Standard Deviation vs. Time")
        plt.show()

    def plot_folded_dynamic(self, median_filter_length=29) -> None:
        """
        Plot the folded dynamic spectra.

        Args:
            median_filter_length - The length of the median filter used to
                                    remove the bandpass
        """
        nsamps, _ = self.folded.shape
        xmax = 1000 * nsamps * self.pulse_search_params.yr_obj.your_header.tsamp
        ymin = self.pulse_search_params.yr_obj.chan_freqs[-1]
        ymax = self.pulse_search_params.yr_obj.chan_freqs[0]
        bandpass = ndimage.median_filter(
            np.median(self.folded, axis=0), size=median_filter_length, mode="mirror"
        )
        plt.imshow((self.folded - bandpass).T, extent=[0, xmax, ymin, ymax])
        plt.xlabel("Time [millisecond]")
        plt.ylabel("Frequency [Mhz]")
        plt.title("Folded Dynamic Spectra")
        plt.show()

    def plot_folded_profile(self) -> None:
        """
        Plot the folded pulse profile and print the Signal to Noise
        """
        time_series = self.folded.mean(axis=1)
        times = (
            np.arange(len(time_series))
            * self.pulse_search_params.yr_obj.your_header.tsamp
        )
        plt.plot(times, time_series)
        plt.xlabel("Time [Seconds]")
        plt.ylabel("Intensity")
        plt.title("Folded Time Series")
        plt.show()
        max_pulse = detect_max_pulse(
            time_series, box_car_length=self.pulse_search_params.box_car_length
        )
        print(f"Folded Pulse SNR: {max_pulse.snrs:.2f}")


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
        # cut the rolled part
        dynamic_spectra_trim = dynamic_spectra_dispered[
            pulse_search_params.samples_lost : -pulse_search_params.samples_lost
        ]
        folded += dynamic_spectra_trim
        time_series = dynamic_spectra_trim.mean(axis=1)

        pulse = detect_max_pulse(
            time_series,
            box_car_length=pulse_search_params.box_car_length,
            # sigma=pulse_search_params.sigma,
        )

        stds[j] = pulse.std
        snrs[j] = pulse.snrs

    return PulseSNRs(
        snrs=snrs,
        stds=stds,
        folded=folded,
        pulse_search_params=pulse_search_params,
        pulse_locations=pulse_locations,
    )
