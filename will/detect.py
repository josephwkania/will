#!/usr/bin/env python3
"""
Pulse analysis routine
"""
import logging
import os
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from jess.calculators import median_abs_deviation_med
from jess.dispersion import dedisperse, delay_lost
from jess.fitters import median_fitter
from rich.console import Console
from rich.progress import track
from rich.table import Table
from scipy import ndimage, signal, stats
from your import Your


# pylint: disable=invalid-name
def find_first_pulse(
    file: str,
    dm: float,
    start: int,
    gulp: int,
    box_car_length: int = 1,
) -> None:
    """
    Help to find the index of the first pulse by ploting the
    dedispersed dynamic spectra and time series.

    Args:
        file - Path to the file to investigate

        dm - Dispersion measure of the pulsar

        start - Start Index

        gulp - Number of samples to get

        box_car_length - If > 1, dynamic spectra and time series
                         get convolved with a boxcar of this length

        num_cands - Number of candidates to print

    """
    yr_obj = Your(file)
    lost = delay_lost(
        dm=dm, chan_freqs=yr_obj.chan_freqs, tsamp=yr_obj.your_header.tsamp
    )
    dynamic_spectra = yr_obj.get_data(start, gulp + lost)
    dynamic_spectra = dedisperse(
        dynamic_spectra,
        dm=dm,
        tsamp=yr_obj.your_header.tsamp,
        chan_freqs=yr_obj.chan_freqs,
    )[:-lost]

    dynamic_spectra = dynamic_spectra - median_fitter(
        np.median(dynamic_spectra, axis=0)
    )

    if box_car_length > 1:
        _, nchans = dynamic_spectra.shape
        window = signal.boxcar(box_car_length) / np.sqrt(box_car_length)
        dynamic_spectra = signal.fftconvolve(
            np.broadcast_to(window[:, None], (box_car_length, nchans)),
            dynamic_spectra,
            "full",
            axes=0,
        )
        dynamic_spectra = dynamic_spectra[
            box_car_length // 2 - 1 : -box_car_length // 2
        ]

    time_series = dynamic_spectra.mean(axis=1)
    ts_std, ts_med = median_abs_deviation_med(time_series, scale="Normal")
    time_series -= ts_med
    time_series /= ts_std
    max_indices = time_series.argsort()[::-1][:5]
    max_values = time_series[max_indices]
    max_indices += start

    std, med = median_abs_deviation_med(dynamic_spectra, scale="normal", axis=None)
    fig, axis = plt.subplots(2, figsize=(10, 10), sharex=True)
    fig.suptitle(os.path.basename(file), size=15)
    axis[0].plot(time_series)
    axis[0].set_ylabel("SNR", size=12)
    axis[1].imshow(dynamic_spectra.T, vmin=med - 3 * std, vmax=med + 6 * std)
    axis[1].set_xlabel("Time Sample #", size=12)
    axis[1].set_ylabel("Channel #", size=12)
    axis[1].set_aspect("auto")
    plt.tight_layout()
    plt.show()

    table = Table(title="5 Largest SNRs")
    table.add_column("Index", justify="center")
    table.add_column("SNR", justify="center")
    for loc, snr in zip(max_indices, max_values):
        table.add_row(f"{loc}", f"{snr:.1f}")
    console = Console()
    console.print(table)


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

    locations: Union[np.int64, float, np.ndarray]
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
        PulseInfo[Locations, SNRs]

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
    search_window_frac: float = 0.50,
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
    bottom_limit = np.around((0.5 - search_window_frac / 2) * len(time_series)).astype(
        int
    )
    top_limit = np.around((0.5 + search_window_frac / 2) * len(time_series)).astype(int)
    window_slice = slice(bottom_limit, top_limit)
    max_index = np.argmax(flattened_times_series[window_slice])
    # adjust the index so it has the original value
    max_index += bottom_limit
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

    # takes care if only one value is passed
    assert isinstance(pulses.snrs, np.ndarray), "You need to provide an array"

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


@dataclass
class PulseSearchParamters:
    """
    The parameters to use for the single pulse search.

    file - Fits for Fil contaning the pulse.

    first_pulse - Location of the first pulse in seconds.

    period - Period of the pulse in seconds.

    dm - Dispersion Measure of the pulse

    box_car_length - Length of the boxcar for the matched filter.

    samples_around_pulse - The number of samples on either side of
                           of the boxcar

    search_window_frac - The fraction, around the center, of the time
                         series search for a pulse.

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
    search_window_frac: float = 0.50
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
        """
        The percent of pulse bins with a single above snr
        present.
        """
        mask = self.snrs >= self.pulse_search_params.sigma
        return 100 * mask.mean()

    @property
    def folded_properties(self):
        """
        Get the pulse properties of the folded profile
        """
        time_series = self.folded.mean(axis=1)
        max_pulse = detect_max_pulse(
            time_series,
            box_car_length=self.pulse_search_params.box_car_length,
            search_window_frac=self.pulse_search_params.search_window_frac,
        )
        return max_pulse

    def plot_snrs(self, cut_snrs: bool = False, title: Union[None, str] = None) -> None:
        """
        Plot Signal to Noise Ratios as a function of time.

        Args:
            cut_snrs: Don't plot SNRs below the cutoff specified in PulseSearchParamters

            title - The title of the plot, default `SNR vs. Time`
        """

        # pulse locations in seconds
        locs = self.pulse_locations * self.pulse_search_params.yr_obj.your_header.tsamp
        if cut_snrs:
            snr_mask = self.snrs > self.pulse_search_params.sigma
            plt.plot(locs[snr_mask], self.snrs[snr_mask])
        else:
            plt.plot(locs, self.snrs)

        if title is None:
            title = "SNR vs. Time"

        plt.xlabel("Time [Seconds]")
        plt.ylabel("SNR")
        plt.title(title)
        plt.show()

    def plot_stds(self, title: Union[None, str] = None) -> None:
        """
        Plot Standard Deviation (As calculated via Median Absolute Deviation)
        as a function of time.

        Args:
            title - The tile of the plot, default `Standard Deviation vs. Time`
        """
        locs = self.pulse_locations * self.pulse_search_params.yr_obj.your_header.tsamp

        if title is None:
            title = "Standard Deviation vs. Time"

        plt.plot(locs, self.stds)
        plt.xlabel("Time [Seconds]")
        plt.ylabel("Standard Deviation")
        plt.title(title)
        plt.show()

    def plot_folded_dynamic(self, median_filter_length: int = 29) -> None:
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
        max_pulse = self.folded_properties
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

    nsamp = (
        pulse_search_params.box_car_length
        + 2 * pulse_search_params.samples_around_pulse
        + pulse_search_params.samples_lost
    )
    offset = (
        pulse_search_params.box_car_length // 2
        + pulse_search_params.samples_around_pulse
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

    for j, location in enumerate(
        track(pulse_locations, description="Searching for Pulses", transient=True)
    ):
        start = np.around(location).astype(int) - offset
        # this seemed to happen occasionally, I think due to double counting
        # the delay lost, this has been fixed.
        if start < 0:
            logging.warning("Start is before start of file %i", start)
        dynamic_spectra = pulse_search_params.yr_obj.get_data(start, nsamp)
        dynamic_spectra_dispered = dedisperse(
            dynamic_spectra,
            dm=pulse_search_params.dm,
            tsamp=pulse_search_params.yr_obj.your_header.tsamp,
            chan_freqs=pulse_search_params.yr_obj.chan_freqs,
        )
        # cut the rolled part
        dynamic_spectra_trim = dynamic_spectra_dispered[
            : -pulse_search_params.samples_lost
        ]
        folded += dynamic_spectra_trim
        time_series = dynamic_spectra_trim.mean(axis=1)

        pulse = detect_max_pulse(
            time_series,
            box_car_length=pulse_search_params.box_car_length,
            # sigma=pulse_search_params.sigma,
            search_window_frac=pulse_search_params.search_window_frac,
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
