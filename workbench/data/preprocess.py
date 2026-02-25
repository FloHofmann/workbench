from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from workbench.videography.camera_process import track_platform
from workbench.data.h5data import loadmat, save_processed_data
from workbench.data.db_interaction import createFolderStructure, connectDb
import pandas as pd
import numpy as np
import scipy.io
import h5py
from scipy.signal import filtfilt, butter


def process_exp(folderpath, *args):
    import warnings
    """
    input path to start processing of an experiment
    requires path to exported .mat file from .smrx
    entry is a sqlite row from the database. it provides
    the function with recording information and path to
    various data sources.
    inputting args tracking: true will search for a baseline
    video and extract the rotational data for the platform
    """
    if "data" in Path(folderpath).stem.lower():
        datapath = folderpath
        folderpath = Path(folderpath).parent
    else:
        datapath = sorted(Path(folderpath).glob('Data*.mat'))[0]

    name = Path(folderpath).stem
    print(f"{str(folderpath).split("\\")[-5:-1]} is being processed")

    # load raw data
    data = loadmat(datapath)
    # initialize the spikesorting
    info, raw, sorted_spikes = data.export_processed_data()
    channels = {
        key.split("_")[-1][2:]: val
        for key, val in raw.items()
        if "ch" in key.split("_")[-1].lower()
    }
    # retrieve spikesorted data from spikesorted
    spikesorted = sorted_spikes.exportData()
    keyboard_channel = channels['31']
    angles = None
    x = None
    y = None
    n_ttls = None
    nframes = None
    ttl_times = None
    ch4_stims = None
    ch5_stims = None

    if 'tracking' in args:
        # ToDo videopath needs to be computed first
        videopath = sorted(Path(folderpath).glob('*.avi'))[0]
        # outputs a 4,1 tuple
        angles, nframes, x, y = track_platform(videopath)

        if 'ch3' in [x.lower for x in raw.keys()] and channels['3']['title'] == 'TTL puls':
            ttl_times, n_ttls = get_tracking_ttl(channels['3'])
            if n_ttls != nframes:
                warnings.warn(
                    f"Number of TTLs do not match to number of frames for {name}")
    if 'behav' in args:
        if 'sound' in str(datapath):
            ch4_stims = correct_keyboard_times(
                channels['4'], keyboard_channel['times'].flatten())
        if 'tones' in str(datapath):
            ch5_stims = correct_keyboard_times(
                channels['5'], keyboard_channel['times'].flatten())

    save_processed_data(
        folderpath,
        sorted_spikes=spikesorted,
        angles=angles,
        x_values=x,
        y_value=y,
        nframes=nframes,
        n_ttls=n_ttls,
        ttl_times=ttl_times,
        event_times=keyboard_channel['times'],
        event_codes=keyboard_channel['codes'],
        sampling_rate=1/channels['1']['interval'],
        ch4_stims=ch4_stims,
        ch5_stims=ch5_stims
    )


def correct_keyboard_times(channel: dict, keyboard_times):
    """
    Docstring for correct_keyboard_times
    
    :param channel: A signal channel from loading the .mat file exported from spike 2.
        The channel is used to receive the signal from the sound generation fed back into the machine.
        Has to have a key 'values' which stores the signal at 25 kHz sampling rate.
    :type channel: dict
    :param keyboard_times: This is a list storing the spike 2 set times at which the keyboard press happenend.
        This is inherently inaccurate and not stable. The values need to be in seconds.
    """
    from scipy.signal import find_peaks

    pks, _ = find_peaks(np.abs(np.diff(channel['values'])), height=0.04)
    peaktimes = (pks-1)/25000

    corrected = np.zeros_like(keyboard_times)

    for i, t in enumerate(keyboard_times):
        diffs = np.abs(peaktimes - t)
        min_diff = np.min(diffs)
        if min_diff > 0.01:
            corrected[i] = t + 0.002
        else:
            corrected[i] = peaktimes[np.argmin(diffs)]

    return corrected


def get_tracking_ttl(ch3):
    """
    returns ttl_times as samples (pks)
    returns the total amount of ttls
    """
    from scipy.signal import find_peaks
    import numpy as np
    values = ch3['values'].flatten()
    val_diff = np.diff(values)
    pks, _ = find_peaks(val_diff, threshold=5)
    return pks, len(pks)


def batch_process():
    conn, _ = connectDb(
        r"\\172.25.250.112\burgalossi\lab share\Data\Florian\Recordings_FH.db")

    createFolderStructure(conn)

    query = """
            SELECT Folderpath FROM Recordings
            WHERE exp_type == 'juxta' AND use = 1
            ORDER BY Cell_Id ASC;
            """
    db = pd.read_sql(query, conn)
    for i in db['Folderpath'].to_list():
        target_dir = Path(i)
        h5_target = list(target_dir.glob("exp_data.h5"))
        # if not h5_target:
        if 'baseline' in i.lower():
            process_exp(i, 'tracking')
        else:
            process_exp(i)

    conn.close()


def TriggRasterPY(
    tone_triggers,
    spkT,
    SR=25000,
    halftime=0.6,
    nbins=100
):
    """
    This is a replica of the TriggRaster.m function.
    It does not implement the shuffling procedure, so far no need for it.
    There is a minor discrepancy between the rate values of roughly 1 Hz.
    This comes to play because Matlab treats the value of time2bin differently and rounds it weirdly
    """
    import numpy as np

    spkT_samples = np.round(spkT*SR)
    halfsamples = round(halftime*SR)
    chunks = np.zeros((len(tone_triggers), (halfsamples*2)+1))

    for idx, ttrig in enumerate(tone_triggers):
        c = np.linspace(ttrig-halfsamples, ttrig+halfsamples,
                        (halfsamples*2)+1, dtype=int)
        isCell = np.isin(c, spkT_samples)
        chunks[idx, :] = isCell

    timechunk = np.linspace(-halftime, halftime, (halfsamples*2)+1)

    edges = np.round(np.linspace(0, len(timechunk), nbins))
    rows, cols = np.nonzero(chunks)
    sorting_columns_idx = np.argsort(cols)
    cols = cols[sorting_columns_idx]
    rows = rows[sorting_columns_idx]
    counts, edges = np.histogram(cols, bins=edges)

    # mimic the medfilt1
    edges2plot = np.concatenate([[edges[0]], np.median(
        np.vstack([edges[:-1], edges[1:]]), axis=0)])
    edges2plot = edges2plot[1::]
    time2scale = np.round(
        np.median(np.diff(timechunk[np.round(edges2plot).astype(int)])),
        4)
    ntrial_timebin = len(tone_triggers)*time2scale

    # estimation of firing rate
    rate = np.divide(counts, ntrial_timebin)
    raster = {
        'raster_times': timechunk[cols]*1000,
        'raster_row': rows,
        'raster_rate': rate,
        'time': np.round(timechunk[np.round(edges2plot).astype(int)]*1000),
        'time_bin': time2scale*1000
    }
    return raster


def combTableCreate(datatable):
    """
    Creates a combined table of preprocessed data.
    Input is a pandas table containing the sql database of the recordings info.
    Datapath is the path to where the final table is supposed to be saved to.
    Filename is the name of the final table
    """
    import polars as pl
    COLS = ["Animal_Id", "Cell_Id", "Condition", "exp_type", "Folderpath"]
    datatable = pl.from_pandas(datatable[COLS])
    comb_table = []

    for row in datatable.iter_rows(named=True):
        row_result = processTableRow(row)
        comb_table.append(row_result)

    return comb_table


def processTableRow(
    row_dict,
    pupil_sr=50,
    ephys_sr=25000,
    nbins=800,
    halftime_spikes=8,
    halftime_mot=8,
):
    """
    This processes a singular datarow and returns the a dictionary with the computed information about the recording that was put through this funciton.
    Port of tonestim_output_ramp.m (ramp condition), returning a
    1-row pandas DataFrame. Uses the schema helpers so all columns
    exist even if certain data (like pupil) are missing.
    """

    # unpack the row_dict
    animal_id = row_dict['Animal_Id']
    cell_id = row_dict["Cell_Id"]
    condition = row_dict["Condition"]
    exp_type = row_dict["exp_type"]
    folderpath = Path(row_dict["Folderpath"])

    # constants
    rasterBins = halftime_spikes * nbins
    smooth_pupil_value = 14    # for smoothing pupil signal
    mot_samples = halftime_mot * 2 * pupil_sr

    if condition.lower() == 'baseline':
        # fix parameters for baseline
        params = {
            'speed_smoothing':15,
            'speed_threshold':0.1,
            'polarBins':36,
            'smooth_angular_win':2,
            'n_shuffle':1000
        }

        speed_smoothing = 15
        speed_threshold = 0.1  # radians per second
        polarBins = 36
        halftime_spikes = 2

        print(f"{'':-^100}")
        print(f"Animal_Id {animal_id}; Cell_Id {
              cell_id}, Baseline being processed")
        print(f"{'':-^100}")
        # load the data
        try:
            d_path = Path(folderpath, 'exp_data.mat')
            exp = scipy.io.loadmat(
                d_path,
                struct_as_record=False,
                squeeze_me=False,
                simplify_cells=True,
            )
        except Exception:
            print(f"Missing exp_data.mat for {animal_id} Cell_Id {
                  cell_id}")
            return

        if 'processed_data' not in exp.keys():
            print("No processed data in exp_data")
            return
        
        if 'raw_data' not in exp.keys():
            print("No raw data in exp_data")
            return

        processed_data = exp.get('processed_data', None)
        raw_data = exp.get('raw_data', None)

        if 'tracking_data' not in processed_data.keys():
            print("No tracking data in exp_data")
            return

        # get relevant variables
        spkT = processed_data['spike_sorting_data']['spike_times']
        videoT = raw_data['ephys_data']['ttl_times']
        angles = processed_data['tracking_data']['angles']

        # get angle speed and velocity
        interval = np.median(np.diff(videoT))
        _, angular_speed = angular_derivative_from_angles(
            angles, interval, speed_smoothing)

        spike_angular_speed = np.interp(spkT, videoT, angular_speed)
        spkT_high_speed = spkT[spike_angular_speed > speed_threshold]

        nspk = len(spkT_high_speed)
        hdRateSmooth, _, direction, spike_angle_mean, _, hdRateUnsmooth, _ = plot_polar_graph(
            spkT_high_speed,
            videoT,
            angles,
            isplot_polarplot=False,
            speed_threshold=speed_threshold,
            num_directions=polarBins
        )
        hdRate = len(spkT_high_speed) / abs(spkT[-1] - spkT[0])

        # hdRateUnsmooth(isnan(...)) = 0
        hdRateUnsmooth = np.array(hdRateUnsmooth, dtype=float, copy=True)
        hdRateUnsmooth[np.isnan(hdRateUnsmooth)] = 0.0

        # Rayleigh test:
        # note in MATLAB they use 'direction' and drop the last bin
        # d = 360/polarBins (degrees) -> radians
        angles_rad = np.deg2rad(direction[:-1])
        weights = hdRateUnsmooth[:-1]
        bin_spacing = np.deg2rad(360.0 / polarBins)

        rayleigh_pval, z_stat, hdi = circ_rtest(
            angles_rad, w=weights, d=bin_spacing)

        # Align tuning curve
        center = int(np.ceil(len(direction) / 2))
        rateClose2Center = direction - spike_angle_mean  # degrees
        minRateIndex = int(np.argmin(np.abs(rateClose2Center)))
        shift = center - minRateIndex

        # calculate p-values
        #dir_centers_deg = np.asarray(direction[:-1], dtype=float)
        #w = np.asarray(hdRateSmooth, dtype=float)
        #binwidth_rad = np.deg2rad(np.median(np.diff(direction)))
        pval, hdi = calc_hd_significance(processed_data=processed_data, raw_data=raw_data, n_shuffle=1000, params=params)

        half_wd, peak2tr, waveform, _, _, _, _ = calc_spikewidth_from_traces(
            spike_traces=processed_data["spike_sorting_data"]["spike_traces"],
            plot=True,
            save_dir=None
        )

        processed_row = {
            'Animal_Id': animal_id,
            'Cell_Id': cell_id,
            'Condition': condition,
            'hdRateSmooth': hdRateSmooth,
            'hdRateSmoothAlign': np.roll(hdRateSmooth, shift),
            'HDpeakFRsmooth': hdRateSmooth.max(),
            'HDpeakFR': hdRateUnsmooth.max(),
            'pValS': pval,
            'pValR': rayleigh_pval,
            'HDInx': hdi,
            'nSpk': nspk,
            'HDFR': hdRate,
            'HDAngle': spike_angle_mean,
            'halfwidth': half_wd,
            'peak2trough': peak2tr,
            'waveform': waveform,
        }
        return processed_row

    elif condition.lower() in ('tones-95', 'soso', 'ramp', 'intensity'):
        # init shared vars
        tone_onset = None
        pupil_psth = None
        whisk_psth = None
        eye_psth = None
        RasterTimes = dict()
        RasterRows = dict()
        RasterRate = dict()
        whisk_avg = dict()
        pupil_avg = dict()
        eye_avg = dict()
        trigger_time = None
        processed_row = dict()
        processed_row['Animal_Id'] = animal_id
        processed_row['Cell_Id'] = cell_id
        processed_row['Condition'] = condition

        # retrieve information about the tone letters being used
        if condition.lower() == 'tones-95':
            tone_letters = ['a', 'w', 'e', 'r', 't', 'z', 'u', 'u', 'i', 'k', 'p']
        elif condition.lower() == 'soso':
            tone_letters = ['a', 'w', 'e', 'r']
        elif condition.lower() == 'ramp':
            if exp_type == 'juxta':
                tone_letters = ['a', 'w', 'e', 't']
            else:
                tone_letters = ['a', 'w', 'e', 'r', 't', 'z', 'u', 'u', 'i', 'k', 'p']
        elif condition.lower() == 'intensity':
            tone_letters = ['a', 'w', 'e', 'r', 't', 'z', 'u', 'u', 'i', 'k', 'p']

        print(f"{'':-^100}")
        print(f"Animal_Id {animal_id}; Cell_Id {
              cell_id}, {condition} being processed")
        print(f"{'':-^100}")

        if exp_type == 'juxta':
            try:
                d_path = Path(folderpath, 'exp_data.mat')
                print(d_path)
                exp = scipy.io.loadmat(
                    d_path,
                    struct_as_record=False,
                    squeeze_me=False,
                    simplify_cells=True
                )
            except Exception:
                print(f"Missing exp_data.mat for {
                      animal_id} Cell_Id {cell_id}")
                # we'll still build an empty row at the end
                spkT = None
            else:
                processed_data = exp.get('processed_data', None)
                raw_data = exp.get('raw_data', None)
                # exp_info = exp.get('info', None)
                # analysis = exp.get('analysis', None)

            # processing
            # spike times
            spkT = processed_data['spike_sorting_data']['spike_times']

            # get tone times
            tone_times = raw_data['ephys_data']['keyboard_times'][:]
            # get tone codes
            tone_codes = raw_data['ephys_data']['keyboard_codes'][:, 0]

            # sampling rate
            ephys_sr = round(raw_data['ephys_data']['sampling_rate'])

            # get time calls in ephys samples
            tone_onset = np.zeros(len(tone_times))
            for idx, _ in enumerate(tone_times):
                tmp = np.abs(raw_data['ephys_data']
                             ['ephy_times'] - tone_times[idx])
                min_idx = np.argmin(tmp)
                tone_onset[idx] = min_idx+1

            # obtain psth information
            for idx, char in enumerate(tone_letters):
                this_code = ord(char)  # translate character to ascii
                # find occurrences of ascii character in all stimulations
                this_codes = np.flatnonzero(this_code == tone_codes)

                # filter for code trigger timings in samples
                tone_triggers = tone_onset[this_codes]

                raster = TriggRasterPY(
                    tone_triggers,
                    spkT,
                    ephys_sr,
                    halftime_spikes,
                    rasterBins,
                )
                RasterRows[char] = raster['raster_row']
                RasterTimes[char] = raster['raster_times']
                RasterRate[char] = raster['raster_rate']

            processed_row['RasterRows'] = RasterRows
            processed_row['RasterTimes'] = RasterTimes
            processed_row['RasterRate'] = RasterRate

        # Pupil Data
        p = folderpath / "pupil_data.mat"

        pupil_out = load_mat_any(p, top_var = "pupil_out")


        if 'pupil_times' not in pupil_out:
            stri = "No pupil times in Animal {}, Cell_Id {}, Condition {}".format(
                animal_id, cell_id, condition)
            Warning(stri)

        pupil_times = np.asarray(pupil_out["pupil_times"]).squeeze()
        pupil_area = np.asarray(pupil_out["pupil_area"]).squeeze()
        whisk_motion = np.asarray(pupil_out["motion"]).squeeze()
        eyelid = np.asarray(pupil_out["blink"]).squeeze()
        fs = float(np.asarray(pupil_out["sr"]).squeeze())

        # artifacts in motion
        max_min = np.nanmax(whisk_motion) - np.nanmin(whisk_motion)
        mm_std_ratio = max_min / np.nanstd(whisk_motion)
        if mm_std_ratio > 7:
            thr = np.nanpercentile(whisk_motion, 99.7)
            whisk_motion = whisk_motion.copy()
            whisk_motion[whisk_motion > thr] = np.nan

        # pupil smooth + lowpass 1 Hz (order=3)
        pupil_area = smooth_moving_average(pupil_area, smooth_pupil_value)
        pupil_area = but_filter_low(pupil_area, order=3, cutoff_hz=1.0, fs_hz=fs)

        # eyelid smooth + highpass 1 Hz
        eyelid = smooth_moving_average(eyelid, 10)
        eyelid = but_filter_high(eyelid, order=3, cutoff_hz=1.0, fs_hz=fs)

        # whisk smooth
        whisk_motion = smooth_moving_average(whisk_motion, 10)

        # movmedian omitnan
        eyelid = movmedian_omitnan(eyelid, 5)
        whisk_motion = movmedian_omitnan(whisk_motion, 5)

        # normalize range
        pupil_area = normalize_range(pupil_area)
        whisk_motion = normalize_range(whisk_motion)
        eyelid = normalize_range(eyelid)

        pupil_psth = dict()
        whisk_psth = dict()
        eye_psth = dict()

        for idx, tone in enumerate(tone_letters):
            this_code = ord(tone)
            this_codes = np.flatnonzero(this_code == tone_codes)

            tone_triggers = tone_onset[this_codes]

            # triggers to time
            if tone_triggers.max() > 100000:
                tone_triggers = np.divide(tone_triggers, ephys_sr)

            pupil_chunks = np.zeros((len(tone_triggers), mot_samples))
            whisk_chunks = np.zeros((len(tone_triggers), mot_samples))
            eyelid_chunks = np.zeros((len(tone_triggers), mot_samples))

            # loop across stimuli
            for jdx, tt in enumerate(tone_triggers):
                chunk_win = np.array([-halftime_mot, halftime_mot]) + tt

                if chunk_win.min() < 0 or chunk_win.max() > pupil_times[-1]:
                    continue

                my_window = fixed_window_indices(pupil_times, tone_triggers[jdx], mot_samples)
                if my_window is None:
                    continue
                pupil_chunks[jdx, :] = pupil_area[my_window]
                whisk_chunks[jdx, :] = whisk_motion[my_window]
                eyelid_chunks[jdx, :] = eyelid[my_window]

            # motion trigger time
            trigger_time = np.linspace(-halftime_mot,
                                       halftime_mot, mot_samples)

            pupil_psth[tone] = pupil_chunks
            whisk_psth[tone] = whisk_chunks
            eye_psth[tone] = eyelid_chunks
            pupil_avg[tone] = np.nanmean(pupil_chunks, axis=0)
            whisk_avg[tone] = np.nanmean(whisk_chunks, axis=0)
            eye_avg[tone] = np.nanmean(eyelid_chunks, axis=0)

        processed_row['trigger_time'] = trigger_time
        processed_row['pupil_psth'] = pupil_psth
        processed_row['whisk_psth'] = whisk_psth
        processed_row['eye_psth'] = eye_psth
        processed_row['pupil_avg'] = pupil_avg
        processed_row['whisk_avg'] = whisk_avg
        processed_row['eye_avg'] = eye_avg

    else:
        Warning(f"Unknown Condition {condition}")
        return

    return processed_row

def smooth_moving_average(x: np.ndarray, span: int) -> np.ndarray:
    """Approximate MATLAB smooth(x, span) as moving average."""
    x = np.asarray(x, dtype=float)
    span = int(span)
    if span <= 1:
        return x.copy()
    k = np.ones(span, dtype=float) / span
    return np.convolve(x, k, mode="same")


def but_filter_low(x: np.ndarray, order: int, cutoff_hz: float, fs_hz: float) -> np.ndarray:
    wn = cutoff_hz / (fs_hz / 2.0)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def but_filter_high(x: np.ndarray, order: int, cutoff_hz: float, fs_hz: float) -> np.ndarray:
    wn = cutoff_hz / (fs_hz / 2.0)
    b, a = butter(order, wn, btype="high")
    return filtfilt(b, a, x)


def movmedian_omitnan(x: np.ndarray, k: int) -> np.ndarray:
    """Nan-aware rolling median approximating movmedian(...,'omitnan')."""
    x = np.asarray(x, dtype=float)
    k = int(k)
    if k <= 1:
        return x.copy()
    pad = k // 2
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - pad)
        hi = min(len(x), i + pad + 1)
        out[i] = np.nanmedian(x[lo:hi])
    return out

def normalize_range(x: np.ndarray) -> np.ndarray:
    """normalize(x,'range') with NaN safety."""
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn)


def expand_dict_columns(df: pd.DataFrame,
                        dict_columns,
                        flatten_2d: bool = False,
                        sep: str = "_") -> pd.DataFrame:
    """
    For each column name in `dict_columns`, assume cells are either:
      - dicts with keys like 'a','w','e','r', each value being a 1D or 2D array
      - or None/NaN.

    Creates new columns <col><sep><key> with:
      - 1D arrays -> Python list
      - 2D arrays -> list-of-lists (or flattened list if flatten_2d=True)

    Drops the original dict column.
    """
    df = df.copy()

    for col in dict_columns:
        s = df[col]  # this is a Series

        # collect all subkeys actually present
        keys = set()
        for v in s:
            if isinstance(v, dict):
                keys.update(v.keys())

        for subkey in sorted(keys):
            new_col = f"{col}{sep}{subkey}"

            def transform(cell, sk=subkey):
                if not isinstance(cell, dict):
                    return None
                if sk not in cell:
                    return None

                arr = np.asarray(cell[sk])

                if arr.ndim == 1:
                    return arr.tolist()              # list
                elif arr.ndim == 2:
                    return arr.ravel().tolist() if flatten_2d else arr.tolist()
                else:
                    # fallback: flatten higher dims
                    return arr.reshape(-1).tolist()

            df[new_col] = s.apply(transform)

        # drop original dict column
        df = df.drop(columns=[col])

    return df


def calc_spikewidth_from_traces(
    spike_traces,
    plot=False,
    save_dir=None,
    animal_id=None,
    cell_id=None,
    samples_per_ms=25,
    interp_step_ms=0.0001,
    interp_window_ms=2.0,
):
    """
    Compute spike half-width and peak-to-trough time from spike waveforms.

    Parameters
    ----------
    spike_traces : np.ndarray (T, N)
        Columns are individual spike waveforms (same as MATLAB: rows=samples, cols=spikes).
    plot : bool
        If True, saves a PDF plot (requires save_dir + animal_id + cell_id or will generate a name).
    save_dir : str or Path or None
        Directory to write the PDF plot into (created if it doesn't exist).
    animal_id : str or int or None
        Used in output filename when plotting.
    cell_id : str or int or None
        Used in output filename when plotting.
    samples_per_ms : float
        Number of samples per millisecond in the ORIGINAL waveform (MATLAB used 25).
        Equivalent to sampling rate = samples_per_ms * 1000 samples/second.
    interp_step_ms : float
        Interpolation time step in milliseconds (MATLAB used 0.0001 ms).
    interp_window_ms : float
        Interpolation window in milliseconds (MATLAB used 0..2 ms).

    Returns
    -------
    half_width : float
        Half-width of the spike in milliseconds.
    peak2trough : float
        Time from peak to trough (ms), measured on the interpolated waveform.
    waveform : np.ndarray (T,)
        Mean waveform (original sampling), baseline-aligned.
    peakidx : int
        Index of the peak on the INTERPOLATED waveform.
    trough_idx : int
        Index of the trough (after the peak) on the INTERPOLATED waveform.
    interp_time : np.ndarray
        Interpolated time vector (ms).
    interp_wave : np.ndarray
        Interpolated waveform.
    """

    spike_traces = np.asarray(spike_traces)
    if spike_traces.ndim != 2:
        raise ValueError("spike_traces must be 2D with shape (T, N).")

    # ---- inline linearizeSpike() ----
    # subtract each spike's first sample so every waveform starts at 0
    # (broadcast subtract along columns)
    first_samples = spike_traces[0, :]
    linwaves = spike_traces - first_samples  # (T, N)

    # mean across spikes -> (T,)
    meanwaves = linwaves.mean(axis=1)
    waveform = meanwaves.copy()

    # Build original time vector in ms (MATLAB: linspace(0, len/25, len))
    # samples_per_ms=25  -> sample rate = 25 kHz
    T = len(waveform)
    total_ms = T / samples_per_ms
    time_ms = np.linspace(0.0, total_ms, T)

    # Interpolate to high-res grid (MATLAB: 0:0.0001:2 ms)
    # Note: we clip to interp_window_ms just like MATLAB fixed 0..2 ms
    interp_time = np.arange(0.0, interp_window_ms + 1e-12, interp_step_ms)
    # Guard: if original time is shorter than interpolation window, cap the right bound
    right_bound = min(time_ms[-1], interp_window_ms)
    # Interpolate safely only over available range; values beyond are held at edges
    interp_wave = np.interp(interp_time, time_ms, waveform)

    # Find peak on interpolated waveform
    peakidx = int(np.argmax(interp_wave))
    peak = float(interp_wave[peakidx])

    # Half-width: first and last time where wave >= peak/2
    half_amp = peak / 2.0
    above = np.flatnonzero(interp_wave >= half_amp)
    if len(above) == 0:
        # Degenerate case: cannot compute half-width
        half_width = np.nan
    else:
        half_width = float(interp_time[above[-1]] - interp_time[above[0]])

    # Trough AFTER the peak
    if peakidx < len(interp_wave) - 1:
        post_peak = interp_wave[peakidx + 1:]
        trough_local_idx = int(np.argmin(post_peak))
        trough_idx = peakidx + 1 + trough_local_idx
        trough_val = float(interp_wave[trough_idx])
    else:
        trough_idx = peakidx
        trough_val = float(interp_wave[trough_idx])

    # Peak-to-trough in ms: index difference * interp_step_ms
    peak2trough = abs(trough_idx - peakidx) * float(interp_step_ms)

    # For plotting, the MATLAB code also computed an index 'ind' where wave crosses half peak on rising side.
    # We'll approximate the rising crossing index:
    if peakidx > 0:
        rising = interp_wave[: peakidx + 1]
        # nearest index where crossing occurs
        rising_cross = np.argmin(np.abs(rising - half_amp))
        ind = rising_cross
    else:
        ind = 0

    # ---- Plot (optional) ----
    if plot:
        # Figure and styling
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(interp_time, interp_wave, linewidth=1)

        # Half-width marker: horizontal segment at half_amp from ind to ind+half_width
        x0 = interp_time[ind]
        x1 = x0 + half_width
        ax.plot([x0, x1], [half_amp, half_amp],
                marker='|', linewidth=1, color='r')

        # Peak-to-trough marker: horizontal segment above the peak (cosmetic)
        y_top = peak + 0.01  # same cosmetic offset as MATLAB
        ax.plot(
            [interp_time[peakidx], interp_time[trough_idx]],
            [y_top, y_top],
            marker='|',
            linewidth=1,
        )

        # axes labels/limits/title
        ymin = float(interp_wave.min())
        ax.set_ylim([ymin + 0.25 * ymin, peak + 0.075 * peak]
                    if ymin < 0 else [0, peak * 1.075])
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Potential [mV]")

        hw_str = f"Halfwidth = {half_width:.2f} ms"
        p2t_str = f"Peak2Trough = {peak2trough:.2f} ms"
        ax.legend([hw_str, p2t_str], loc="upper right", fontsize=10)

        if save_dir is None:
            save_dir = Path.cwd() / "individual_waveshapes"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if animal_id is None:
            animal_id = "Animal"
        if cell_id is None:
            cell_id = "Cell"
        filename = f"{animal_id}_Id{cell_id}_waveshape.pdf"
        out_path = save_dir / filename
        fig.suptitle(f"Cell ID {cell_id}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    return (
        float(half_width),
        float(peak2trough),
        waveform,           # original-mean waveform (ms scale: time_ms)
        int(peakidx),
        int(trough_idx),
        interp_time,
        interp_wave,
    )


def angular_derivative_from_angles(angles, interval, speed_smoothing):
    angles_rad = np.deg2rad(angles)
    d_angles = np.diff(angles_rad)
    d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    d_angles = np.concatenate([d_angles, [d_angles[-1]]])

    def smooth_moving_average(x, span):
        kernel = np.ones(span)/span
        return np.convolve(x, kernel, mode='same')

    d_angles = smooth_moving_average(d_angles, speed_smoothing)

    angular_velocity = d_angles / interval
    angular_speed = np.abs(d_angles) / interval

    return angular_velocity, angular_speed


def load_mat_any(path: Union[str, Path], *, top_var: Optional[str] = None) -> Any:
    """
    Load MATLAB .mat files safely:
      - v7.3 (HDF5): uses h5py and converts structs/cells/strings
      - pre-v7.3: uses scipy.io.loadmat

    Args:
        path: path to .mat
        top_var: if provided, returns only that top-level variable (e.g. "pupil_out")

    Returns:
        A Python object (usually dict) with numpy arrays, lists, and strings.
    """
    path = Path(path)

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            out = _h5_to_py(f, f)
            # MATLAB v7.3 files often have '#refs#' plus real variables
            if isinstance(out, dict) and "#refs#" in out:
                out.pop("#refs#", None)
            return out[top_var] if (top_var is not None) else out

    # Non-v7.3:
    data = scipy.io.loadmat(
        path,
        struct_as_record=False,
        squeeze_me=False,
        simplify_cells=True,
    )
    # scipy adds these:
    data.pop("__header__", None)
    data.pop("__version__", None)
    data.pop("__globals__", None)
    return data[top_var] if (top_var is not None) else data


def _h5_to_py(obj: Any, root: h5py.File) -> Any:
    """
    Recursively convert HDF5 objects into Python types,
    with MATLAB-specific handling for:
      - char arrays (uint16/uint8) -> str
      - object references -> dereference
      - cell arrays (datasets of refs) -> list
      - structs (groups) -> dict
    """
    if isinstance(obj, h5py.Group):
        d: Dict[str, Any] = {}
        for k in obj.keys():
            d[k] = _h5_to_py(obj[k], root)
        return d

    if isinstance(obj, h5py.Dataset):
        # MATLAB cells/struct fields sometimes stored as refs
        if obj.dtype == object or h5py.check_dtype(ref=obj.dtype) is not None:
            data = obj[()]
            return _convert_ref_array(data, root)

        arr = obj[()]

        # Try to decode MATLAB char arrays stored as uint16/uint8
        maybe_str = _maybe_decode_matlab_char(arr)
        if maybe_str is not None:
            return maybe_str

        # Normal numeric array
        return np.array(arr)

    # Fallback (shouldn't happen often)
    return obj


def _convert_ref_array(x: Any, root: h5py.File) -> Any:
    """
    Convert an array/scalar of HDF5 references into Python objects.
    This covers MATLAB cell arrays (arrays of refs) and nested structs.
    """
    # Scalar reference
    if isinstance(x, h5py.Reference):
        if not x:
            return None
        return _h5_to_py(root[x], root)

    # Numpy array of references
    x = np.array(x)
    if x.dtype == object:
        # Could be nested python objects already; recurse elementwise
        return [[_convert_ref_array(elem, root) for elem in row] for row in x]

    # h5py sometimes yields dtype=object-like but not literally object; handle refs elementwise
    if x.ndim == 0:
        return _convert_ref_array(x.item(), root)

    # Convert to list while preserving shape
    def rec(idx_prefix: tuple) -> Any:
        if len(idx_prefix) == x.ndim:
            return _convert_ref_array(x[idx_prefix], root)
        return [rec(idx_prefix + (i,)) for i in range(x.shape[len(idx_prefix)])]

    return rec(tuple())


def _maybe_decode_matlab_char(arr: Any) -> Optional[str]:
    """
    Heuristic: MATLAB stores strings as 2D char arrays (uint16 or uint8),
    often shaped (N,1) or (1,N) or (N,M).
    """
    a = np.array(arr)

    if a.dtype not in (np.uint16, np.uint8):
        return None

    # If it's not at least 2D, probably not a MATLAB char matrix
    if a.ndim < 2:
        return None

    # Decode using MATLAB column-major order
    flat = a.flatten(order="F")

    # Remove trailing zeros (MATLAB padding)
    flat = flat[flat != 0]

    # Guard: if values don't look like text, skip
    if flat.size == 0:
        return ""

    # Most ASCII/Unicode text is in a sane range; if it's crazy, it's probably not a string
    if np.any(flat > 0x10FFFF):
        return None

    try:
        return "".join(chr(int(c)) for c in flat)
    except Exception:
        return None
# ---------- small helpers ----------


def wrap_to_pi(x):
    """Wrap radians to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def wrap_to_360_deg(x):
    """Wrap degrees to [0, 360)."""
    return np.mod(x, 360.0)


def smooth_moving_average(x, span):
    """Simple 1D moving average (non-circular)."""
    span = int(span)
    if span <= 1:
        return x.copy()
    k = np.ones(span) / span
    return np.convolve(x, k, mode="same")


def calculate_angular_speed_from_angles(angles_deg, dt, speed_smoothing):
    """
    Replicates your earlier pipeline:
      angles -> radians -> wrapped diff -> smooth derivative -> divide by dt
    Returns angular_speed (abs derivative / dt).
    """
    ang = np.deg2rad(angles_deg)
    dphi = np.diff(ang, prepend=ang[0])
    dphi = wrap_to_pi(dphi)                # like wrapToPi in MATLAB
    dphi_smooth = smooth_moving_average(dphi, speed_smoothing)
    angular_velocity = dphi_smooth / dt    # signed
    angular_speed = np.abs(dphi_smooth) / dt  # unsigned
    return angular_speed


def np_interp_strict(x_old, y_old, x_new):
    """
    1D linear interpolation (like interp1 / interp1q).
    Assumes x_old is increasing. NaN outside the range (MATLAB-style ‘q’).
    """
    y_new = np.interp(x_new, x_old, y_old, left=np.nan, right=np.nan)
    return y_new


def calculate_hd_occupancy(angles_deg, num_directions, dt, smooth_angular_win):
    directions = np.linspace(0, 360, num_directions + 1)

    ang = wrap_to_360_deg(angles_deg)  # your wrap maps 360 -> 0 which is fine

    # Use MATLAB histc behavior (same helper as in calculate_hd_rate)
    counts_full = histc_matlab(ang, directions)      # length 37
    counts_full[0] += counts_full[-1]                # merge 360 into 0
    counts = counts_full[:-1]                        # keep 36 bins

    hd_occupancy = counts.astype(float) * dt
    occ_smooth = _cyclic_smoothing(hd_occupancy, smooth_angular_win)
    return occ_smooth, hd_occupancy, directions


def calculate_hd_rate(video_times, spike_times, video_angles, smooth_angular_win, directions):
    """
    Python equivalent of MATLAB:
      SpikeAngle = interp1q(video_times, video_angles, spike_times)';
      AngleSpikes = histc(SpikeAngle, directions);
      AngleSpikes([1 end]) = AngleSpikes(1) + AngleSpikes(end);
      AngleSpikes = AngleSpikes(1:end-1);
      hdOccupancySpikesUnsmooth = AngleSpikes;
      hdOccupancySpikesSmooth   = CyclicSmoothing(AngleSpikes, smooth_angular_win);

    Parameters
    ----------
    video_times : (N,) array-like (monotonic)
    spike_times : (M,) array-like
    video_angles: (N,) array-like (degrees)
    smooth_angular_win : int | float
        Window length for moving average smoothing (circular).
    directions : (K,) array-like
        Bin edges in degrees, e.g. np.linspace(0, 360, 37)

    Returns
    -------
    hdOccupancySpikesSmooth   : (K-1,) ndarray
    hdOccupancySpikesUnsmooth : (K-1,) ndarray
    """
    # 1) Spike angles via linear interpolation (like interp1q)

    spike_angle = np.interp(spike_times, video_times, video_angles)

    AngleSpikes = histc_matlab(spike_angle, directions)

    AngleSpikes[0] += AngleSpikes[-1]

    AngleSpikes = AngleSpikes[:-1]

    hd_unsmooth = AngleSpikes.astype(float)

    hd_smooth = _cyclic_smoothing(hd_unsmooth, smooth_angular_win)

    return hd_smooth, hd_unsmooth

def histc_matlab(values, edges):
    """
    Exact MATLAB histc behavior.
    edges: array of bin edges, length K
    Returns: counts of length K
    """
    values = np.asarray(values)
    edges = np.asarray(edges)

    # main bins: edges[i] ≤ x < edges[i+1]
    counts = np.zeros(len(edges), dtype=int)

    # np.digitize, right=False gives bins based on x < edges[i]
    idx = np.digitize(values, edges, right=False)

    # digitize returns 1..len(edges)
    # values equal to edges[-1] become idx == len(edges)
    for i in idx:
        if 1 <= i <= len(edges):
            counts[i-1] += 1

    return counts

def _cyclic_smoothing(x, half_window):
    """
    Match MATLAB CyclicSmoothing(X, HalfWindow)
    WindowLength = 2*HalfWindow + 1
    """
    x = np.asarray(x, dtype=float)
    half_window = int(half_window)

    if half_window <= 0:
        return x.copy()

    win_len = 2 * half_window + 1
    kernel = np.ones(win_len, dtype=float) / win_len

    # cyclic extension exactly like MATLAB:
    x_ext = np.concatenate([x[-half_window:], x, x[:half_window]])

    y = np.convolve(x_ext, kernel, mode="full")

    # MATLAB crop: LongSmoothX(1 + 2*HalfWindow : end - 2*HalfWindow)
    start = 2 * half_window
    end = len(y) - 2 * half_window
    return y[start:end]


def average_firing_rate(spike_times):
    """Simple average firing rate in Hz over entire session duration."""
    st = np.asarray(spike_times)
    if st.size < 2:
        return np.nan
    duration = st.max() - st.min()
    if duration <= 0:
        return np.nan
    return st.size / duration


def plot_polar_with_theta_direction(theta_edges_rad, values):
    """
    Make a polar plot from bin edges (theta_edges_rad) and values per bin center.
    We’ll plot as a closed line.
    """
    # Convert edges to centers
    centers = 0.5 * (theta_edges_rad[:-1] + theta_edges_rad[1:])
    # ensure closed curve
    centers_closed = np.concatenate([centers, centers[:1]])
    values_closed = np.concatenate([values, values[:1]])

    ax = plt.subplot(111, projection='polar')
    ax.plot(centers_closed, values_closed, linewidth=2)
    r = np.nanmax(values)
    ax.set_rmax(r if np.isfinite(r) and r > 0 else 1.0)
    return np.nanmax(values_closed)

# ---------- main translation ----------


def plot_polar_graph(
    spike_times,
    video_times,
    video_angles,
    *,
    isplot_polarplot=True,
    speed_threshold=0.0,
    speed_smoothing=15,
    smooth_angular_win=2,
    after_smooth_win=None,
    minimal_occupancy=0.02,
    num_directions=36
):
    """
    Python translation of your MATLAB plot_polar_graph function.

    Returns:
      (hdRateSmooth, hdOccupancySpikesUnsmooth, directions, spike_angle_mean,
       max_firing_direction, hdRateUnsmooth, hdOccupancySmooth)
    """
    # Default returns
    hdRateSmooth = []
    hdRateUnsmooth = []
    hdOccupancySpikesUnsmooth = []
    hdOccupancySmooth = []
    directions = []
    spike_angle_mean = np.nan
    max_firing_direction = np.nan

    spike_times = np.asarray(spike_times, dtype=float)
    video_times = np.asarray(video_times, dtype=float)
    video_angles = np.asarray(video_angles, dtype=float)

    # --- Preprocess: video angular speed
    video_interval = np.median(np.diff(video_times))
    video_angular_speed = calculate_angular_speed_from_angles(
        video_angles, dt=video_interval, speed_smoothing=speed_smoothing
    )

    # Spike angular speed via interpolation at spike times
    spike_angular_speed = np_interp_strict(
        video_times, video_angular_speed, spike_times)

    # Apply speed threshold
    mask_spk = spike_angular_speed >= speed_threshold
    mask_vid = video_angular_speed >= speed_threshold
    spike_times_threshed = spike_times[mask_spk]
    video_angles_threshed = video_angles[mask_vid]

    if spike_times_threshed.size == 0:
        warnings.warn(
            "plot_polar_graph(): No spike times after speed threshold")
        return (hdRateSmooth, hdOccupancySpikesUnsmooth, directions,
                spike_angle_mean, max_firing_direction, hdRateUnsmooth, hdOccupancySmooth)

    # Round spike times to nearest tracking step (like MATLAB)
    spike_times_thre_round = np.round(
        spike_times_threshed / video_interval) * video_interval

    # --- Occupancy (time per direction bin)
    hdOccupancySmooth, hdOccupancyUnsmooth, directions = calculate_hd_occupancy(
        video_angles_threshed, num_directions, video_interval, smooth_angular_win
    )

    # Minimal occupancy check
    hdOcc = hdOccupancySmooth.copy()
    hdOcc[hdOcc < minimal_occupancy] = np.nan
    hdOccupancySmooth = hdOcc  # NaNs where too little occupancy

    # --- Spike occupancy (spike counts per bin)
    hdOccupancySpikesSmooth, hdOccupancySpikesUnsmooth = calculate_hd_rate(
        video_times, spike_times_thre_round, video_angles, smooth_angular_win, directions
    )

    # --- Firing rates: spike counts / occupancy time
    with np.errstate(invalid="ignore", divide="ignore"):
        hdRateSmooth = np.divide(hdOccupancySpikesSmooth, hdOccupancySmooth)
        hdRateUnsmooth = np.divide(hdOccupancySpikesUnsmooth, hdOccupancySmooth)

    # Close the unsmoothed curve like MATLAB (append first bin)
    hdRateUnsmooth = np.concatenate([hdRateUnsmooth, hdRateUnsmooth[:1]])

    # --- Mean preferred direction using vector sum of smoothed rates
    DirectionCenters = 0.5 * (directions[:-1] + directions[1:])  # degrees
    # complex vector average across bins
    vec = np.nansum(hdRateSmooth * np.exp(1j *
                    np.deg2rad(DirectionCenters))) / num_directions
    vec = vec / np.nanmean(hdRateSmooth)
    spike_angle_mean = wrap_to_360_deg(np.rad2deg(np.angle(vec)))
    theta_pref_direction = np.deg2rad(spike_angle_mean)

    # --- Optional post-smoothing on hdRateSmooth (non-circular)
    if after_smooth_win is not None:
        hd_tmp = hdRateSmooth.copy()
        nan_mask = np.isnan(hd_tmp)
        hd_tmp[nan_mask] = 0.0
        hd_tmp_s = smooth_moving_average(hd_tmp, after_smooth_win)
        hd_tmp_s[nan_mask] = np.nan
        hdRateSmooth = hd_tmp_s

    # Close the smoothed curve by appending first bin
    hdRateSmooth = np.concatenate([hdRateSmooth, hdRateSmooth[:1]])

    # Max firing direction (by index on closed curve)
    idx_m = np.nanargmax(hdRateSmooth)
    # directions is edges; align to same length by taking edge at idx
    # For the appended last point, idx_m==len(directions) means wrap to first edge
    if idx_m >= len(directions):
        idx_m = 0
    max_firing_direction = directions[idx_m]
    theta_max = np.deg2rad(max_firing_direction)

    # --- Plotting (polar)
    if isplot_polarplot:
        r = plot_polar_with_theta_direction(
            np.deg2rad(directions), hdRateSmooth)

        # preferred direction (red)
        plt.polar([0, theta_pref_direction], [0, r], color="r", linewidth=1)
        # max direction (blue)
        plt.polar([0, theta_max], [0, r], color="b", linewidth=1)

        # simple whole-session average
        avg_fr = average_firing_rate(spike_times)
        # Rayleigh test not implemented here (requires extra package)
        pval = np.nan
        hdi = np.nan
        title = f"AvgFR={avg_fr:.3f} Hz, MaxFR={np.nanmax(hdRateSmooth):.3f} Hz, HDI={hdi if np.isfinite(
            hdi) else float('nan'):.3f} (p={pval if np.isfinite(pval) else float('nan'):.3f})"
        plt.title(title, fontsize=8, weight="normal")
        plt.show()

    return (hdRateSmooth,
            hdOccupancySpikesUnsmooth,
            directions,
            spike_angle_mean,
            max_firing_direction,
            hdRateUnsmooth,
            hdOccupancySmooth)


def circ_r(alpha, w=None, d=0.0):
    """
    Mean resultant length r for circular data.
    alpha : array of angles [radians]
    w     : optional weights (e.g., counts or rates)
    d     : optional bin spacing [radians] for binned data (bias correction)
    """
    alpha = np.asarray(alpha)
    if w is None:
        w = np.ones_like(alpha, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("alpha and w must have the same shape")

    # Weighted vector sum
    C = np.sum(w * np.cos(alpha))
    S = np.sum(w * np.sin(alpha))
    n = np.sum(w)

    R = np.hypot(C, S)           # sqrt(C^2 + S^2)
    r = R / n if n > 0 else np.nan

    # Bias correction for binned data (Berens toolbox)
    # multiply by c = (d/2) / sin(d/2), then cap at 1
    if d and d > 0:
        c = (d / 2.0) / np.sin(d / 2.0)
        r = min(1.0, c * r)

    return r


def circ_rtest(alpha, w=None, d=0.0):
    """
    Rayleigh test for non-uniformity of circular data.
    Returns pval, z, r — matching MATLAB circ_rtest.
    """
    alpha = np.asarray(alpha)
    if alpha.ndim != 1:
        alpha = alpha.ravel()

    if w is None:
        r = circ_r(alpha)
        n = len(alpha)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("alpha and w must have the same shape")
        r = circ_r(alpha, w, d)
        n = float(np.sum(w))

    # Rayleigh's R and z (Zar)
    R = n * r
    z = (R * R) / n if n > 0 else np.nan

    # p-value approximation used in the MATLAB toolbox
    # p = exp( sqrt(1 + 4n + 4(n^2 - R^2)) - (1 + 2n) )
    if n > 0:
        pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n * n - R * R)) - (1 + 2 * n))
    else:
        pval = np.nan

    return pval, z, r


def calc_hd_significance(processed_data, raw_data, params, n_shuffle = 1000):

    time_begin = 0

    video_times = raw_data['ephys_data']['ttl_times']
    spike_times = processed_data['spike_sorting_data']['spike_times']
    video_angles = processed_data['tracking_data']['angles']

    if 'time_end' not in params:
        time_end = video_times[-1]
    else:
        time_end = params['time_end']

    video_chosen = (video_times >= time_begin) & (video_times <= time_end)
    video_times_for_hdsig = video_times[video_chosen]
    video_angles_for_hdsig = video_angles[video_chosen]

    spike_times_chosen = (spike_times >= time_begin) & (spike_times <= time_end)
    spike_times_for_hdsig = spike_times[spike_times_chosen]

    # rezero the chosen video range for the shuffling procedure
    video_times_for_hdsig = video_times_for_hdsig - video_times_for_hdsig[0]
    spike_times_for_hdsig = spike_times_for_hdsig - spike_times_for_hdsig[0]

    shuffle_endtimestamp = video_times_for_hdsig[-1]
    rng = np.random.default_rng(0)
    random_shuffles_times = np.concatenate(([0.0], shuffle_endtimestamp * rng.random(n_shuffle)))

    # do the shuffling
    HDI_rand_vectors = np.zeros(len(random_shuffles_times), dtype=float)
    for idx, elem in enumerate(random_shuffles_times):
        spike_times_in = spike_times_for_hdsig + elem

        # check if spikes behind recording
        times_overflow = spike_times_in > shuffle_endtimestamp
        spike_times_in[times_overflow] = spike_times_in[times_overflow] - shuffle_endtimestamp

        # calculate the tuning firing rate curve for this shuffle
        hdRateSmooth, _, directions, _,_,_,_ = plot_polar_graph(
            spike_times_in,
            video_times_for_hdsig,
            video_angles_for_hdsig,
            isplot_polarplot=False,
            )
        _, _, hdi_smooth = circ_rtest(
            np.deg2rad(directions),
            hdRateSmooth,
            np.deg2rad(np.median(np.diff(directions))))
        
        HDI_rand_vectors[idx] = hdi_smooth

    # remove the observed _HDI
    observed_HDI, HDI_rand_vectors = HDI_rand_vectors[0], HDI_rand_vectors[1:]
    p_value = calc_p_value_from_distribution(HDI_rand_vectors, observed_HDI)

    return p_value, observed_HDI


def calc_p_value_from_distribution(distribution: np.ndarray, real_value: float) -> float:
    """
    MATLAB's calc_p_value_from_distribution:
      - sort dist, find closest to real_value, compute pctile = idx/len
      - return 1 - pctile   (upper-tail style, matching original)
    """
    x = np.sort(distribution)
    idx = np.argmin(np.abs(x - real_value))
    pctile = (idx + 0) / len(distribution)  # MATLAB uses 1-based; effect is negligible
    return 1.0 - pctile


def fixed_window_indices(pupil_times: np.ndarray, t0: float, mot_samples: int) -> np.ndarray | None:
    """
    Return exactly mot_samples indices centered on t0 by nearest-sample indexing.
    """
    # closest sample to trigger
    center = int(np.argmin(np.abs(pupil_times - t0)))

    half = mot_samples // 2
    start = center - half
    end = start + mot_samples  # exclusive

    if start < 0 or end > len(pupil_times):
        return None

    return np.arange(start, end, dtype=np.int64)

if __name__ == "__main__":
    batch_process()
