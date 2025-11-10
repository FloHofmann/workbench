from pathlib import Path

from workbench.videography.camera_process import track_platform
from workbench.data.hdf5_interface.h5data import loadmat, save_processed_data
from workbench.data.db_interaction import createFolderStructure, connectDb
import workbench.data.db_interaction
import pandas as pd


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
            ch4_stims = process_behavioral(
                channels['4'], keyboard_channel['times'].flatten())
        if 'tones' in str(datapath):
            ch5_stims = process_behavioral(
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


def process_behavioral(channel: dict, keyboard_times):
    import numpy as np
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


if __name__ == "__main__":
    batch_process()
