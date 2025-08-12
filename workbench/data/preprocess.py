from pathlib import Path
from workbench.videography.camera_process import track_platform
from workbench.data.hdf5_interface import loadmat, save_processed_data
import workbench.data.db_interaction
import pandas as pd


def process_exp(folderpath, *args):
    """
    input path to start processing of an experiment
    requires path to exported .mat file from .smrx
    entry is a sqlite row from the database. it provides
    the function with recording information and path to
    various data sources.
    inputting args tracking: true will search for a baseline
    video and extract the rotational data for the platform
    """
    datapath = sorted(Path(folderpath).glob('Data*.mat'))[0]
    # load raw data
    data = loadmat(datapath)
    # initialize the spikesorting
    sorted_spikes = data.export_processed_data()
    # retrieve spikesorted data from spikesorter
    spikesorted = sorted_spikes.exportData()

    if 'tracking' in args:
        # ToDo videopath needs to be computed first
        videopath = sorted(Path(folderpath).glob('FH*.avi'))[0]
        angles, nframes, x_values, y_values = track_platform(videopath)

    save_processed_data(folderpath, spikesorted, angles,
                        nframes, x_values, y_values)


def process_behavioral():
    return


def batch_process():
    conn, _ = workbench.data.db_interaction.connectDb(
        r"../../data/Recordings.db")
    workbench.data.db_interaction.createFolderStructure(conn)

    query = """
            SELECT Folderpath FROM Recordings
            WHERE exp_type == 'juxta' AND use = 1
            ORDER BY Cell_Id ASC;
            """
    db = pd.read_sql(query, conn)
    for i in db['Folderpath'].to_list():
        target_dir = Path(i)
        h5_target = list(target_dir.glob("exp_data.h5"))
        if h5_target:
            if 'baseline' in i.lower():
                process_exp(i, 'tracking')


if __name__ == "__main__":
    print('running processing as main')
