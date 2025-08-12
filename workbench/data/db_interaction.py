import sqlite3
import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Union


def connectDb(dbpath: Union[str, Path]):
    """
    connects to the database and return the connection and the cursor for interactivity
    WARNING: This Method requires manual closure of the database
    """
    if isinstance(dbpath, str):
        dbpath = Path(dbpath)
    conn = sqlite3.connect(dbpath)
    cursor = conn.cursor()
    return conn, cursor


def createFolderStructure(conn: sqlite3.Connection,
                          tablename="Recordings",
                          animal_table="Animals"):
    query = f"""
        SELECT {tablename}.*, {animal_table}.Datafolder
        FROM {tablename} LEFT JOIN {animal_table}
        ON {tablename}.Animal_Id = {animal_table}.Animal_Id
        WHERE Folders_generated IS 0
        ORDER BY Cell_Id ASC;
        """

    df = pd.read_sql_query(query, conn)
    if df.emtpy():
        print("=================All Folders already exist=================")
        conn.close()
        return

    processed_keys = set()  # (Animal_Id, Cell_Id)

    # Group by animal → cell
    for animal_id, animal_df in df.groupby("Animal_Id"):
        raw_data_path = animal_df["Raw_Data_Path"].iloc[0]
        if not raw_data_path or not isinstance(raw_data_path, str):
            print(f"[WARN] Missing Raw_Data_Path for Animal_Id={
                  animal_id}; skipping.")
            continue

        raw_data_path = Path(raw_data_path)
        analysis_folder = raw_data_path / "analysis"
        analysis_folder.mkdir(parents=True, exist_ok=True)

        # Find the first folder containing "data"
        data_folder = next(
            (raw_data_path / item for item in os.listdir(raw_data_path)
             if "data" in item.lower()),
            None
        )
        if data_folder is None or not data_folder.exists():
            print(f"[WARN] No 'data' folder found under {
                  raw_data_path}; skipping Animal_Id={animal_id}.")
            continue

        for cell_id, cell_df in animal_df.groupby("Cell_Id"):
            cell_folder = analysis_folder / f"Data{cell_id}"
            cell_folder.mkdir(parents=True, exist_ok=True)

            # Iterate rows for conditions/videos
            for row in cell_df.itertuples(index=False):
                # Adjust these names if your columns differ:
                condition = getattr(row, "Condition")
                rotation_vid = getattr(row, "Rotation_Video")
                pupil_vid = getattr(row, "Pupil_Video")

                condition_folder = cell_folder / str(condition)
                condition_folder.mkdir(parents=True, exist_ok=True)
                print(f"{condition_folder} created")

                # Baseline video copy (matches 0001 style)
                cond_lower = str(condition).lower()
                try:
                    if ("baseline" in cond_lower or "lowlumbaseline" in cond_lower) and rotation_vid != "-":
                        n = int(rotation_vid)
                        match = next((f for f in os.listdir(
                            data_folder) if f"{n:04d}" in f), None)
                        if match:
                            shutil.copy(data_folder / match, condition_folder)
                        else:
                            print(f"[INFO] No baseline video matched {
                                  n:04d} in {data_folder}")
                except (ValueError, TypeError):
                    print(f"[WARN] Invalid Rotation_Video value: {
                          rotation_vid}")

                # Pupil video copy
                try:
                    if pupil_vid != "-":
                        n = int(pupil_vid)
                        pupil_src = data_folder / "Pupil"
                        if pupil_src.exists():
                            match = next((f for f in os.listdir(
                                pupil_src) if f"{n:04d}" in f), None)
                            if match:
                                pv = condition_folder / "PupilVid"
                                pv.mkdir(exist_ok=True)
                                shutil.copy(pupil_src / match, pv)
                            else:
                                print(f"[INFO] No pupil video matched {
                                      n:04d} in {pupil_src}")
                        else:
                            print(f"[WARN] Missing Pupil subfolder under {
                                  data_folder}")
                except (ValueError, TypeError):
                    print(f"[WARN] Invalid Pupil_Video value: {pupil_vid}")

                # Mark composite key as processed
                processed_keys.add((animal_id, cell_id))

    # Bulk update using composite key
    if processed_keys:
        update_sql = f"""
        UPDATE {tablename}
           SET Folders_generated = 1
         WHERE Animal_Id = ? AND Cell_Id = ?;
        """
        conn.executemany(update_sql, list(processed_keys))
        conn.commit()
        print(f"Updated {len(processed_keys)
                         } (Animal_Id, Cell_Id) pairs with Folders_generated = 1")

    print("------------Folder creation done------------")
    conn.close()


if __name__ == "__main__":
    conn, _ = connectDb(r"../../data/Recordings.db")
    createFolderStructure(conn)
