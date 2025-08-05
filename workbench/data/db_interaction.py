import sqlite3
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
                          cur: sqlite3.Cursor,
                          tablename="Recordings",
                          animal_table="Animals"):
    query = """
        SELECT {tablename}.*, {animal_table}.Datafolder
        FROM {tablename} LEFT JOIN {animal_table}
        ON {tablename}.Animal_Id = {animal_table}.Animal_Id
        WHERE Folders_generated IS 0
        ORDER BY Cell_Id ASC;
        """.format(tablename, animal_table)

    df = pd.read_sql_query(query, conn)
    if df.emtpy():
        print("=================All Folders already exist=================")
        conn.close()
    else:
        animal_df = df[['Animal_Id', 'raw_data_path']].drop_duplicates(subset='Animal_Id')

        for row in animal_df.itertuples(index=False):
            animal_id = row.Animal_Id
            raw_path = row.raw_data_path


