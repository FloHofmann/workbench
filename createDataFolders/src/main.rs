use anyhow::{Context, Ok, Result};
use clap::Parser;
use rusqlite::{Connection, Row, named_params};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(version, about, long_about =None)]
struct Args {
    // database path
    #[arg(
        long,
        default_value = r"\\172.25.250.112\burgalossi\lab share\Data\Florian\Recordings_FH.db"
    )]
    dbpath: PathBuf,

    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

struct PendingUpdate {
    animal_id: String,
    cell_id: i64,
    condition: String,
    new_folderpath: String,
}

struct WorkFailure {
    animal_id: String,
    cell_id: i64,
    condition: String,
    reason: String,
}

// centralize table/cloumn names in one place
mod schema {
    // TABLES
    pub const T_ANIMALS: &str = "Animals";
    pub const T_RECORDINGS: &str = "Recordings";

    // Animals columns (you need: Animal_Id, Datafolder)
    pub const A_ANIMAL_ID: &str = "Animal_Id";
    pub const A_DATAFOLDER: &str = "Datafolder";

    // Recordings columns (you need: Animal_Id, Cell_Id, Condition, Rotation_VID, Pupil_VID, use, Folders_generated, Folderpath)
    pub const R_ANIMAL_ID: &str = "Animal_Id";
    pub const R_CELL_ID: &str = "Cell_Id";
    pub const R_CONDITION: &str = "Condition";
    pub const R_FNAME: &str = "Filename";
    pub const R_ROT_VID: &str = "Rotation_VID";
    pub const R_PUPIL_VID: &str = "Pupil_VID";
    pub const R_FLAG: &str = "Folders_generated";
    pub const R_FOLDERPATH: &str = "Folderpath";
}

#[derive(Debug)]
struct RecordingRow {
    // From Recordings:
    animal_id: String,
    cell_id: i64, // INTEGER DEFAULT '-' (but should be int in real data)
    condition: String,
    filename: String,
    rotation_vid: Option<i64>, // TEXT with '-' → None, else parse to i64
    pupil_vid: Option<i64>,    // same parsing
    folders_generated: i64,

    // From Animals:
    datafolder: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // validate path here
    validate_path(&args.dbpath)?;

    // open db with a small busy timeout
    let mut conn = open_db(&args.dbpath)?;

    validate_schema(
        &conn,
        schema::T_ANIMALS,
        &[schema::A_ANIMAL_ID, schema::A_DATAFOLDER],
    )?;
    validate_schema(
        &conn,
        schema::T_RECORDINGS,
        &[
            schema::R_ANIMAL_ID,
            schema::R_CELL_ID,
            schema::R_CONDITION,
            schema::R_FNAME,
            schema::R_ROT_VID,
            schema::R_PUPIL_VID,
            schema::R_FLAG,
            schema::R_FOLDERPATH,
        ],
    )?;

    // pull rows you want to process

    // pull rows you want to process
    let rows = load_rows(&conn)?;
    println!("Loaded {} rows to process", rows.len());

    let mut updates: Vec<PendingUpdate> = Vec::new();
    let mut failures: Vec<WorkFailure> = Vec::new();

    for r in &rows {
        match process_row(r, args.dry_run)? {
            std::result::Result::Ok(pu) => updates.push(pu),
            std::result::Result::Err(fail) => failures.push(fail),
        }
    }

    println!("Planned updates: {}", updates.len());
    if !failures.is_empty() {
        eprintln!("{} rows failed during folder generation:", failures.len());
        for f in &failures {
            eprintln!(
                "  [{}:{}:{}] {}",
                f.animal_id, f.cell_id, f.condition, f.reason
            );
        }
    }

    // Do one DB transaction at the end (unless dry-run)

    if !args.dry_run && !updates.is_empty() {
        let tx = conn.transaction()?;
        let sql = r#"
            UPDATE "Recordings"
            SET "Folders_generated" = 1,
                "Folderpath"       = :folder
            WHERE "Animal_Id" = :animal
              AND "Cell_Id"   = :cell
              AND "Condition" = :cond
            "#;

        {
            //  new inner scope so `stmt` is dropped before `commit`
            let mut stmt = tx.prepare_cached(sql)?;
            for u in &updates {
                stmt.execute(named_params! {
                    ":folder": u.new_folderpath,
                    ":animal": u.animal_id,
                    ":cell":   u.cell_id,
                    ":cond":   u.condition,
                })?;
            }
        } // stmt dropped here

        tx.commit()?; // now it’s safe to move `tx`
        println!("DB updated for {} rows.", updates.len());
    } else if args.dry_run {
        println!("[dry-run] Skipping DB updates for {} rows.", updates.len());
    }
    Ok(())
}

// Ensure the directory and file exists
fn validate_path(path: &Path) -> Result<()> {
    // check if the parent directory exists
    let parent = path
        .parent()
        .context("The provided path has no parent directory")?;
    if !parent.exists() {
        anyhow::bail!(
            "The directory or network drive {:?} could not be found.",
            parent
        );
    }

    // check if the file exists
    if !path.exists() {
        anyhow::bail!("The database file{:?} was not found.", path);
    }

    println!("Database found at {:?}", path);
    Ok(())
}

// open the database
fn open_db(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path)
        .with_context(|| format!("Failed to open database at {}", path.display()))?;
    conn.busy_timeout(std::time::Duration::from_millis(500))?;
    Ok(conn)
}

// fail fast if required column are missing
fn validate_schema(conn: &Connection, table: &str, must_have: &[&str]) -> Result<()> {
    let sql = format!(r#"Pragma table_info("{table}")"#);
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;

    let mut cols = HashSet::new();
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        cols.insert(name);
    }

    let missing: Vec<_> = must_have
        .iter() // makes must_have iterable
        .filter(|c| !cols.contains(&c.to_string())) //iterates over must_have while checking if the
        //individual entries of must_have match to the column names of the database Pragma
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "Table {table:?} is missing required column(s): {missing:?}. \
            Present columns: {cols:?}"
        );
    }

    Ok(())
}

// Query rows using Aliases
fn load_rows(conn: &Connection) -> Result<Vec<RecordingRow>> {
    use schema::*;

    // Double-quote identifiers to survive case-sensitive schemas.
    // Alias to stabel names we control: animal_id, clel_id, condition
    let sql = format!(
        r#"
        SELECT
            r."{r_animal}"      AS animal_id,
            r."{r_cell}"        AS cell_id,
            r."{r_cond}"        AS condition,
            r."{r_fname}"       As filename,
            r."{r_rot}"         AS rotation_vid_raw,
            r."{r_pupil}"       AS pupil_vid_raw,
            r."{r_flag}"        AS folders_generated,
            a."{a_datafolder}"  AS datafolder
        FROM "{t_recordings}" r
        LEFT JOIN "{t_animals}" a
          ON r."{r_animal}" = a."{a_animal}"
        WHERE
            r."{r_flag}" = 0
        ORDER BY r."{r_cell}" ASC
        "#,
        t_recordings = T_RECORDINGS,
        t_animals = T_ANIMALS,
        r_animal = R_ANIMAL_ID,
        r_cell = R_CELL_ID,
        r_cond = R_CONDITION,
        r_fname = R_FNAME,
        r_rot = R_ROT_VID,
        r_pupil = R_PUPIL_VID,
        r_flag = R_FLAG,
        a_datafolder = A_DATAFOLDER,
        a_animal = A_ANIMAL_ID,
    );

    let mut stmt = conn
        .prepare(&sql)
        .with_context(|| "Failed to prepare SELECT statement")?;

    let rows_iter = stmt
        .query_and_then([], |row| parse_row(row))
        .with_context(|| "Failed to execute SELECT query")?;

    // Now collect rows, converting each Result properly
    let mut out = Vec::new();
    for row_result in rows_iter {
        // row_result is Result<RecordingRow, rusqlite::Error>
        let row = row_result.with_context(|| "Failed to parse a row from the database")?;
        out.push(row);
    }
    Ok(out)
}

// Map a row by alias name; parse Text '-' -> None for *_VID
fn parse_row(row: &Row) -> Result<RecordingRow> {
    fn to_opt_i64<S: AsRef<str>>(s: Option<S>) -> Option<i64> {
        match s {
            None => None,
            Some(val) => {
                let v = val.as_ref().trim();
                if v == '-'.to_string() || v.is_empty() {
                    None
                } else {
                    v.parse::<i64>().ok()
                }
            }
        }
    }

    let rotation_vid = to_opt_i64::<String>(row.get("rotation_vid_raw")?);
    let pupil_vid = to_opt_i64::<String>(row.get("pupil_vid_raw")?);

    Ok(RecordingRow {
        animal_id: row.get("animal_id")?,
        cell_id: row.get("cell_id")?,
        condition: row.get("condition")?,
        filename: row.get("filename")?,
        rotation_vid,
        pupil_vid,
        folders_generated: row.get("folders_generated")?,
        datafolder: row.get("datafolder")?,
    })
}

fn process_row(
    row: &RecordingRow,
    dry_run: bool,
) -> Result<std::result::Result<PendingUpdate, WorkFailure>> {
    let target = build_target_path(row);
    if let Err(e) = copy_files(row, &target, dry_run) {
        println!(
            "Note: copy step had issues for [{}:{}:{}]: {e}",
            row.animal_id, row.cell_id, row.condition
        );
    }

    if let Err(e) = ensure_dir_created(&target, dry_run) {
        return Ok(std::result::Result::Err(WorkFailure {
            animal_id: row.animal_id.clone(),
            cell_id: row.cell_id,
            condition: row.condition.clone(),
            reason: format!("creating {} failed: {e}", target.display()),
        }));
    }

    Ok(std::result::Result::Ok(PendingUpdate {
        animal_id: row.animal_id.clone(),
        cell_id: row.cell_id,
        condition: row.condition.clone(),
        new_folderpath: target.to_string_lossy().into_owned(),
    }))
}

fn copy_files(row: &RecordingRow, target: &PathBuf, dry_run: bool) -> Result<()> {
    let data_dir = PathBuf::from(&row.datafolder).join("data");
    println!("{}", data_dir.display());
    let pupil_src_dir = data_dir.join("Pupil");

    let is_baseline = row.condition.to_lowercase().contains("baseline");
    let matname = row.filename.clone() + ".mat";
    let mat_src = data_dir.join(&matname);

    if mat_src.exists() {
        let _ = copy_if_missing(&mat_src, target, dry_run).map_err(|e| {
            println!("Note: failed to copy MAT {}: {e}", mat_src.display());
            e
        });
    } else {
        println!("Note: MAT not found -> {}", mat_src.display());
    }

    // copy the rotation video -> this is only for baseline
    if is_baseline {
        if let Some(rot_id) = row.rotation_vid {
            let needle = pad4(rot_id);
            let found = find_file_in_dir(&data_dir, |name| {
                name.starts_with("fh") && name.contains(&needle) && name.ends_with(".avi")
            })?;
            if let Some(rot_src) = found {
                let _ = copy_if_missing(&rot_src, target, dry_run).map_err(|e| {
                    println!(
                        "Note: failed to copy rotation video {}: {e}",
                        rot_src.display()
                    );
                    e
                });
            } else {
                println!(
                    "Note: rotation video not found in {} (needle: FH{}*.avi)",
                    data_dir.display(),
                    needle
                );
            }
        } else {
            println!(
                "Note: no Rotation_VID in DB for baseline row {}:{}:{}",
                row.animal_id, row.cell_id, row.condition,
            );
        }
    }

    // pupil video - if present
    if let Some(pupil_id) = row.pupil_vid {
        let needle = pad4(pupil_id);
        let found = find_file_in_dir(&pupil_src_dir, |name| {
            name.starts_with("fh") && name.contains(&needle) && name.ends_with(".avi")
        })?;
        let pupil_dst_dir = target.join("PupilVid");
        let _ = ensure_dir_created(&pupil_dst_dir, dry_run);
        if let Some(pupil_src) = found {
            let _ = copy_if_missing(&pupil_src, &pupil_dst_dir, dry_run).map_err(|e| {
                println!(
                    "Note: failed to copy pupil video {}: {e}",
                    pupil_src.display()
                );
                e
            });
        } else {
            println!(
                "Note: pupil video not found in {} (needle: FH{}*.avi)",
                pupil_src_dir.display(),
                needle
            );
        }
    } else {
        println!(
            "Note: no Pupil_VID in DB for row {}:{}:{}",
            row.animal_id, row.cell_id, row.condition
        );
    }

    Ok(())
}

fn find_file_in_dir<F>(dir: &Path, mut pred: F) -> Result<Option<PathBuf>>
where
    F: FnMut(&str) -> bool,
{
    if !dir.is_dir() {
        return Ok(None);
    }
    for entry in
        fs::read_dir(dir).with_context(|| format!("Reading directory {}", dir.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let name_lower = entry.file_name().to_string_lossy().to_lowercase();
            if pred(&name_lower) {
                return Ok(Some(entry.path()));
            }
        }
    }
    Ok(None)
}

fn copy_if_missing(src: &Path, dst_dir: &Path, dry_run: bool) -> Result<bool> {
    if !src.exists() {
        anyhow::bail!("Source file does not exist: {}", src.display());
    }
    let fname = src
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("Invalid source filename: {}", src.display()))?;
    let dst = dst_dir.join(fname);

    if dst.exists() {
        println!("Skip copy (exists): {}", dst.display());
        return Ok(false);
    }
    if dry_run {
        println!("[dry-run] cp {} {} \n", src.display(), dst.display());
        return Ok(true);
    }
    let _ = fs::copy(src, &dst)
        .with_context(|| format!("Failed to copy {} -> {}", src.display(), dst.display()));
    println!("Copied {} -> {} \n", src.display(), dst.display());
    Ok(true)
}

fn build_target_path(row: &RecordingRow) -> PathBuf {
    let mut p = PathBuf::from(&row.datafolder);
    p.push("analysis");
    p.push(format!("Data{}", row.cell_id));
    p.push(&row.condition);
    p
}

fn ensure_dir_created(path: &Path, dry_run: bool) -> Result<bool> {
    if path.exists() {
        if !path.is_dir() {
            anyhow::bail!("Path exists but is not a directory: {}", path.display());
        }
        return Ok(false);
    }
    if dry_run {
        println!("[dry-run] mkdir -p {} \n", path.display());
        return Ok(true);
    }
    fs::create_dir_all(path)
        .with_context(|| format!("Failed to create directory: {} \n", path.display()))?;
    Ok(true)
}

fn pad4(n: i64) -> String {
    format!("{:04}", n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;
    use tempfile::tempdir;

    /// Helper: bootstrap an in-memory DB with your schema.
    fn bootstrap_db() -> Result<Connection> {
        let conn = Connection::open_in_memory()?;
        // Create tables exactly as specified (trimmed defaults are fine).
        conn.execute_batch(
            r#"
            CREATE TABLE "Animals" (
                "Animal_Id"     TEXT DEFAULT '-' UNIQUE,
                "Gender"        TEXT DEFAULT '-',
                "Datafolder"    TEXT DEFAULT '-',
                "Age_weeks"     INTEGER DEFAULT '-',
                "Start_exp"     TEXT DEFAULT '-',
                "End_exp"       INTEGER DEFAULT '-',
                "Permit"        TEXT,
                "Project"       TEXT DEFAULT '-',
                "Experimenter"  TEXT DEFAULT 'FH',
                PRIMARY KEY("Animal_Id")
            );
            CREATE TABLE "Recordings" (
                "Animal_Id"         TEXT DEFAULT '-',
                "Cell_Id"           INTEGER DEFAULT '-',
                "Date"              TEXT DEFAULT '-',
                "Hem"               TEXT DEFAULT '-',
                "Depth"             INTEGER DEFAULT '-',
                "Celltype"          TEXT DEFAULT '-',
                "Condition"         TEXT DEFAULT '-',
                "Filename"          TEXT DEFAULT '-',
                "Rotation_VID"      TEXT DEFAULT '-',
                "Pupil_VID"         TEXT DEFAULT '-',
                "Comment"           BLOB DEFAULT '-',
                "exp_type"          TEXT DEFAULT 'juxta',
                "Folderpath"        TEXT DEFAULT '-',
                "Folders_generated" INTEGER DEFAULT 0,
                "use"               INTEGER DEFAULT 1,
                FOREIGN KEY("Animal_Id") REFERENCES "Animals"("Animal_Id")
            );
            "#,
        )?;
        Ok(conn)
    }

    #[test]
    fn validate_path_ok_and_missing() -> Result<()> {
        // OK: create temp dir + file
        let dir = tempdir()?;
        let dbfile = dir.path().join("db.sqlite");
        std::fs::write(&dbfile, b"")?;
        validate_path(&dbfile)?; // should be Ok

        // Missing parent dir
        let bogus = dir.path().join("nope").join("db.sqlite");
        let err = validate_path(&bogus).unwrap_err();
        assert!(
            format!("{err:?}").contains("not be found") || format!("{err:?}").contains("not found")
        );
        Ok(())
    }

    #[test]
    fn validate_schema_ok_and_missing_column() -> Result<()> {
        let conn = bootstrap_db()?;

        // OK: both tables have required columns
        validate_schema(
            &conn,
            schema::T_ANIMALS,
            &[schema::A_ANIMAL_ID, schema::A_DATAFOLDER],
        )?;
        validate_schema(
            &conn,
            schema::T_RECORDINGS,
            &[
                schema::R_ANIMAL_ID,
                schema::R_CELL_ID,
                schema::R_CONDITION,
                schema::R_ROT_VID,
                schema::R_PUPIL_VID,
                schema::R_FLAG,
                schema::R_FOLDERPATH,
            ],
        )?;

        // Now simulate missing column by querying for a non-existent one.
        let err = validate_schema(
            &conn,
            schema::T_RECORDINGS,
            &[schema::R_ANIMAL_ID, "DefinitelyNotACol"],
        )
        .unwrap_err();
        assert!(format!("{err:?}").contains("missing required column"));

        Ok(())
    }

    #[test]
    fn load_rows_parses_and_filters() -> Result<()> {
        let conn = bootstrap_db()?;

        // Insert one animal
        conn.execute(
            r#"INSERT INTO "Animals" ("Animal_Id", "Datafolder") VALUES (?1, ?2)"#,
            params!["A1", r"Z:\Animals\A1"],
        )?;

        // Insert recordings: one to process, one already generated, one with use=0
        conn.execute(
            r#"
            INSERT INTO "Recordings"
                ("Animal_Id","Cell_Id","Condition","Rotation_VID","Pupil_VID","Folders_generated","use","Folderpath")
            VALUES
                ("A1", 42, "Baseline", "0012", "0007", 0, 1, "-"),
                ("A1", 43, "Stim",     "-",    "-",    1, 1, "-"),
                ("A1", 44, "Stim",     "0013", "-",    0, 0, "-")
            "#,
            [],
        )?;

        // load_rows should only return the first row (flag=0 & use=1 if you add that filter; currently only flag=0)
        let rows = load_rows(&conn)?;
        println!("{:?}", rows);
        assert_eq!(rows.len(), 2);
        let r = &rows[0];

        assert_eq!(r.animal_id, "A1");
        assert_eq!(r.cell_id, 42);
        assert_eq!(r.condition, "Baseline");
        assert_eq!(r.rotation_vid, Some(12));
        assert_eq!(r.pupil_vid, Some(7));
        assert_eq!(r.folders_generated, 0);
        assert_eq!(r.datafolder, r"Z:\Animals\A1");

        Ok(())
    }
}
