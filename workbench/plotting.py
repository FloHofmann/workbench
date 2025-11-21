import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401 (registers .hvplot)
import holoviews as hv

hv.extension("bokeh")
pn.extension()


def serve_cell_dashboard(
    row_idx,
    comb_table,
    new_rate,
    DIRECTIONS,
    speakers=("a", "w", "e", "r"),
    speaker_position=None,
    bins_plot=None,
    time_bin=0.002,
    tmin=-100.0,
    tmax=500.0,
):
    """
    Start a Panel/hvPlot app on localhost showing the mosaic for one cell.
    """
    if speaker_position is None:
        speaker_position = {"a": 93, "w": 178, "e": 272, "r": 356}

    # If comb_table is Polars, get pandas view
    if not isinstance(comb_table, pd.DataFrame):
        comb_pd = comb_table.to_pandas()
    else:
        comb_pd = comb_table

    # -----------------------------
    # Build per-speaker plots
    # -----------------------------
    raster_plots = []
    psth_plots = []
    whisk_plots = []
    raster_times = comb_pd['RasterTimes'][row_idx]
    raster_rows = comb_pd['RasterRows'][row_idx]
    whisk_avg = comb_pd['whisk_avg'][row_idx]
    HDRateSmooth = comb_pd['hdRateSmooth'][row_idx]

    for enum, key in enumerate(speakers):
        # ---------- RASTER ----------
        rt = np.array(raster_times[key])
        rr = np.array(raster_rows[key])

        if rt.size > 0:
            df_raster = pd.DataFrame(
                {"time": rt, "trial": rr}
            )
            r_plot = df_raster.hvplot.scatter(
                x="time",
                y="trial",
                size=5,
                alpha=0.6,
                xlabel="Time (ms)",
                ylabel=f"{speaker_position[key]}°",
                title=f"Raster {key}",
                xlim=(tmin, tmax),
            )
        else:
            r_plot = hv.Curve([]).opts(
                xlabel="Time (ms)",
                ylabel=f"{speaker_position[key]}°",
                title=f"Raster {key}",
                xlim=(tmin, tmax),
            )
        raster_plots.append(r_plot)

        # ---------- PSTH ----------
        if bins_plot is not None:
            df_psth = pd.DataFrame(
                {
                    "time": bins_plot,
                    "rate": new_rate[row_idx, :, enum],
                }
            )
            p_plot = df_psth.hvplot.step(
                x="time",
                y="rate",
                xlabel="Time (ms)",
                ylabel="FR [Hz]",
                title=f"PSTH {key}",
                xlim=(tmin, tmax),
            )
        else:
            p_plot = hv.Curve([]).opts(
                xlabel="Time (ms)",
                ylabel="FR [Hz]",
                title=f"PSTH {key}",
                xlim=(tmin, tmax),
            )
        psth_plots.append(p_plot)

        # ---------- WHISK LINE ----------
        wa = np.array(whisk_avg[key])
        # assume wa is 1D; time axis from trigger_time or something similar
        # here we fake a time axis same as bins_plot (adjust as needed)
        if bins_plot is not None and wa.size == bins_plot.size:
            df_whisk = pd.DataFrame({"time": bins_plot, "whisk": wa})
            w_plot = df_whisk.hvplot.line(
                x="time",
                y="whisk",
                xlabel="Time (ms)",
                ylabel="Whisk",
                title=f"Whisk {key}",
                xlim=(tmin, tmax),
            )
        else:
            w_plot = hv.Curve([]).opts(
                xlabel="Time (ms)",
                ylabel="Whisk",
                title=f"Whisk {key}",
                xlim=(tmin, tmax),
            )
        whisk_plots.append(w_plot)

    # -----------------------------
    # POLAR tuning curve
    # -----------------------------
    angles_deg = list(speaker_position.values())
    angles_rad = np.deg2rad(angles_deg)
    hdr = np.array(HDRateSmooth)

    df_polar = pd.DataFrame(
        {
            "angle_deg": DIRECTIONS,
            "angle_rad": np.deg2rad(DIRECTIONS),
            "rate": hdr,
        }
    )

    polar_curve = df_polar.hvplot.line(
        x="angle_rad",
        y="rate",
        xlabel="angle",
        ylabel="rate",
        title="HD tuning",
    ).opts(
        frame_width=300,
        frame_height=300,
    )

    # Speaker markers on polar (approximate using overlay)
    marker_df = pd.DataFrame(
        {"angle_rad": angles_rad, "rate": np.ones_like(angles_rad) * hdr.max()}
    )
    polar_markers = marker_df.hvplot.scatter(
        x="angle_rad", y="rate", size=10, color="red"
    )
    polar_plot = (polar_curve * polar_markers).opts(title="HD tuning")

    # -----------------------------
    # INFO panel
    # -----------------------------
    row = comb_pd.iloc[row_idx]
    infotext = (
        f"Animal:   {row['Animal_Id']}\n"
        f"Cell:     {row['Cell_Id']}\n"
        f"PD:       {row['HDAngle']:.2f}\n"
        f"PeakFR:   {row['HDpeakFR']:.2f}\n"
        f"MeanFR:   {row['HDFR']:.2f}\n"
        f"HDInx:    {row['HDInx']:.2f}\n"
        f"pValR:    {row['pValR']:.2f}\n"
        f"binsize:  {1000*time_bin:.2f} ms\n"
    )
    info_pane = pn.pane.Markdown(infotext, sizing_mode="stretch_both")

    # -----------------------------
    # Layout: approx. 4x4 mosaic
    # -----------------------------
    # Column 0: rasters
    col_raster = pn.Column(*raster_plots, sizing_mode="stretch_both")
    # Column 1: PSTHs
    col_psth = pn.Column(*psth_plots, sizing_mode="stretch_both")
    # Column 2: whisk lines
    col_whisk = pn.Column(*whisk_plots, sizing_mode="stretch_both")
    # Column 3: polar + info stacked
    col_polar_info = pn.Column(polar_plot, info_pane, sizing_mode="stretch_both")

    dashboard = pn.Row(col_raster, col_psth, col_whisk, col_polar_info)

    # Serve on localhost; this will block until you stop it
    pn.serve(dashboard, title=f"Cell {row['Animal_Id']} / {row['Cell_Id']}")
