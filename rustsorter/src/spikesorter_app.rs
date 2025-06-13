use eframe::{
    NativeOptions,
    egui::{self, CentralPanel, Color32, Layout},
};
use egui_plot::{self, Bar, BarChart, Line, Plot, PlotPoints};
use rand::prelude::*;
use std::f64;

#[derive(PartialEq)]
enum AppState {
    TraceView,
    PostThreshold,
}

pub struct SpikeSorterApp {
    trace_data: Vec<[f64; 2]>,
    sr: f64,
    threshold: Option<f64>,
    threshold_mode: bool,
    state: AppState,
    spike_traces: Vec<Vec<f64>>,
    intervals: Vec<Bar>,
    pca_2d: Vec<[f64; 2]>,
}

impl Default for SpikeSorterApp {
    /// Default implementation to test the system on artificial signal
    /// trace data are the y values from the voltage signal
    /// sr is the sampling rate of the signal. Is set to 25000 as default
    /// threshold is a placeholder for later sorting operations
    /// state is the state of the spikesorter window
    fn default() -> Self {
        let mut rng = rand::rng();
        let trace_data = (0..25000)
            .map(|i| {
                let x = i as f64 * 0.01;
                [x, (x * 10.0).sin() * (x * 0.5).cos()]
            })
            .collect();
        let spike_traces = (0..1000)
            .map(|_| {
                (0..150)
                    .map(|i| ((i as f64 / 10.0).sin() * rng.random::<f64>() * 0.2) + 0.5)
                    .collect()
            })
            .collect();

        let isi = &(0..1000).map(|_| rng.random_range(5.0..100.0)).collect();
        let intervals = compute_isi_histogram(isi, 250);

        let pca_2d = (0..100)
            .map(|i| {
                let t = i as f64 / 10.0;
                [t.cos(), t.sin()]
            })
            .collect();

        Self {
            trace_data,
            sr: 25000.0,
            threshold: None,
            threshold_mode: false,
            state: AppState::TraceView,
            spike_traces,
            intervals,
            pca_2d,
        }
    }
}

impl eframe::App for SpikeSorterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // this will initiate the threshold setting
        if self.state == AppState::TraceView {
            if ctx.input(|i| i.key_pressed(egui::Key::T)) {
                self.threshold_mode = !self.threshold_mode;
                println!(
                    "Threshold mode {}",
                    if self.threshold_mode { "ON" } else { "Off" }
                );
            }

            // lock threshold with enter
            if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                if self.threshold.is_some() {
                    self.state = AppState::PostThreshold;
                    println!("Threshold locked at {:?}", self.threshold);
                }
            }

            CentralPanel::default().show(ctx, |ui| {
                let available_size = ui.available_size();
                let _ = egui_plot::Plot::new("Filtered trace")
                    .view_aspect(available_size.x / available_size.y)
                    .include_y(-1.5)
                    .include_x(1.5)
                    .allow_zoom(true)
                    .allow_drag(true)
                    .allow_boxed_zoom(true)
                    .allow_double_click_reset(true)
                    .allow_scroll(true)
                    .show(ui, |plot_ui| {
                        let line = Line::new("", PlotPoints::from(self.trace_data.clone()));
                        plot_ui.line(line);

                        if let Some(click) = plot_ui.pointer_coordinate() {
                            if plot_ui.response().clicked_by(egui::PointerButton::Primary) {
                                self.threshold = Some(click.y);
                                println!("Set threshold at: {:.?}", self.threshold);
                            }
                        }

                        if let Some(thresh) = self.threshold {
                            let x_min = self.trace_data.first().unwrap()[0];
                            let x_max = self.trace_data.last().unwrap()[0];
                            let threshold_line = Line::new(
                                "",
                                PlotPoints::from(vec![[x_min, thresh], [x_max, thresh]]),
                            )
                            .color(egui::Color32::RED)
                            .style(egui_plot::LineStyle::Dashed { length: 5.0 });
                            plot_ui.line(threshold_line);
                        }
                    });
            });
        } else {
            if ctx.input(|i| i.key_pressed(egui::Key::R)) {
                self.state = AppState::TraceView;
                self.threshold = None;
                println!("Threshold cleared, ready to set a fresh one")
            }
            CentralPanel::default().show(ctx, |ui| {
                ui.allocate_ui_with_layout(
                    ui.available_size(),
                    Layout::top_down(egui::Align::Min),
                    |ui| {
                        let top_height = ui.available_height() * 0.6;

                        ui.allocate_ui(egui::Vec2::new(ui.available_width(), top_height), |ui| {
                            ui.columns(2, |columns| {
                                //left column: spike waveform
                                Plot::new("Spikes").view_aspect(1.0).show(
                                    &mut columns[0],
                                    |plot_ui| {
                                        for spike in &self.spike_traces {
                                            let points: PlotPoints = spike
                                                .iter()
                                                .enumerate()
                                                .map(|(j, &y)| [j as f64, y])
                                                .collect();
                                            plot_ui.line(
                                                Line::new("", points)
                                                    .color(Color32::from_rgb(100, 150, 255))
                                                    .width(1.0),
                                            );
                                        }
                                    },
                                );
                                //right column: pca
                                Plot::new("PCA").view_aspect(1.0).show(
                                    &mut columns[1],
                                    |plot_ui| {
                                        let points =
                                            egui_plot::Points::new("", self.pca_2d.clone())
                                                .color(Color32::LIGHT_RED)
                                                .radius(2.0);
                                        plot_ui.points(points);
                                    },
                                );
                            });
                        });
                        ui.allocate_ui(ui.available_size(), |ui| {
                            Plot::new("ISI Histogram")
                                .view_aspect(4.0)
                                .show(ui, |plot_ui| {
                                    let Bar = BarChart::new("", self.intervals.clone())
                                        .color(Color32::LIGHT_GRAY);
                                    plot_ui.bar_chart(Bar);
                                });
                        });
                    },
                );
            });

            //            CentralPanel::default().show(ctx, |ui| {
            //                egui_plot::Plot::new("locked th |reshold").show(ui, |plot_ui| {
            //                    let points: PlotPoints = (0 |..500)
            //                        .map(|i| {
            //                            let x = i as f64 *  |0.01;
            //                            [x, (x * 2.0).cos() |]
            //                        })
            //                        .collect();
            //                    plot_ui.line(Line::new("",  |points));
            //                });
            //            });
        }
    }
}

fn compute_isi_histogram(data: &Vec<f64>, num_bins: usize) -> Vec<Bar> {
    let min = data.iter().cloned().fold(0. / 0., f64::min);
    let max = data.iter().cloned().fold(0. / 0., f64::max);

    let bin_width = (max - min) / num_bins as f64;
    let mut counts = vec![0; num_bins];

    for &value in data {
        if value == max {
            // edge case, data goes into last bin
            counts[num_bins - 1] += 1;
        } else {
            let bin_index = ((value - min) / bin_width).floor() as usize;
            counts[bin_index] += 1;
        }
    }

    //Convert into Bars
    counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| {
            let center = min + (i as f64 + 0.5) * bin_width;
            Bar::new(center, count as f64).width(bin_width)
        })
        .collect()
}
