use eframe::{
    NativeOptions,
    egui::{self, CentralPanel},
};
use egui_plot::{self, Line, PlotPoints};

#[derive(PartialEq)]
enum AppState {
    Normal,
    ThresholdSetting,
    ThresholdLocked,
}

struct SpikeSorterApp {
    trace_data: Vec<[f64; 2]>,
    threshold: Option<f64>,
    state: AppState,
}

impl Default for SpikeSorterApp {
    fn default() -> Self {
        let trace_data = (0..10000)
            .map(|i| {
                let x = i as f64 * 0.001;
                [x, (x * 10.0).sin() * (x * 0.5).cos()]
            })
            .collect();

        Self {
            trace_data,
            threshold: None,
            state: AppState::Normal,
        }
    }
}

impl eframe::App for SpikeSorterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|i| i.key_pressed(egui::Key::T)) {
            match self.state {
                AppState::Normal => {
                    self.state = AppState::ThresholdSetting;
                    println!("Treshold mode ON. Click to set threshold");
                }
                AppState::ThresholdSetting => {
                    self.state = AppState::Normal;
                    println!("Threshold mode OFF.");
                }
                AppState::ThresholdLocked => {
                    // if lockec 't' doesn't toggle anything
                }
            }
        }

        // lock threshold with enter
        if ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
            if self.threshold.is_some() {
                self.state = AppState::ThresholdLocked;
                println!("Threshold locked at {:?}", self.threshold);
            }
        }
        CentralPanel::default().show(ctx, |ui| match self.state {
            AppState::ThresholdLocked => {
                egui_plot::Plot::new("locked threshold").show(ui, |plot_ui| {
                    let points: PlotPoints = (0..500)
                        .map(|i| {
                            let x = i as f64 * 0.01;
                            [x, (x * 2.0).cos()]
                        })
                        .collect();
                    plot_ui.line(Line::new("", points));
                });
            }
            _ => {
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

                        if self.state == AppState::ThresholdSetting {
                            if let Some(click) = plot_ui.pointer_coordinate() {
                                if plot_ui.response().clicked_by(egui::PointerButton::Primary) {
                                    self.threshold = Some(click.y);
                                    println!("Set threshold at: {:.?}", self.threshold);
                                }
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
            }
        });
    }
}

fn main() {
    let opts = NativeOptions::default();
    let _ = eframe::run_native(
        "Spikesorter",
        opts,
        Box::new(|_cc| Ok(Box::new(SpikeSorterApp::default()))),
    );
}
