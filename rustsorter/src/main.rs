mod spikesorter_app;
use spikesorter_app::SpikeSorterApp;
use eframe::NativeOptions;

fn main() {
    let opts = NativeOptions::default();
    let _ = eframe::run_native(
        "Spikesorter",
        opts,
        Box::new(|_cc| Ok(Box::new(SpikeSorterApp::default()))),
    );
} 
