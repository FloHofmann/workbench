mod py_interface;
mod spikesorter_app;
use eframe::NativeOptions;
use spikesorter_app::SpikeSorterApp;

fn main() {
    // this initializes the spikesorter app
    let opts = NativeOptions::default();
    let _ = eframe::run_native(
        "Spikesorter",
        opts,
        Box::new(|_cc| Ok(Box::new(SpikeSorterApp::default()))),
    );
}
