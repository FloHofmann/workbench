use spikeops::core::detect::detect_peaks;

#[test]
fn detects_simple_peak() {
    let trace = vec![0.0, 1.0, 0.0];
    let peaks = detect_peaks(&trace, 0.5, 1);
    assert_eq!(peaks, vec![1]);
}

#[test]
fn refractory_enforced() {
    let trace = vec![0.0, 1.0, 0.0, 1.0, 0.0];
    let peaks = detect_peaks(&trace, 0.5, 3);
    assert_eq!(peaks, vec![1]);
}
