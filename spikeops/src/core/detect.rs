/// detect_peaks of a filtered voltage trace
///
/// # Parameters
///
/// - 'trace': A ndarray of type float of filtered voltage trace
/// - 'threshold': float - sets the lower threshold for peak detection
/// - 'upper_threshold': float - sets the upper threshold for peak detection
/// - 'min_distance': int - the refractory period in samples to take into consideration
///
/// # Returns
///
/// A ndarray of type int where each entry refers to a sample a peak of a spike is detected on.
///
pub fn detect_peaks(
    trace: &[f64],
    threshold: f64,
    upper_threshold: f64,
    min_distance: usize,
) -> Vec<usize> {
    let n = trace.len();
    let mut peaks = Vec::new();
    if n < 3 {
        return peaks;
    }

    let d = min_distance.max(1);
    let mut i = 1;

    while i < n - 1 {
        // Find the first candidate local maximum >= threshold
        if trace[i] >= threshold
            && trace[i - 1] < trace[i]
            && trace[i] >= trace[i + 1]
            && trace[i] <= upper_threshold
        {
            // Search within the refractory window for the *best* (highest) peak
            let window_end = (i + d).min(n - 1);

            let mut best_i = i;
            let mut best_x = trace[i];

            // Note: start at i+1 because i already checked
            let mut j = i + 1;
            while j < window_end {
                let x = trace[j];
                if x >= threshold && x <= upper_threshold && trace[j - 1] < x && x >= trace[j + 1] {
                    if x > best_x {
                        best_x = x;
                        best_i = j;
                    }
                }
                j += 1;
            }

            peaks.push(best_i);
            i = window_end; // jump past refractory window
        } else {
            i += 1;
        }
    }

    peaks
}

/// extract_waveforms, means to retrieve the waveforms
///
/// # Parameters:
///
/// - 'trace': ndarray of type float. Contains the filtered voltage trace
/// - 'spiketimes': ndarray of type int. Contains the samples of spike occurences
/// - 'sampling_rate': dtype int. Sampling rate in Hz
/// - 'time_pre': dtype float. Timewindow in seconds (!) prior to spike
/// - 'time_post': dtype float. Timewindow in seconds (!) after spike
///
/// # Returns
///
/// - tuple of Waveshapes flattened to 1D and the amount of samples a single waveshape is made up
/// out of.
///
pub fn extract_waveforms(
    trace: &[f64],
    spiketimes: &[usize],
    sampling_rate: usize,
    time_pre: f64,
    time_post: f64,
) -> (Vec<f64>, usize, Vec<usize>) {
    let mut waves = Vec::new();
    // type conversion of sampling rate to to math
    let sampling_rate = sampling_rate as f64;
    // retrieve the samples as ints
    let samples_pre: usize = (time_pre * sampling_rate).round() as usize;
    let samples_post: usize = (time_post * sampling_rate).round() as usize;

    let num_samples_per_wave = samples_pre + samples_post + 1;
    let mut kept = Vec::with_capacity(spiketimes.len());
    waves.reserve(spiketimes.len() * num_samples_per_wave);

    // loop the peak samples and extract the waveshape
    for &i in spiketimes.into_iter() {
        if i < samples_pre || i + samples_post >= trace.len() {
            continue;
        }
        let start = i - samples_pre;
        let end = i + samples_post + 1;
        // append the vector by the slice of the trace
        kept.push(i);
        waves.extend_from_slice(&trace[start..end]);
    }

    (waves, num_samples_per_wave, kept)
}
