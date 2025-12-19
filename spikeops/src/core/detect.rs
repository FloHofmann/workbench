// peak detection (positive/negative)
// refractory hadnling (distance)
// optional prominence-lite
// alignment
// snippet extraction

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
