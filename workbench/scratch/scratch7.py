from pathlib import Path
from scipy.io import loadmat
from scipy.signal import filtfilt, butter, find_peaks
from spikeops import detect_peaks
import numpy as np
import matplotlib.pyplot as plt
import timeit

p = Path(r"\\172.25.250.112\burgalossi\lab share\Data\Florian\ADN\ADNFH4\analysis\Data7\tones-3\exp_data.mat")
file = loadmat(p, simplify_cells=True)

raw_trace = file['raw_data']

voltage = raw_trace['ephys_data']['traces']

fs = 25000

nyquist = 0.5 * 25000

normal_cutoff = 300/nyquist

b, a = butter(4, normal_cutoff, btype='high', analog=False)

filtered = np.array(filtfilt(b, a, voltage), dtype='float')

k = 6
u = 30

sigma = np.median(np.abs(filtered - np.median(filtered))) / 0.6745
threshold = k * sigma
upper_threshold = u * sigma

pks = detect_peaks(filtered, threshold, upper_threshold, int(0.001*fs))

# warm-up
_ = detect_peaks(filtered, threshold, upper_threshold, int(0.001 * fs))
_ = find_peaks(filtered, height=threshold, distance=int(0.001 * fs))

n_runs = 20
min_dist = int(0.001 * fs)

t_rust = timeit.timeit(
    lambda: detect_peaks(filtered, threshold, upper_threshold, min_dist),
    number=n_runs,
)

t_scipy = timeit.timeit(
    lambda: find_peaks(filtered, height=threshold, distance=min_dist),
    number=n_runs,
)

print(f"Rust detect_peaks: {t_rust/n_runs*1e3:.2f} ms per run")
print(f"SciPy find_peaks:  {t_scipy/n_runs*1e3:.2f} ms per run")

pks_rust = detect_peaks(filtered, threshold, upper_threshold, min_dist)
pks_scipy, _ = find_peaks(filtered, height=threshold, distance=min_dist)
pks_scipy = pks_scipy[filtered[pks_scipy] < upper_threshold]

print("Rust peaks:", len(pks_rust))
print("SciPy peaks:", len(pks_scipy))
print("Intersection:", len(set(pks_rust).intersection(pks_scipy)))

p_r = np.asarray(pks_rust)
p_s = np.asarray(pks_scipy)

# peaks in rust but not scipy
only_r = np.setdiff1d(p_r, p_s)
only_s = np.setdiff1d(p_s, p_r)

print(len(only_r), len(only_s))
if len(only_r) == 0:
    print("Perfect match: no mismatched peaks.")
else:
    nearest = np.min(np.abs(only_r[:, None] - p_s[None, :]), axis=1)
    print("median mismatch (samples):", np.median(nearest))
    print("max mismatch (samples):", np.max(nearest))

plt.plot(filtered)
plt.axhline(float(threshold), color='r')
plt.axhline(float(upper_threshold), color='r')
plt.scatter(pks, filtered[pks], color='b')

plt.show()
