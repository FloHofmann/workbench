# This will result in a collective doc listing necessary functionalities

## Data
-[ ] load the data from .smrx files
-[ ] store the data in a h5 format for fast data retrieval. Downside is, that it need be be binarized in the process which has further implications for data retrieval -> maybe it's worth to check for changeability of that filetype or if changeability/appendability is even necessary
    - maybe a numpy file is more suitable, for appendability -> needs research
-[ ] Spikesorting. Either directly from the .smrx files, or from .h5 files. Would also result in a h5 file for storage. Best case would be rust (already some progress here), or c++ for smoother and faster interactivity with the sorting window.
## Analysis
-[ ] Angular movement dependent firing rate (through the receptive field?)

## Videography
-[ ] top down platform angle tracking, very important this time, if i want to apply it to the hypothesis of SC aspirated directional impairment. (Does already exist, may need to look into it once more to validate)

## Figures
-[ ] function for formatting figures and saving as .pdf and .png at defined path
