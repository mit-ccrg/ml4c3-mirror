## Tensorizer viewer

You can use this visualizer to plot signals
from hd5 files.

**Instructions**
- Type the path of the folder containing
the HD5 files on the input box _'input file path'_.
You can find a large number of hd5 files on the path `/media/ml4c3/hd5`.

- Select an hd5 file from that folder on the
dropdown located at its side.

- Select a visit ID on the dropdown
located below

**Warnings**

- Bedmaster waveforms have a lot of points
and may be very slow to load. You may consider cropping or
downsampling them with the sliders

- For some reason, plotting the signal as
points takes much more time than plotting it as
a line.

- If you select a event on an optimized graph, the signal will
be cropped to only the points that were taken during the event

**Useful resources**
- You can find a large number of hd5 files on the path `/media/ml4c3/hd5`.

-------------------------------------------
This viewer is designed to visualize hd5 files tensorized with
[ml4c3 repo](https://github.com/aguirre-lab/ml4c3).

Files contain data coming from two databases: Bedmaster and EDW.
Each file corresponds to a patient (MRN) and it contains the data
from all its recorded visits (CSN)

For more information, checkout the github code and wiki:
* Code: [github](https://github.com/aguirre-lab/ml4c3)
* Wiki: [github](https://github.com/aguirre-lab/ml4c3/wiki)
