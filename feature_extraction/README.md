In this folder are all the tools needed to extract features from droplet videos.

The file ```droplet_tracker``` contains a class ```DropletTracker``` that can:
- open a video file and extract the position of all droplets at each frame. See function ```analyse_video()``` that populate the variable ```track_data```
- given some tracking_data, some high-level behaviors can be analysed. See the function ```ffx_movement()```, ```ffx_division()```, and ```ffx_directionality()```

To run the analysis on all videos from the dataset, you should:
- have all data under the data folder

- run the script ```analyse_all_videos.py```, i.e. do ```python analyse_all_videos.py```

This will add two files aside with the video:
- ```features.json``` contains the behavioral measures as a json formatted dictionary. E.g. for ```octanoic/0000``` we have ```{"division": 2, "directionality": 7.411264642958634, "movement": 0.43765790198706866}```
- ```tracking_info.json``` contains all information about the droplet position for each frame.
