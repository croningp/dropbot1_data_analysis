

##

This set of tools has been tested under Python 2.7.6 on Ubuntu 14.04 LTS. Despite all our efforts, we cannot guarantee everything will be executable on other OS or Python version.

## Dependencies

- [opencv](http://opencv.org/): Image analysis with python binding.
Version: cv2.__version__ is '2.4.8'

- [numpy](http://www.numpy.org/): Scientific computing in Python.
Version: numpy.__version__ is '1.10.4'

- [scipy](http://www.scipy.org/scipylib/index.html): More scientific computing in Python.
Version: scipy.__version__ is '0.16.1'

- [joblib](https://pythonhosted.org/joblib/): Easy parallel computing.
See also: https://github.com/joblib/joblib
Version: joblib.__version__ is '0.9.1'


## Download the raw data

Link to figshare

## Video analysis and feature extraction

Run the script [analyse_all_videos.py](feature_extraction/analyse_all_videos.py) in the [feature_extraction](feature_extraction) folder. This script will store a ```features.json``` along each ```video.avi```. See [feature_extraction](feature_extraction) folder for more info.

## Generating datasets

Run the script [generate_datasets.py](datasets/generate_datasets.py) in the [datasets](datasets) folder. This script will collect all experiments from the ```data``` folder and compile them into easy to use csv files. See [datasets](datasets) folder for more info.
