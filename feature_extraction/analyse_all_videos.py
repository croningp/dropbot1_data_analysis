import os
import time

import json
import droplet_tracker


def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


# anaylse a video for using the DropletTracker class from DropletTracker.py file
def process_video(video_path):
    start = time.time()

    print 'Analysing ', video_path, ' ...'

    (head, tail) = os.path.split(video_path)
    (filename, ext) = os.path.splitext(tail)

    # prepare the new file name
    tracking_data_file = os.path.join(head, 'tracking_info.json')
    feature_file = os.path.join(head, 'features.json')

    # if outputed file already exist, the video has been analysed, so we skip it
    # comment this to reanalyse every video
    if os.path.exists(feature_file):
        print 'Already processed ', video_path
        return

    # created the tracker
    dropTracker = droplet_tracker.DropletTracker(video_path)

    try:
        # analyse the video
        dropTracker.analyse_video(debug=False)

        # save important data that are costly to reprocess
        tracking_data = {}
        tracking_data['total_frames'] = dropTracker.total_frames
        tracking_data['track_data'] = dropTracker.track_data

        save_to_file(tracking_data, tracking_data_file)

        # get features and save
        features = {}
        features['movement'] = dropTracker.ffx_movement()
        features['division'] = dropTracker.ffx_division()
        features['directionality'] = dropTracker.ffx_directionality()

        save_to_file(features, feature_file)

    except:
        print 'FAILED: {}'.format(video_path)
    ##
    elapsed = time.time() - start
    print 'It took {} seconds to analyse {}'.format(elapsed, video_path)

    return (video_path, elapsed)

if __name__ == "__main__":

    # this get our current location in the file system
    import inspect
    here_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    data_folder = os.path.join(here_path, '..', 'data')
    videofile_pattern = 'video.avi'

    # adding parent directory to path, so we can access the utils easily
    import sys
    utils_path = os.path.join(here_path, '..')
    sys.path.append(utils_path)

    # filetools is a custom library that we provide here
    # this library can be found online at https://github.com/jgrizou/filetools
    from utils import filetools
    # look for all video.avi files under data_folder
    video_files = filetools.list_files(data_folder, [videofile_pattern])

    from joblib import Parallel, delayed
    import multiprocessing

    # use maximum number of core available
    num_cores = multiprocessing.cpu_count()

    # process all video found in parallel
    results = Parallel(n_jobs=num_cores)(delayed(process_video)(video_path) for video_path in video_files)
