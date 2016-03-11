import os
import time

import json
import DropletTracker


def save_to_file(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def process_video(video_path):
    start = time.time()

    print 'Analysing ', video_path, ' ...'

    (head, tail) = os.path.split(video_path)
    (filename, ext) = os.path.splitext(tail)

    tracking_data_file = os.path.join(head, 'tracking_info.json')
    feature_file = os.path.join(head, 'features.json')

    if os.path.exists(feature_file):
        print 'Already processed ', video_path
        return

    dropTracker = DropletTracker.DropletTracker(video_path)

    try:
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
        print 'FAILED ', video_path
    ##
    elapsed = time.time() - start
    print elapsed, ' for ', video_path

    return (video_path, elapsed)

if __name__ == "__main__":

    import inspect
    here_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    data_folder = os.path.join(here_path, '../data')
    videofile_pattern = 'video.avi'

    import sys
    utils_path = os.path.join(here_path, '..')
    sys.path.append(utils_path)

    from utils import filetools

    video_files = filetools.list_files(data_folder, [videofile_pattern])

    from joblib import Parallel, delayed
    import multiprocessing

    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(delayed(process_video)(video_path) for video_path in video_files)
