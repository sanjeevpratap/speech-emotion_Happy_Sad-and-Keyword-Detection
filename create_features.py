import os
import time
import joblib
import librosa
import numpy as np

# from config import SAVE_DIR_PATH
# from config import TRAINING_FILES_PATH
SAVE_DIR_PATH = 'speech-emotion_happy_sad_keyword/joblib_features'
TRAINING_FILES_PATH = 'speech-emotion1/dataset-actor'





class CreateFeatures:

    @staticmethod
    def features_creator(path, save_dir, target_emotions=('happy', 'sad')) -> str:
        lst = []
        start_time = time.time()

        for subdir, dirs, files in os.walk(path):
            print(f"Processing subdirectory: {subdir}")
            for file in files:
                # print(file)
                try:
                    # Load librosa array, obtain mfcss, store the file and the mcss information in a new array
                    file_path = os.path.join(subdir, file)
                    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
                    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate,
                                                         n_mfcc=40).T, axis=0)
                    
                    file = int(file[7:8]) - 1
                    if file==2 or file==3:
                        arr = mfccs, file
                        lst.append(arr)

                # If the file is not valid, skip it
                except ValueError as err:
                    print(err)
                    continue

        if not lst:
            print("No valid files found.")
            return "No valid files found."

        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

        # Creating X and y: zip makes a list of all the first elements, and a list of all the second elements.
        X, y = zip(*lst)

        # Array conversion
        X, y = np.asarray(X), np.asarray(y)

        # Array shape check
        print(X.shape, y.shape)

        # Preparing features dump
        X_name, y_name = 'X.joblib', 'y.joblib'

        joblib.dump(X, os.path.join(save_dir, X_name))
        joblib.dump(y, os.path.join(save_dir, y_name))

        return "Completed"

if __name__ == '__main__':
    print('Routine started')
    target_emotions = ('happy', 'sad')
    FEATURES = CreateFeatures.features_creator(path=TRAINING_FILES_PATH, save_dir=SAVE_DIR_PATH, target_emotions=target_emotions)
    print('Routine completed.')