import keras
import librosa
import numpy as np



import speech_recognition as sr
EXAMPLES_PATH='speech-emotion_happy_sad_keyword/examples'
MODEL_DIR_PATH='speech-emotion_happy_sad_keyword/model'

class LivePredictions:
    """
    Main class of the application.
    """

    def __init__(self, file):
        """
        Init method is used to initialize the main parameters.
        """
        self.file = file
        self.path = MODEL_DIR_PATH + '\\Emotion_Voice_Detection_Model.h5'
        self.loaded_model = keras.models.load_model(self.path)

    def make_predictions(self):
        """
        Method to process the files and create your features.
        """
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        
        # Reshape mfccs to match the expected input shape of the model
        mfccs = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(mfccs, axis=0)
        
        predictions_prob = self.loaded_model.predict(x)
        predicted_class = np.argmax(predictions_prob)

        print("Prediction is", " ", self.convert_class_to_emotion(predicted_class))


    @staticmethod
    def convert_class_to_emotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {
                            '2': 'happy',
                            '3': 'sad',
                            }

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

def count_word_occurrences(file_path, target_word):
    recognizer = sr.Recognizer()

    with sr.AudioFile(file_path) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

        try:
           response = recognizer.recognize_google(audio_data, show_all=True)
        #    print("Full Response:", response)
           # Adjust indexing based on the structure of the response
           text = response['alternative'][0]['transcript']
           word_count = text.lower().split().count(target_word.lower())
        #    print(f"Complete Text: {text}")
           return word_count

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


if __name__ == '__main__':
    live_prediction = LivePredictions(file=EXAMPLES_PATH + '\\03-01-03-01-01-01-06.wav')
    # live_prediction.loaded_model.summary()
    live_prediction.make_predictions()
    audio_file_path = EXAMPLES_PATH + '\\03-01-03-01-01-01-06.wav'
    target_word = "door"

    occurrences = count_word_occurrences(audio_file_path, target_word)
    print(f"The word '{target_word}' occurs {occurrences} times.")




    

    # live_prediction = LivePredictions(file=EXAMPLES_PATH + '\\10-16-07-29-82-30-63.wav')
    # live_prediction.make_predictions()
