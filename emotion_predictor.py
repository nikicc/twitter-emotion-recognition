import html
import pickle
import re

import pandas as pd
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import sequence


class EmotionPredictor:
    def __init__(self, classification, setting):
        """

        Args:
            classification (str): Either 'ekman', 'plutchik', 'poms'
                or 'unison'.
            setting (str): Either 'mc' or 'ml'.
        """
        if classification not in ['ekman', 'plutchik', 'poms', 'unison']:
            raise ValueError('Unknown emotion classification: {}'.format(
                classification))
        if setting not in ['mc', 'ml']:
            raise ValueError('Unknown setting: {}'.format(setting))

        self.classification = classification
        self.setting = setting
        self.model = self._get_model()
        self.embeddings_model = self._get_embeddings_model()
        self.char_to_ind = self._get_char_mapping()
        self.class_values = self._get_class_values()
        self.max_len = self._get_max_sequence_length()

    def _get_model(self):
        return load_model('models/{}-{}.h5'.format(
            self.classification, self.setting))

    def _get_embeddings_model(self):
        last_layer_output = K.function([self.model.layers[0].input,
                                        K.learning_phase()],
                                       [self.model.layers[-3].output])
        return lambda x: last_layer_output([x, 0])[0]

    @staticmethod
    def _get_char_mapping():
        with open('models/allowed-chars.pkl', 'rb') as f:
            return pickle.load(f)

    def _get_class_values(self):
        if self.classification == 'ekman':
            return ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
        elif self.classification == 'plutchik':
            return ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise',
                    'Trust', 'Anticipation']
        elif self.classification == 'poms':
            return ['Anger', 'Depression', 'Fatigue', 'Vigour', 'Tension',
                    'Confusion']

    def _get_max_sequence_length(self):
        if self.classification in ['ekman', 'plutchik']:
            return 141
        elif self.classification == 'poms':
            return 143

    def predict_classes(self, tweets):
        indices = self._tweet_to_indices(tweets)
        predictions = self.model.predict(indices, verbose=False)

        df = pd.DataFrame({'Tweet': tweets})
        if self.setting == 'mc':
            df['Emotion'] = [self.class_values[i] for i in
                        predictions.argmax(axis=-1)]
        else:
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            for emotion, values in zip(self.class_values, predictions.T):
                df[emotion] = values
        return df

    def predict_probabilities(self, tweets):
        indices = self._tweet_to_indices(tweets)
        predictions = self.model.predict(indices, verbose=False)

        df = pd.DataFrame({'Tweet': tweets})
        for emotion, values in zip(self.class_values, predictions.T):
            df[emotion] = values
        return df

    def embedd(self, tweets):
        indices = self._tweet_to_indices(tweets)
        embeddings = self.embeddings_model(indices)

        df = pd.DataFrame({'Tweet': tweets})
        for index, values in enumerate(embeddings.T, start=1):
            df['Dim{}'.format(index)] = values
        return df

    def _tweet_to_indices(self, tweets):
        indices = []
        for t in tweets:
            t = html.unescape(t)                            # unescape HTML
            t = re.sub(r"http\S+", "", t)                   # remove normal URLS
            t = re.sub(r"pic\.twitter\.com/\S+", "", t)     # remove pic.twitter.com URLS
            indices.append([self.char_to_ind[char] for char in t])
        return sequence.pad_sequences(indices, maxlen=self.max_len)
