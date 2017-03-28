import pandas as pd

from emotion_predictor import EmotionPredictor

pd.options.display.max_colwidth = 100   # show whole tweet's content
pd.options.display.width = 150          # don't break columns
pd.set_option('precision', 3)           # only 3 decimal places


model = EmotionPredictor('ekman', 'mc')
tweets = [
    'Thinking of u still brings tears to my eyes',
    'Watching the sopranos again from start to finish!',
    'I\'m so depressed today ...'
]

predictions = model.predict_classes(tweets)
print(predictions, '\n')

probabilities = model.predict_probabilities(tweets)
print(probabilities, '\n')

embeddings = model.embedd(tweets)
print(embeddings, '\n')
