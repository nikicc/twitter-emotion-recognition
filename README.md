# Twitter Emotion Recognition

**start comment**  
This fork of https://github.com/nikicc/twitter-emotion-recognition has been updated to include:  
1. A requirements.txt file that creates a working environment (tested on Binder);  
2. A link to launch this repo in Binder. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/polsci/twitter-emotion-recognition/master)  
3. A notebook to run the demo (basically duplicating demo.py)  
**end comment**  

Trained recurrent neural network (RNN) models for predicting emotions from English tweets.
Our models work on characters hence we pass the whole tweet without any preprocessing as an input to the RNN.
We are predicting three emotion classifications:
* Ekman's six basic emotions,
* Plutchik's eight basic emotions,
* Profile of Mood States (POMS) six mood states.

### Example Usage
The following examples show how to predict Ekman's emotions from tweet's content.
First let's import `EmotionPredictor`.
```python
>>> from emotion_predictor import EmotionPredictor
```
Next we instantiate the model and define our tweets.
In this example we will work with Ekman's emotions.
Use `plutchik` to predict Plutchik's emotions or `poms` for Profile of Mood States.
To use models in multilabel setting instead of multiclass provide `ml` as the `setting` argument.
```python
>>> model = EmotionPredictor(classification='ekman', setting='mc')

>>> tweets = [
    "Watching the sopranos again from start to finish!",
    "Finding out i have to go to the  dentist tomorrow",
    "I want to go outside and chalk but I have no chalk",
    "I HATE PAPERS AH #AH #HATE",
    "My mom wasn't mad",
    "Do people have no Respect for themselves or you know others peoples homes",
]
```
We obtain model's predictions by calling `predict_classes` method:
```python  
>>> model.predict_classes(tweets)
                                                                       Tweet   Emotion
0                          Watching the sopranos again from start to finish!       Joy
1                          Finding out i have to go to the  dentist tomorrow      Fear
2                         I want to go outside and chalk but I have no chalk   Sadness
3                                                 I HATE PAPERS AH #AH #HATE     Anger
4                                                          My mom wasn't mad  Surprise
5  Do people have no Respect for themselves or you know others peoples homes   Disgust
```

To observe probabilities for each class use `predict_probabilities` method:
```python
>>> model.predict_probabilities(tweets)
                                               Tweet     Anger   Disgust      Fear       Joy   Sadness  Surprise
0  Watching the sopranos again from start to finish!  0.000717  0.000244  0.003829  0.946539  0.005610  0.043061
1  Finding out i have to go to the  dentist tomorrow  0.007705  0.000039  0.783890  0.198629  0.008950  0.000787
2  I want to go outside and chalk but I have no c...  0.002772  0.000095  0.004137  0.025035  0.963712  0.004249
3                         I HATE PAPERS AH #AH #HATE  0.956343  0.006368  0.031387  0.000350  0.004375  0.001176
4                                  My mom wasn't mad  0.063969  0.004990  0.013971  0.079884  0.218708  0.618478
5  Do people have no Respect for themselves or yo...  0.070003  0.801428  0.067724  0.003646  0.038480  0.018718
```
If you would rather just use the final hidden state representation call `embed`:
```python
>>> model.embed(tweets)
                                               Tweet      Dim1      Dim2    ...       Dim798    Dim799    Dim800
0  Watching the sopranos again from start to finish! -0.128762 -0.000000    ...    -0.260896 -0.009062 -0.110209
1  Finding out i have to go to the  dentist tomorrow -0.525602  0.407847    ...    -0.000088 -0.001489  0.142871
2  I want to go outside and chalk but I have no c... -0.057850  0.566420    ...    -0.091341 -0.003914 -0.037481
3                         I HATE PAPERS AH #AH #HATE  0.019670 -0.288512    ...     0.100234  0.013350 -0.014305
4                                  My mom wasn't mad -0.004135  0.657584    ...    -0.029319 -0.007455 -0.066208
5  Do people have no Respect for themselves or yo... -0.246179  0.069080    ...     0.029919  0.011467 -0.000520

[6 rows x 801 columns]
```
### Files and Folders:
* __*demo.py*__: script is showing how to use our models for predicting emotions or embedding tweets.
* __*emotion_prediction.py*__: helper scripts that defines EmotionPredictor class.
* __*models/*__: contains trained RNN models.

### Citing:

If you use our models in a scientific publication, we would appreciate citations to the following paper:

*Colnerič, N., & Demšar, J. (2018). Emotion Recognition on Twitter: Comparative Study and Training a Unison Model. IEEE Transactions on Affective Computing, PP (99), 1. https://doi.org/10.1109/TAFFC.2018.2807817*
