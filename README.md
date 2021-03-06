# Facial-Detection-and-Recognition

<h3>Aim</h3>
Detect faces, predict gender, age & emotion


<h3>Business Values</h3>
Shopping mall & shops - marketing campaign 
<br>
Smart marketing - Advertisement


<h3>Process</h3>

<img width="1440" alt="1" src="https://user-images.githubusercontent.com/80112729/122373729-4c786480-cf94-11eb-8b67-a3770dbc8dc2.png">

<h3>Data Collection</h3>
1. UTKFACE
<br>
Gender, age, race
<br>
20K+ cropped faces images
<br>
male and female 
<br>
age 1-116
<br>
5 Ethnicities
<br>   
Reference:
https://www.kaggle.com/jangedoo/utkface-new
<br>
<br>
2. EMOTION 
<br>
35K+ cropped face images 
<br>
Contains 7 emotions (happiness, neutral, sadness, anger, surprise, disgust, fear)
<br>
Reference:
https://www.kaggle.com/ananthu017/emotion-detection-fer

<h3>Exploratory Data Analysis</h3>
<img width="1439" alt="2" src="https://user-images.githubusercontent.com/80112729/122373959-86496b00-cf94-11eb-9973-fdd8609bce5f.png">


<h3>***Gender Prediction***</h3>

```
Test accuracy = ~0.85
```

<img width="584" alt="3 1" src="https://user-images.githubusercontent.com/80112729/122376327-a24e0c00-cf96-11eb-81b0-dc01a0042a22.png">


<h3>***Age Prediction***</h3>

```
Test accuracy = ~0.46
```

<img width="584" alt="4 1" src="https://user-images.githubusercontent.com/80112729/122376602-dfb29980-cf96-11eb-8770-a0bb709a8f49.png">


<h3>***Emotion Prediction***</h3>

```
Test accuracy = ~0.85
```

<img width="584" alt="5 1" src="https://user-images.githubusercontent.com/80112729/122376916-2dc79d00-cf97-11eb-83bf-598bc93391f5.png">

<h3>Challenges</h3>

```
1. Age prediction on elderly
Inaccurate prediction on elderly
Dataset consists small amount of images of elderly, resulting the model was trained witih a skewed dataset
For an image of elder people below, the age predicton are, 0-2, 3-11 and 25-34, which are inaccurate
```

<img width="700" alt="7" src="https://user-images.githubusercontent.com/80112729/122378515-9a8f6700-cf98-11eb-8e44-3cd99db7bde8.png">


```
2. Emotion prediciton on "unhappy" emotion
Inaccurate prediction on unhappy emotion
The situation is very similar to age prediction. The dataset does not consist enough images with unhappy emotion, the dataset was also skewed. 
For an image of unhappy person below, the emotion prediction should be "unhappy", but the model predicted "Neutral", the result is not desirable for emotion prediction
```
<img width="1375" alt="8" src="https://user-images.githubusercontent.com/80112729/122379921-e8f13580-cf99-11eb-921b-e9734321e429.png">

<h3>Conclusion</h3>
<br>
 - High accuracy (~80%) on gender and emotion, but not for age (~46%).
<br>
<br>
 - Skewed dataset impact the model's accuracy significantly. 
<br>
<br>
 - The fewer the classes, the more accurate the model can be.
<br>
