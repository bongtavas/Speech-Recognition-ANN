Speech Recognition ANN Implementation
=====================================

An implementation of Speech Recognition using Artificial Neural Networks. 

Language Used: Python

You need numpy and scipy for this to work.

Words Recognized: "Apple", "Banana", "Kiwi", "Lime", "Orange"


#How to add new words

1. Record your new word in Audacity or any audio processing software. Set the sampling rate to 44100Hz then export into a .wav file. It would be better to record a lot of samples from different speakers to improve accuracy.

2. Put the wav files into the training_sets directory. Rename your wav files to the word you want to add + -sample_index (ex: hello-1.wav,hello-2.wav). In this way, the feature extractor later can iterate within the files easily.

3. In the featureExtractor.py, append your new word to the words array.

4. Run the featureExtractor.py. Numpy files with Mel Cepstrum Coefficients will be generated in the mfccData folder.

5. In anntrainer.py, go to the main function, open another file instance: Ex. f6 = open("mfccData/hello_mfcc.npy").

6. Load the npy file by using np.load() then concatenate it in the inputArray

7. You have to edit the Neural network target outputs, so if I'm going to add the word hello, I'll need to edit the results as follows

```
t1 = np.array([[1,0,0,0,0,0] for _ in range(len(inputArray1))]) #Apple
t2 = np.array([[0,1,0,0,0,0] for _ in range(len(inputArray2))]) #Banana
t3 = np.array([[0,0,1,0,0,0] for _ in range(len(inputArray3))]) #Kiwi
t4 = np.array([[0,0,0,1,0,0] for _ in range(len(inputArray4))]) #Lime
t5 = np.array([[0,0,0,0,1,0] for _ in range(len(inputArray5))]) #Orange
t6 = np.array([[0,0,0,0,0,1] for _ in range(len(inputArray6))]) #Hello

target = np.concatenate([t1,t2,t3,t4,t5,t6])
```

then run anntrainer.py. This could take a lot of time to compute. Grab a coffee while you wait =)

#Running the speech recognizer
Just run main.py! =)
You can view demo.mp4 for sample usage.

#Developers
A CS 180 Artificial Intelligence Project, University of the Philippines Diliman
Developers: Romelio Tavas Jr., Dion Melosantos
