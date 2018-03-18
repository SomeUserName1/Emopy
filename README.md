# Emopy
Emotion recognition from facial expression using Dlib, Keras and Tensorflow by 
[Mitiku1](https://github.com/mitiku1).

Including (work in progress)
- Dlib output, RGB image of arbitary size, Video, WebCam Stream pre-processing
- Basic NN (either images or dlib), MultiInput NN (images and dlib), LSTM (either dlib or images), CapsNet (images)
- Visualizations of the neurons using techniques of 
[deep visualization toolbox](https://github.com/yosinski/deep-visualization-toolbox)

## Dependencies
- dlib
- boost
- Keras
- Tensorflow
- OpenCV
- keras_vggface
- numpy
- scipy
- sklearn
- pandas

#### Troubleshooting
- Windows:
    - When installing the dlib/boost package on Python 3.6, you may get an EncodingError. This 
[PEP528](https://www.python.org/dev/peps/pep-0528/) changes the standard encoding of pip3 on the 
cmd.exe to utf8. For some reason Windows 10 uses some special codepage.   
To fix this: ([Credits to robinxb](https://github.com/pypa/pip/issues/4251#issuecomment-279117184))   
Type in cmd/PowerShell: 
 `chcp`   
 Then edit `[...]/Python36/Lib/Site-packages/pip/compat/__init__.py: Line 75`  
 `return s.decode('utf_8')` to `return s.decode('cp123') `  
  where 123 is to be replaced by the chcp output.
  
 - MacOS: No trouble up to now
  
## Training  
### Images: 
#### CK 
[CK data set info page](http://www.pitt.edu/~emotion/ck-spread.htm) 

There are 327 images with direct emotional labels available.  
For the other images is FACS data available,
 but currently this project does not contain a FACS to "emotional label parser"/mapping to three continuous dimensions:
 bipolar arousal, unipolar positive and negative valence as proposed e.g. by 
 [Affective Signal Processing, Egon von den Broek, Ch. 5.2.2, P.77](https://research.utwente.nl/en/publications/affective-signal-processing-asp-unraveling-the-mystery-of-emotion)

1. Fill the CK and CK+ DATABASE USER AGREEMENT form to receive a download link using the network of your currently 
enrolled university
[CK and CK+ DATABASE USER AGREEMENT](http://www.consortium.ri.cmu.edu/ckagree/)
2. Check your E-Mails
3. Login and download either the extended data base
4. Extract the data set to ```~/DataSets``` 
- all images and labels in one folder  
- all xls and doc files  
so that it looks like this
    ```
    -CK
      -EmoPyData
      -images
        -S005
            -001
                S005_001_0000001.png
                ...
                S005_001_0000011.png
                S005_001_0000011_emotion.txt
                S005_001_0000011_facs.txt
                S005_001_0000011_landmarks.txt
        -S010
            ...
        ...        
  emotions.txt
  facs.csv
  Cohn-Kanade Database FACS codes[...]
  ReadMe
      ...
    ```
#### fer2013
[Kaggle Page](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
1. Download and extract the data set to ```~/DataSets/fer2013```
``` 
-fer2013
 fer2013.bib
 fer2013.csv
 README        
 ```

Then run the runner with ```session = 'init_data'```
#### Video
 [75 Emotions in less than a minute on YouTube](https://www.youtube.com/watch?v=ypqQ_mJIU3M)
 
 
### To run the training start the runner script with ```session = 'train'```

## Prediction

## Experiments
1. Replacing max pooling with conv layers with strides
    . slows learning rate
    . reduces test accuracy 
    . increases over fitting rate
2. Adding dropout layer increases training time.
3. Large learning rate with dropout layer after each 
    conv layer might hinder learning


## TODO:
### General
- Update DocStrings, README.md's and UML diagrams
- move webcam and video stuff out of keras_models and rather into data_collect/some preprocessor

### keras_models 
- refactor build & implement predict
- Write a nice GUI for experimenting & tweaking the architectures & their hyper-parameters
- implement some InfoGAN/StackGAN model

### preprocess
- refactor inheritance
- what does feature extractor exactly extract?
- add EMG. EDA, ECG preprocessor
- add FACS preprocessor

### util
- implement 3 continuous dimension classifier for arrousal, positive valence and negative valence
- what does postprocess do?
