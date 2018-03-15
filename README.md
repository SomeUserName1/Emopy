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
#### Images: [CK data set info page](http://www.pitt.edu/~emotion/ck-spread.htm) 

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
4. Extract the data set: 
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
5. run the ck_folder_structure.py script with
 ``python ck_folder_structure.py --ck Path/to/CK --out Path/to/EmoPyData``
5. Alter config.py, train_config.py and test_config.py accordingly
6. Start training

#### Video
 [75 Emotions in less than a minute on YouTube](https://www.youtube.com/watch?v=ypqQ_mJIU3M)
 
## Evaluation & Prediction

## Experiments

- Experimental results:
1. Replacing max pooling with conv layers with strides
    . slows learning rate
    . reduces test accuracy 
    . increases over fitting rate
2. Adding dropout layer increases training time.
3. Large learning rate with dropout layer after each 
    conv layer might hinder learning


## TODO:
1. Get data setup finished for the 3 data sets
2. Refactor Keras models (inheritance: fields, params, init, build, train, eval & predict, ...)
3. Build EMG, EDA, ECG preprocessor
4. Update DocStrings, README.md's and UML diagrams


