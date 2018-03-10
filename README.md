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
  
## Training  <!--- TODO --> 
[CK data set](http://www.pitt.edu/~emotion/ck-spread.htm) 


## TODO
#### Prio 1:
- refactor NN classes & inheritance structure
- Remove duplications
- Add CK data set & update instructions for installation
- get all models to run the training phase properly


#### Prio 2:
- test all models & fix bugs
- Update DocStrings & README.md
- include [deep visualization toolbox](https://github.com/yosinski/deep-visualization-toolbox)

#### Prio 3:
- Implement StackGANEmoPy
- Write a nice GUI for experimenting & tweaking the architectures & their hyper-parameters