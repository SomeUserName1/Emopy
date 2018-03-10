# Emopy
Emotion recognition from facial expression using Keras & Tensorflow by 
[Mitiku1](https://github.com/mitiku1).

## Dependencies
- dlib
- boost
- Keras
- Tensorflow
- OpenCV

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
  
## Training  <!--- TODO --> 
[CK data set](http://www.pitt.edu/~emotion/ck-spread.htm) 


## TODO
#### Prio 1:
- Add DocStrings & update README.md
- refactor NN classes & inheritance structure
- Remove duplications

#### Prio 2:
- Add CK data set instructions 
- test all models (& unit tests?)

#### Prio 3:
- Implement StackGANEmoPy