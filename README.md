# 通用型人脸识别系统
## Quick Start
### Environment
```
conda create -n fr_env python=3.8
conda activate fr_env
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Patch
>Modify function `compare_faces` in line 107,face_rec.py as follows:
> 
```python
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    # return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
    return list(face_distance(known_face_encodings, face_encoding_to_check))
```

### Upload pictures
> Upload a picture of your face  to the 'image' folder
> 

### Start face recognition
>Run face_rec.py
> 
