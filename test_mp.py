import importlib
import mediapipe as mp

print('mp version:', mp.__version__)

try:
    mpt = importlib.import_module('mediapipe.tasks.python')
    print('tasks.python OK:', type(mpt))
except Exception as e:
    print('tasks.python error:', e)

try:
    mpv = importlib.import_module('mediapipe.tasks.python.vision')
    print('tasks.python.vision OK:', type(mpv))
    print('HandLandmarker:', hasattr(mpv, 'HandLandmarker'))
except Exception as e:
    print('tasks.python.vision error:', e)

try:
    hl = mpv.HandLandmarker
    print('HandLandmarker class:', hl)
except Exception as e:
    print('HandLandmarker error:', e)

import os
print('task file exists:', os.path.exists('model/hand_landmarker.task'))

