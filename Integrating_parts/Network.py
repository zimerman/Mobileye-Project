import seaborn as sbn
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from Frame_container import FrameContainer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def network(images, model):
    crop_shape = (81, 81)
    predictions = model.predict(images.reshape([-1] + list(crop_shape) + [3]))
    return predictions[0][1] > .97


