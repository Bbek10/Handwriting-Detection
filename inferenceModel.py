import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs



class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text
    
    def detect(path, filename):
        configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202301111911/configs.yaml")
        model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

        image = cv2.imread(path)
        prediction_text = model.predict(image)
        # cer = get_cer(prediction_text)
        print(f"Image: {path}, Prediction: {prediction_text}")
        image = cv2.resize(image, (image.shape[1] * 4, image.shape[0] * 4))
        return prediction_text

