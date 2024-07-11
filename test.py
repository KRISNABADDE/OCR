import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from ocr.model import OCRModel

if __name__ == "__main__":

    ocr_model = OCRModel(img_w=64, img_h=128, num_classes=37, max_length=10)

    model =ocr_model.model_building()
    model.summary()