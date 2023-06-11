import cv2
from sklearn.cluster import KMeans
import os
import pickle

import imutils
import colorsys
import pytesseract
from pytesseract import Output
import torch

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops

# Define the dictionary
language_ratio = {
    1.0000: 1,
    0.0000: 2,
    0.0128: 3,
    0.0012: 4,
    0.0025: 5,
    0.8281: 6,
    0.9421: 7,
    0.8224: 9
}

# Define global variable
y_pred = []

class ImageProcessor:

    @staticmethod
    def get_image_paths(directory):
        """Return a list of image file paths in the given directory."""
        return sorted([os.path.join(directory, filename)
                       for filename in os.listdir(directory)
                       if filename.endswith('.jpg') or filename.endswith('.png')])

    @staticmethod
    def get_major_color(image):
        """Gets the major color of the image."""
        reshape = image.reshape(-1, 3)
        cluster = KMeans(n_clusters=1, n_init=10).fit(reshape)
        return cluster.cluster_centers_[0]

    @staticmethod
    def get_text(image):
        """Uses OCR to extract text from the image."""
        return pytesseract.image_to_string(image)

    @staticmethod
    def is_dark_color(rgb):
        """Determines whether a color is dark."""
        r, g, b = [x / 255.0 for x in rgb]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return l < 0.5

def closest_aspect_ratio_document(aspect_ratio):
    document1f_ratio = 701/994/1
    document5f_ratio = 689/956/1

    diff1 = abs(aspect_ratio - document1f_ratio)
    diff2 = abs(aspect_ratio - document5f_ratio)

    if diff1 < diff2:
        return 0 #'document1b'
    else:
        return 8 #'document5b'

def closest_aspect_ratio_plug(aspect_ratio):
    plug1_ratio = 0.4050632911392405/1
    plug2_ratio = 0.3055141579731744/1

    diff1 = abs(aspect_ratio - plug1_ratio)
    diff2 = abs(aspect_ratio - plug2_ratio)

    if diff1 < diff2:
        return 13 #'plug1'
    else:
        return 14 #'plug2'

def rotate_and_crop(img, foreground_color='black'):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image, setting a threshold to find objects
    if foreground_color == 'black':
        _, thresholded = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    elif foreground_color == 'white':
        _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  #225 #200

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are empty, return None or handle appropriately
    if not contours:
        print("No contours found in the image.")
        return None, None

    # Calculate the minimum enclosing rectangle and related parameters for the largest contour
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = cv2.boxPoints(rect).astype('int')
    center, (width, height), angle = rect
    if angle < -45: angle += 90
    if angle < 0: width, height = height, width

    # Apply the affine transform to the original image
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Crop the rotated rectangle and apply affine transform to cropped area
    Xs, Ys = zip(*cv2.boxPoints(((center[0], center[1]), (width, height), 0)))
    cropped = rotated[min(Ys).astype(int):max(Ys).astype(int), min(Xs).astype(int):max(Xs).astype(int)]

    if cropped.shape[1] < cropped.shape[0]:
        M_crop = cv2.getRotationMatrix2D((cropped.shape[1] / 2, cropped.shape[0] / 2), 90, 1)
        cropped = cv2.warpAffine(cropped, M_crop, (cropped.shape[0], cropped.shape[1]))  # Swapped width and height

    # Calculate aspect ratio
    aspect_ratio = cropped.shape[1] / cropped.shape[0]

    return cropped ,aspect_ratio


def correct_text_orientation(img):
    # Use Tesseract to determine the orientation of the text
    results = pytesseract.image_to_osd(img, config='--dpi 70', output_type=Output.DICT)

    # Rotate the image to correct the text orientation
    rotated = imutils.rotate_bound(img, angle=results["rotate"])
    return rotated

def calculate_ratio(input_str):
    chinese_count = 0
    english_count = 0

    for char in input_str:
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            english_count += 1
        elif '\u4e00' <= char <= '\u9fff':
            chinese_count += 1

    total_count = english_count + chinese_count

    if total_count == 0:
        return 0, 0
    else:
        english_ratio = format(english_count / total_count, '.4f')
        chinese_ratio = format(chinese_count / total_count, '.4f')
        return chinese_ratio, english_ratio

def closest_value(input_value, language_ratio):
    closest_key = min(language_ratio.keys(), key=lambda x:abs(x-input_value))
    return language_ratio[closest_key]

class DetectionPredictor(BasePredictor):

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            image = cv2.imread(img_path)
            # Extract bounding box
            bbox = pred[0, :4].int().numpy()
            '''
            pred:
            tensor([[376.2239,   0.0000, 630.9885, 316.1482,   0.9467,   2.0000],
            [ 59.8077,   0.0000, 342.1169, 199.6768,   0.9441,   6.0000],
            [289.6401, 263.8148, 526.1855, 626.9344,   0.9360,   5.0000],
            [ 74.6951, 168.5061, 354.1398, 568.8468,   0.9255,   8.0000]])
            '''
            bbox_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            for idx in range(len(pred)):
                try:
                    bbox = pred[idx, :4].int().numpy()
                    bbox_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    # 文檔使用OCR後處理
                    if int(pred[idx][-1]) in set(range(10)).difference({0, 8}):
                        bbox_image_white,_ = rotate_and_crop(bbox_image, foreground_color='white')
                        # bbox_image_white = correct_text_orientation(bbox_image_white)
                        text = pytesseract.image_to_string(bbox_image_white, lang='chi_sim')
                        text = text.lower().replace('\n', '').replace(' ', '')
                        chinese_ratio, english_ratio = calculate_ratio(text)
                        if (chinese_ratio + english_ratio) !=0:
                            pred[idx][-1] = int(closest_value(chinese_ratio, language_ratio))
                            pred[idx][-2] = 0.99

                    #document1b, document5b後處理
                    if int(pred[idx][-1])==0 or int(pred[idx][-1])==8:
                        bbox_image_white, aspect_ratio = rotate_and_crop(bbox_image, foreground_color='white')
                        major_color = ImageProcessor.get_major_color(bbox_image_white[idx])
                        if not ImageProcessor.is_dark_color(major_color):
                            pred[idx][-1] = int(closest_aspect_ratio_document(aspect_ratio))
                            pred[idx][-2] = 0.99

                    #線材後處理
                    if int(pred[idx][-1])==13 or int(pred[idx][-1])==14:
                        bbox_image_black, aspect_ratio = rotate_and_crop(bbox_image, foreground_color='black')
                        major_color = ImageProcessor.get_major_color(bbox_image_black[0])
                        if ImageProcessor.is_dark_color(major_color):
                            pred[idx][-1] = int(closest_aspect_ratio_plug(aspect_ratio))
                            pred[idx][-2] = 0.99
                except Exception as e:
                        print("An exception occurred during postprocessing:", str(e))
            
            result = Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred)
            pred_ls = pred.numpy().tolist()
            y_pred.append(pred_ls)

            results.append(result)
        return results

def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on multiple images."""
    model = r'/route/to/your/trained/yolo/model.pt'
    
    # source should be a list of image paths
    source = ImageProcessor.get_image_paths(r'/route/to/your/trained/valid/images')

    for img_path in source:
        args = dict(model=model, source=img_path)
        if use_python:
            from ultralytics import YOLO
            results = YOLO(model)(**args)
        else:
            predictor = DetectionPredictor(overrides=args)
            results = predictor.predict_cli()

if __name__ == '__main__':
    predict()

    with open('../y_preds.pkl', 'wb') as f:
        pickle.dump(y_pred, f)
