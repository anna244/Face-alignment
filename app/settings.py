import os
import pathlib

import torch
from torchvision.transforms import v2

ROOT_DIR = pathlib.Path(__file__).parent.resolve().parent
NUM_WORKERS = min(os.cpu_count(), 10)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
RANDOM_STATE = 42
FACE_DETECTOR_MODEL_PATH = str(ROOT_DIR / 'model_weights/mmod_human_face_detector.dat')
SHAPE_PREDICTOR_MODEL_PATH = str(ROOT_DIR / 'model_weights/shape_predictor_68_face_landmarks.dat')
DF_CACHE_DIR = str(ROOT_DIR / 'df_cache')
STUDY_MODEL_WEIGHT = str(ROOT_DIR / 'model_weights')

IMAGE_DIRS_300W_TRAIN = (
        str(ROOT_DIR / 'data/300W/train/*.jpg'), 
        str(ROOT_DIR / 'data/300W/train/*.png'),
)
IMAGE_DIRS_300W_TEST = (
        str(ROOT_DIR / 'data/300W/test/*.jpg'), 
        str(ROOT_DIR / 'data/300W/test/*.png'), 
)
IMAGE_DIRS_MENPO_TRAIN = (
        str(ROOT_DIR / 'data/Menpo/train/*.jpg'), 
        str(ROOT_DIR / 'data/Menpo/train/*.png'), 
)
IMAGE_DIRS_MENPO_TEST = (
        str(ROOT_DIR / 'data/Menpo/test/*.jpg'), 
        str(ROOT_DIR / 'data/Menpo/test/*.png'), 
)

IMAGE_SIZE = 48
BATCH_SIZE = 64
DATA_TRANSFORMS = {
    'train': v2.Compose([
        v2.ToImage(), # Convert to tensor, only needed if you had a PIL image
        v2.ToDtype(torch.uint8, scale=True),# optional, most input are already uint8 at this point
        v2.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        v2.RandomSolarize(threshold=192.0, p=0.7),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.ColorJitter(brightness=.5, hue=.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'val': v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}