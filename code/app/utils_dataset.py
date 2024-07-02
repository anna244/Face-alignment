import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split

from . import settings
from .utils import get_rectangle, load_pts


def create_df(image_dirs):
    cache_file = settings.DF_CACHE_DIR + '/' + re.sub('[^a-z_]+', '_', ''.join(image_dirs)) + '.pickle'

    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    
    new_data = []
    for image_dir in image_dirs:
        for image_path in glob.glob(image_dir):
            image_dir_path = Path(image_path)
            landmark_path = image_dir_path.with_suffix('.pts')
            loaded_pts = load_pts(landmark_path)
            loaded_pts = np.array(loaded_pts, np.float64)
            rect_x1, rect_y1, rect_x2, rect_y2, quantity_point, quantity_faces = get_rectangle(str(image_dir_path), loaded_pts)
            n_landmarks = len(loaded_pts)

            new_data.append((image_path, loaded_pts, rect_x1, rect_y1, rect_x2,rect_y2,quantity_point, quantity_faces, n_landmarks))

    data_df = pd.DataFrame(new_data, columns = ['image_dir','real_landmarks','rect_x1', 'rect_y1', 'rect_x2','rect_y2','quantity_point', 'quantity_faces', 'n_landmarks'])

    # Удалим те строки из датафрейма, где n_landmarks < 68
    data_df = data_df[data_df['n_landmarks'] == 68]
    data_df = data_df[~data_df.isna().any(axis=1)]

    # Проверим на Nan
    assert data_df[data_df.isna().any(axis=1)].empty

    data_df.reset_index(drop=True, inplace=True)

    data_df.to_pickle(cache_file)
    return data_df


def get_data(train_image_dirs, test_image_dirs, use_val_dataset=True):
    train_df = create_df(train_image_dirs)
    val_df = None
    test_df = create_df(test_image_dirs)
    if use_val_dataset:
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=settings.RANDOM_STATE)
    train_df.reset_index(drop=True, inplace=True)

    if val_df is not None:
        val_df.reset_index(drop=True, inplace=True)

    test_df.reset_index(drop=True, inplace=True)

    return train_df, val_df, test_df

     
class ImagesDataset(torch.utils.data .Dataset):
  def __init__(self, df, image_size=None,  transform=None, target_transform=None):
    self.image_size = image_size or settings.IMAGE_SIZE
    self.transform = transform
    self.target_transform = target_transform

    self.df = df 

  def __getitem__(self, idx):
    image_path = self.df["image_dir"][idx]        

    image = Image.open(image_path).convert('RGB')

    real_landmarks = self.df["real_landmarks"][idx]

    x1 = int(self.df["rect_x1"][idx])
    y1 = int(self.df["rect_y1"][idx])
    x2 = int(self.df["rect_x2"][idx])
    y2 = int(self.df["rect_y2"][idx])

    rect = np.array([x1,y1,x2,y2])

    # landmarks_in_bbox = self.df["quantity_point"][idx]

    # old = landmarks_in_bbox

    # if landmarks_in_bbox != 68:
    #   rect = self.scale_rect(rect, image.size, percentage=0.3)
    #   landmarks_in_bbox = self.calculate_point_in_rect(rect,real_landmarks)
    resized_landmarks = None

    if self.transform:
      #crop image by bbox
      x1 = rect[0] 
      y1 = rect[1] 
      x2 = rect[2] 
      y2 = rect[3] 
      # # check что rect не за пределами изображения
      if x1 < 0:
          x1 = 0
      if x2 > image.size[0]:
          x2 = image.size[0]
      if y1 < 0:
          y1 = 0
      if y2 > image.size[1]:
          y2 = image.size[1]

      image = image.crop((x1,y1,x2,y2))
      image_crop = image

      cropped_landmarks = np.zeros((68,2))
      cropped_landmarks[:,0] = real_landmarks[:,0] - np.full(len(cropped_landmarks), x1)
      cropped_landmarks[:,1] = real_landmarks[:,1] - np.full(len(cropped_landmarks), y1)

      #resize image 
      # image = image.resize(self.image_size, self.image_size)
      image = self.transform(image)  #image.shape - CxHxW

      #Scale точек после resize
      scale_x = self.image_size / image_crop.size[0]  #image.size - WxH
      scale_y = self.image_size / image_crop.size[1]

      resized_landmarks = np.zeros((68,2))

      resized_landmarks[:,0] = cropped_landmarks[:,0]*scale_x
      resized_landmarks[:,1] = cropped_landmarks[:,1]*scale_y

      resized_landmarks = resized_landmarks
        
    # if self.target_transform:
    #     label = self.target_transform(label)

    return image, real_landmarks, resized_landmarks, rect
  
  def scale_rect(self, rect, image_size, percentage=0.20):
        x1 = rect[0]
        y1 = rect[1]
        x2 = rect[2]
        y2 = rect[3]

        h = y2 - y1
        w = x2 - x1

        new_x1 = int(x1 - percentage*w)
        new_x2 = int(x2 + percentage*w)
        new_y1 = int(y1 - percentage*h)
        new_y2 = int(y2 + percentage*h)

        if new_x1 < 0:
            new_x1 = 0
        if new_x2 > image_size[0]:
            new_x2 = image_size[0]
        if new_y1 < 0:
            new_y1 = 0
        if new_y2 > image_size[1]:
            new_y2 = image_size[1]

        new_rect = np.array([new_x1, new_y1, new_x2, new_y2])

        return new_rect

  def calculate_point_in_rect(self, box, landmarks):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    points = 0
    for point in landmarks:
      if (x1<point[0]<x2) and (y1<point[1]<y2):
          points+= 1 
      
    return(points)
  
  def __len__(self):
    return len(self.df)
  

def get_data_loaders(train_df, val_df, test_df):
    train_dataset = ImagesDataset(
        df=train_df,
        transform=settings.DATA_TRANSFORMS['train']
    )

    val_dataset = None
    if val_df is not None:
        val_dataset = ImagesDataset(
            df=val_df,
            transform=settings.DATA_TRANSFORMS['val']
        )

    test_dataset = ImagesDataset(
        df=test_df,
        transform=settings.DATA_TRANSFORMS['val']
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, num_workers=settings.NUM_WORKERS, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, num_workers=settings.NUM_WORKERS, pin_memory=True) if val_dataset else None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, num_workers=settings.NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, test_loader