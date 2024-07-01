import dlib
import numpy as np
import torch
from tqdm.notebook import tqdm

import settings

from .utils import (ced_auc, count_avg_norm_dist, count_avg_norm_dist_butch,
                    upsample_landmarks)
from .utils_dataset import ImagesDataset
from .utils_plot import plot_losses


def shape_to_np(shape):
    coords = np.zeros((68, 2))
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def train(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs, image_data):
    train_losses = []
    val_losses = []
    common_train_auc_0_08 = []
    common_val_auc_0_08 = []
    common_train_RMSE = []
    common_val_RMSE = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_auc_0_08, train_RMSE = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        if val_loader:
          val_loss, val_auc_0_08, val_RMSE = validation_epoch(
              model, criterion, val_loader,
              tqdm_desc=f'Validating {epoch}/{num_epochs}'
          )
        else:
          val_loss = 0
          val_auc_0_08 = 0
          val_RMSE = 0

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        common_train_auc_0_08 += [train_auc_0_08]
        common_val_auc_0_08 += [val_auc_0_08]
        common_train_RMSE += [train_RMSE]
        common_val_RMSE += [val_RMSE]

        plot_losses(train_losses, val_losses, common_train_RMSE[-1], common_val_RMSE[-1], thres=0.08, step=0.0001, common_val_auc_0_08 = common_val_auc_0_08[-1], image_data = image_data)

        lr = scheduler.optimizer.param_groups[0]['lr']

        print(f'''epoch {epoch},\n
                train_losses: {train_loss:.4f},\n
                val_losses: {val_loss:.4f},\n 
                common_train_auc_0_08: {train_auc_0_08:.4f},\n 
                common_val_auc_0_08: {val_auc_0_08:.4f},\n
                lr: {lr}''')

    return train_losses, val_losses, common_train_auc_0_08, common_val_auc_0_08, common_train_RMSE, common_val_RMSE


def training_epoch(model, optimizer, criterion, train_loader, tqdm_desc):
    train_loss = 0.0
    RMSE = []

    model.train()
    for images, real_landmarks, resized_landmarks, rect in tqdm(train_loader, desc=tqdm_desc):
        images = images.to(settings.DEVICE)  # images: batch_size x num_channels x height x width
        resized_landmarks = resized_landmarks.to(settings.DEVICE)  # resized_landmarks: batch_size x 68X2

        optimizer.zero_grad()
        pred_landmarks = model(images)  # pred_landmarks: batch_size x 68X2
        pred_landmarks = pred_landmarks.reshape(-1,68,2).double()

        loss = criterion(pred_landmarks, resized_landmarks)

        images = images.cpu()
        real_landmarks = real_landmarks.cpu()
        pred_landmarks = pred_landmarks.cpu().detach()
        rect = rect.cpu()

        #upscale pred_landmarks
        upsamle_pred_landmarks = upsample_landmarks(images, pred_landmarks, rect)

        loss.backward()
        optimizer.step()

        # calculate loss 
        train_loss += loss.item() * images.shape[0]

        # calculate metric 
        avg_norm_dist = count_avg_norm_dist_butch(real_landmarks.numpy(), upsamle_pred_landmarks, rect.numpy())
        RMSE = np.concatenate((RMSE, avg_norm_dist))

    train_loss /= len(train_loader.dataset)

    train_auc_0_08 = ced_auc(RMSE, thres=0.08)

    train_RMSE = RMSE

    return train_loss, train_auc_0_08, train_RMSE

@torch.no_grad()
def validation_epoch(model, criterion, val_loader, tqdm_desc):
    val_loss  = 0.0
    RMSE = []

    model.eval()
    for images, real_landmarks, resized_landmarks, rect in tqdm(val_loader, desc=tqdm_desc):
        images = images.to(settings.DEVICE)  
        resized_landmarks = resized_landmarks.to(settings.DEVICE)

        pred_landmarks = model(images) 
        pred_landmarks = pred_landmarks.reshape(-1,68,2)

        loss = criterion(pred_landmarks, resized_landmarks)

        images = images.cpu()
        real_landmarks = real_landmarks.cpu()
        pred_landmarks = pred_landmarks.cpu().detach()
        rect = rect.cpu()

        #upscale pred_landmarks
        upsamle_pred_landmarks = upsample_landmarks(images, pred_landmarks, rect)

        val_loss += loss.item() * images.shape[0]

        # calculate metric 
        avg_norm_dist = count_avg_norm_dist_butch(real_landmarks.numpy(), upsamle_pred_landmarks, rect.numpy())
        RMSE = np.concatenate((RMSE, avg_norm_dist))

    val_loss /= len(val_loader.dataset)

    # auc_0_08 = ced_auc(RMSE, thres=0.08, draw = True)
    val_auc_0_08 = ced_auc(RMSE, thres=0.08)

    val_RMSE = RMSE

    return val_loss, val_auc_0_08, val_RMSE


@torch.no_grad()
def model_inference(model, val_loader, tqdm_desc=''):
    model.eval()
    RMSE = []
    pred_landmarks_model = None

    for images, real_landmarks, resized_landmarks, rect in tqdm(val_loader, desc=tqdm_desc):
        images = images.to(settings.DEVICE)  
        resized_landmarks = resized_landmarks.to(settings.DEVICE)

        pred_landmarks = model(images) 
        pred_landmarks = pred_landmarks.reshape(-1,68,2)

        images = images.cpu()
        real_landmarks = real_landmarks.cpu()
        pred_landmarks = pred_landmarks.cpu().detach()
        rect = rect.cpu()
        
        #upscale pred_landmarks
        upsamle_pred_landmarks = upsample_landmarks(images, pred_landmarks, rect)

        # calculate metric 
        avg_norm_dist = count_avg_norm_dist_butch(real_landmarks.numpy(), upsamle_pred_landmarks, rect.numpy())


        RMSE = np.concatenate((RMSE, avg_norm_dist))

        if pred_landmarks_model is not None:
            pred_landmarks_model = np.concatenate((pred_landmarks_model, upsamle_pred_landmarks))
        else:
            pred_landmarks_model = upsamle_pred_landmarks

    test_auc_0_08 = ced_auc(RMSE, thres=0.08)

    return RMSE, test_auc_0_08, pred_landmarks_model
    

def test_dlib(df):
    predictor = dlib.shape_predictor(settings.SHAPE_PREDICTOR_MODEL_PATH)
    test_dataset_dlib = ImagesDataset(df, image_size = None, transform=None)             

    RMSE_dlib = []
    for index in tqdm(range(len(test_dataset_dlib))):
        image, real_landmarks, _ , rect = test_dataset_dlib[index]

        dlib_rect = dlib.rectangle(left=rect[0], top=rect[1], right=rect[2], bottom=rect[3]) 
        pred_landmarks = predictor(np.array(image), dlib_rect)

        pred_landmarks = shape_to_np(pred_landmarks)

        avg_norm_dist = count_avg_norm_dist(pred_landmarks, real_landmarks, rect)
        RMSE_dlib.append(avg_norm_dist)

    RMSE_dlib = np.sort(RMSE_dlib)

    ced_auc_0_08_dlib = ced_auc(RMSE_dlib,thres=0.08, step=0.0001)

    print(f'''RMSE_dlib {RMSE_dlib},\n
        ced_auc_0_08_dlib: {ced_auc_0_08_dlib:.4f}''')

    return ced_auc_0_08_dlib, RMSE_dlib, pred_landmarks 