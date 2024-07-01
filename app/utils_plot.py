import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import clear_output

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 15})


def plot_losses(train_losses, val_losses, train_RMSE, val_RMSE, thres=0.08, step=0.0001, common_val_auc_0_08=None, image_data = None):
    clear_output()

    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    fig. tight_layout ()
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label='train')
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label='val')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].set_title('Loss')


    num_data_train = len(train_RMSE)
    num_data_val = len(val_RMSE)

    coord_x_train = np.arange(0, thres + step, step)
    coord_y_train = np.array([np.count_nonzero(train_RMSE <= x) for x in coord_x_train]) / float(num_data_train)

    coord_x_val = np.arange(0, thres + step, step)
    coord_y_val = np.array([np.count_nonzero(val_RMSE <= x) for x in coord_x_val]) / float(num_data_val)

    axs[1].plot(coord_x_train,coord_y_train, label='train')
    axs[1].plot(coord_x_val,coord_y_val, label='val')
    axs[1].set_xlabel('Normilized Poin-to-Point error')
    axs[1].set_ylabel('Proportion of Image')
    axs[1].set_title(f'CDE_AUC_{image_data}={common_val_auc_0_08}')

    for ax in axs:
        ax.legend()

    plt.show()


def plot_ced_auc_test(RMSE, test_auc_0_08, image_data, thres=0.08, step=0.0001):

    num_data = len(RMSE)

    coord_x = np.arange(0, thres + step, step)
    coord_y = np.array([np.count_nonzero(RMSE <= x) for x in coord_x]) / float(num_data)

    plt.plot(coord_x,coord_y)
    plt.xlabel("Normilized Poin-to-Point error")
    plt.ylabel("Proportion of Image")
    plt.title(f"CED: AUC = {float('{:.4f}'.format(test_auc_0_08))},data = {image_data}")
    plt.grid(visible = True)

    return test_auc_0_08


def plot_CED_AUC_compare(RMSE_model, auc_model, RMSE_dlib, auc_dlib, thres=0.08, step=0.0001, type_model='ONet', image_data='300W'):

    clear_output()
    fig, ax = plt.subplots()

    num_data_my_model = len(RMSE_model)
    num_data_dlib = len(RMSE_dlib)

    coord_x_my_model = np.arange(0, thres + step, step)
    coord_y_my_model = np.array([np.count_nonzero(RMSE_model <= x) for x in coord_x_my_model]) / float(num_data_my_model)

    coord_x_dlib = np.arange(0, thres + step, step)
    coord_y_dlib = np.array([np.count_nonzero(RMSE_dlib <= x) for x in coord_x_dlib]) / float(num_data_dlib)

    ax.plot(coord_x_my_model,coord_y_my_model, label= f'{type_model}:{auc_model:.4f}')
    ax.plot(coord_x_dlib,coord_y_dlib, label=f'DLIB:{auc_dlib:.4f}')
    ax.set_xlabel('Normilized Poin-to-Point error')
    ax.set_ylabel('Proportion of Image')
    ax.set_title(f'CDE_AUC_{image_data}')

    ax.legend()

    plt.show()