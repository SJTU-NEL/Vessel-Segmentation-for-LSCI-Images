import random
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image


def plot_history(History):
    plt.figure(figsize=(4, 4))
    plt.title("Learning curve")
    plt.plot(History.history["loss"], label="loss")
    plt.plot(History.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(History.history["val_loss"]),
             np.min(History.history["val_loss"]),
             marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.legend()

    plt.figure(figsize=(4, 4))
    plt.title("dice coef")
    plt.plot(History.history["dice_coef"], label="dice_coef")
    plt.plot(History.history["val_dice_coef"], label="val_dice_coef")
    plt.xlabel("Epochs")
    plt.ylabel("metrics")
    plt.legend()

    plt.figure(figsize=(4, 4))
    plt.title("precision")
    plt.plot(History.history["precision"], label="precision")
    plt.plot(History.history["val_precision"], label="val_precision")
    plt.xlabel("Epochs")
    plt.ylabel("metrics")
    plt.legend()

    plt.figure(figsize=(4, 4))
    plt.title("recall")
    plt.plot(History.history["recall"], label="recall")
    plt.plot(History.history["val_recall"], label="val_recall")
    plt.xlabel("Epochs")
    plt.ylabel("metrics")
    plt.legend()

    plt.figure(figsize=(4, 4))
    plt.title("specificity")
    plt.plot(History.history["specificity"], label="specificity")
    plt.plot(History.history["val_specificity"], label="val_specificity")
    plt.xlabel("Epochs")
    plt.ylabel("metrics")
    plt.legend()
    plt.show()


def plot_sample(X, y, preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    # if has_mask:
    #    ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('image')

    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    ax[1].set_title('mask')

    ax[2].imshow(preds[ix].squeeze(), cmap='gray')
    # if has_mask:
    #    ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted')

    plt.show()


def visualize_patches(image, img_h, img_w):
    fig, ax = plt.subplots(1, 4, figsize=(img_h, img_w))
    ax[0].imshow(image[0, :, :, 0], plt.cm.gray)
    ax[1].imshow(image[1, :, :, 0], plt.cm.gray)
    ax[2].imshow(image[2, :, :, 0], plt.cm.gray)
    ax[3].imshow(image[3, :, :, 0], plt.cm.gray)
