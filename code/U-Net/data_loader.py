import os
import tensorflow as tf
import random
import numpy as np
from skimage import exposure
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# random.seed(10)

def create_image_array_source(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        # print(image_name)
        # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
        if nr_of_channels == 1:  # Gray scale image -> MR image
            gray_img = load_img(os.path.join(image_path, image_name), color_mode="grayscale")
            std_img = (gray_img - np.mean(gray_img)) / np.std(gray_img) #standardization
            clahe_img = exposure.equalize_hist(std_img) #histogram equalization
            gamma_img = exposure.adjust_gamma(clahe_img, 1.2) #Gamma Correction on the input image
            normal_img = (gamma_img - np.min(gamma_img)) / (np.max(gamma_img) - np.min(gamma_img)) #rescaling/min-max normalization
            image = img_to_array(normal_img) #img_to_arry can be changed
            # image = image/255
            # image = resize(image, (560, 560, 1), mode='constant', preserve_range=True)
            image_dila = binary_dilation(image) #膨胀操作，将细血管加粗，overlap2 二选一 结果是bool型
        else:                   # RGB image -> street view
            image = np.array(Image.open(os.path.join(image_path, image_name)).convert('RGB'))
            #convert to RGB, otherwise image array will have 4 channels, including a transparent channel.
        # image = normalize_array(image)
        image_array.append(image) #Append object to the end of the list

    return np.array(image_array)

# turn target image to array
def create_image_array_target(image_list, image_path, nr_of_channels):
    image_array = []
    for image_name in image_list:
        # print(image_name)
        # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
        if nr_of_channels == 1:  # Gray scale image -> MR image
            # image = np.array(Image.open(os.path.join(image_path, image_name)).convert('L'))
            gray_img = load_img(os.path.join(image_path, image_name), grayscale=True)
            std_img = (gray_img - np.mean(gray_img)) / np.std(gray_img)
            clahe_img = exposure.equalize_hist(std_img)
            gamma_img = exposure.adjust_gamma(clahe_img, 1.2)
            normal_img = (gamma_img - np.min(gamma_img)) / (np.max(gamma_img) - np.min(gamma_img))
            # image = img_to_array(normal_img)
            image = normal_img[:, :, np.newaxis]
        else:                   # RGB image -> street view
            image = np.array(Image.open(os.path.join(image_path, image_name)).convert('RGB'))
            # image = np.array(load_img(os.path.join(image_path, image_name)))
        # image = normalize_array(image)
        # print(image.shape)
        image_array.append(image)

    return np.array(image_array)

# Get and resize train images and masks
def load_preprocess_image(path, im_height, im_width):
    # load images
    ids = next(os.walk(path + '/B_synthetic/'))[2]
    ids.sort()
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float64)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float64)

    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        gray_img = load_img(path + '/B_synthetic/' + id_, grayscale = True)
        std_img = (gray_img - np.mean(gray_img)) / np.std(gray_img)
        clahe_img = exposure.equalize_hist(std_img)
        gamma_img = exposure.adjust_gamma(clahe_img, 1.2)
        normal_img = (gamma_img - np.min(gamma_img)) / (np.max(gamma_img) - np.min(gamma_img))
        normal_img = img_to_array(normal_img)
        normal_img = resize(normal_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        # Save images
        # X[n] = gray_img / 255
        X[n] = normal_img

    # Load masks
    ids = next(os.walk(path + '/A_train/'))[2]
    ids.sort()
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        mask = img_to_array(load_img(path + '/A_train/' + id_, grayscale=True))
        mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
        y[n] = mask / 255

    print('Done!')
    return X, y

# for original UNet
def load_preprocess_image2(path, im_height, im_width):
    # load images
    ids = next(os.walk(path + '/A_images/'))[2]
    ids.sort()
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float64)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float64)

    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        gray_img = load_img(path + '/A_images/' + id_, grayscale = True)
        std_img = (gray_img - np.mean(gray_img)) / np.std(gray_img)
        clahe_img = exposure.equalize_hist(std_img)
        gamma_img = exposure.adjust_gamma(clahe_img, 1.2)
        normal_img = (gamma_img - np.min(gamma_img)) / (np.max(gamma_img) - np.min(gamma_img))
        normal_img = img_to_array(normal_img)
        normal_img = resize(normal_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        # Save images
        # X[n] = gray_img / 255
        X[n] = normal_img

        # Load masks
        mask = img_to_array(load_img(path + '/A_mask/'+ id_.strip('tif')+'gif', grayscale=True))
        mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)
        y[n] = mask / 255

    print('Done!')
    return X, y


# Get and resize test images and masks
def load_test_image(path, im_height, im_width):
    # load images
    ids = next(os.walk(path + '/B_test_20211002/'))[2]
    ids.sort()
    print(ids)
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float64)

    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        gray_img = load_img(path + '/B_test_20211002/' + id_, grayscale = True)
        std_img = (gray_img - np.mean(gray_img)) / np.std(gray_img)
        clahe_img = exposure.equalize_hist(std_img)
        gamma_img = exposure.adjust_gamma(clahe_img, 1.2)
        normal_img = (gamma_img - np.min(gamma_img)) / (np.max(gamma_img) - np.min(gamma_img))
        normal_img = img_to_array(normal_img)
        normal_img = resize(normal_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        # Save images
        X[n] = normal_img

    print('Done!')
    return X, ids


def load_test_mask(path, im_height, im_width):
    # load images
    ids = next(os.walk(path + '/B_manual/'))[2]
    ids.sort()
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float64)

    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        gray_img = load_img(path + '/B_manual/' + id_, grayscale = True)
        gray_img = img_to_array(gray_img)
        gray_img = resize(gray_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        # Save images
        X[n] = gray_img/255

    print('Done!')
    return X


def extract_patch_with_overlap(full_img, full_mask, patch_height, patch_width, overlap_height, overlap_width):
    full_img_height = full_img.shape[1]
    full_img_width = full_img.shape[2]
    N_patch_h = int((full_img_height - patch_height) / overlap_height) + 1
    N_patch_w = int((full_img_width - patch_width) / overlap_width) + 1
    N_patches = N_patch_h * N_patch_w * full_img.shape[0]
    patch_image = np.empty((N_patches, 256, 256, full_img.shape[3]))
    patch_mask = np.empty((N_patches, 256, 256, full_mask.shape[3]))

    patch_i = 0   # iter over the total numbe rof patches (N_patches)
    for full_image_i in range(full_img.shape[0]):  # loop over the full images
        for patch_j in range(N_patch_h):
            for patch_k in range(N_patch_w):
                patch_img = full_img[full_image_i, patch_j*overlap_height:patch_j*overlap_height+patch_height, patch_k*overlap_width:patch_k*overlap_width+patch_width, :]
                patch_mk = full_mask[full_image_i, patch_j*overlap_height:patch_j*overlap_height+patch_height, patch_k*overlap_width:patch_k*overlap_width+patch_width, :]
                patch_img = resize(patch_img, (256, 256, 1), mode='constant', preserve_range=True)
                patch_mk = resize(patch_mk, (256, 256, 1), mode='constant', preserve_range=True)
                patch_image[patch_i] = patch_img
                patch_mask[patch_i] = patch_mk
                patch_i += 1
                if patch_i%100==0:
                    print('{}%'.format(patch_i/N_patches*100))


    return patch_image, patch_mask


def extract_patch_test_with_overlap(full_img, patch_height, patch_width, overlap_height, overlap_width):
    full_img_height = full_img.shape[1]
    full_img_width = full_img.shape[2]
    N_patch_h = int((full_img_height - patch_height) / overlap_height) + 1
    N_patch_w = int((full_img_width - patch_width) / overlap_width) + 1
    N_patches = N_patch_h * N_patch_w * full_img.shape[0]
    patch_image = np.empty((N_patches, 256, 256, full_img.shape[3]))

    patch_i = 0   # iter over the total numbe rof patches (N_patches)
    for full_image_i in range(full_img.shape[0]):  # loop over the full images
        for patch_j in range(N_patch_h):
            for patch_k in range(N_patch_w):
                patch_img = full_img[full_image_i, patch_j*overlap_height:patch_j*overlap_height+patch_height, patch_k*overlap_width:patch_k*overlap_width+patch_width, :]
                patch_img = resize(patch_img, (256, 256, 1), mode='constant', preserve_range=True)
                patch_image[patch_i] = patch_img
                patch_i += 1

    return patch_image


def reconstract_image_from_patch_overlap(patch, full_image_h, full_image_w, patch_h, patch_w, overlap_h, overlap_w):
    patch = resize(patch, (patch.shape[0], patch_h, patch_w, patch.shape[3]), mode = 'constant', preserve_range = True)
    N_patch_h = int((full_image_h - patch_h) / overlap_h) + 1
    N_patch_w = int((full_image_w - patch_w) / overlap_w) + 1
    patch_per_image = N_patch_h * N_patch_w
    N_full_image = int(patch.shape[0]/patch_per_image)
    full_pred_image = np.empty((N_full_image, full_image_h, full_image_w, patch.shape[3]))
    full_img_i = 0
    patch_i = 0
    while patch_i < patch.shape[0]:
        single_pred_image = np.empty((full_image_h, full_image_w, patch.shape[3]))
        for j in range(N_patch_h):
            for k in range(N_patch_w):
                single_pred_image[j*overlap_h:j*overlap_h+patch_h, k*overlap_w:k*overlap_w+patch_w,:] = patch[patch_i]
                patch_i += 1
        full_pred_image[full_img_i] = single_pred_image
        full_img_i += 1
    return full_pred_image

