import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
from keras.preprocessing.image import img_to_array, load_img
from skimage import exposure
from skimage.transform import resize
from skimage.morphology import binary_dilation
import random
import math
import cv2
# from skimage.io import imread


def load_data(nr_of_channels, batch_size=1, nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False,
              use_supervised_learning=False, REAL_LABEL=1.0, SM=0, overlap=[]):
    """
        To load and extract patches with overlaps of images.

        Extracting patched with overlaps include 3 patterns:
        (0)default --SM=0
        (1)with blood vessel dilation --SM=1
        (2)with picking thick blood vessel by threshold value method -SM=2

        nr_of channels: if =1, image is grayscale; if =3, image includes three channels, RGB.
        batch_size: the number of selected samples for each training
        nr_A_train_imgs: training imgs in domain A
        nr_B_train_imgs: training imgs in domain B
        nr_A_test_imgs: testing imgs in domain A
        nr_B_test_imgs: testing imgs in domain B

    """

    datasetA_path = os.path.join('datasets/DRIVE2LSCI',subfolder,'A')  # 40 DRIVE (565*584)
    datasetB_path = os.path.join('datasets/DRIVE2LSCI', subfolder, 'B') # 140 LSCI  (280*400)
    test_proportion=0.2

    np.random.seed(seed=12345)
    datasetA_image_names = os.listdir(datasetA_path)
    np.random.shuffle(datasetA_image_names)
    testA_image_names = datasetA_image_names[:math.ceil(test_proportion*len(datasetA_image_names))]
    trainA_image_names= datasetA_image_names[:math.ceil(test_proportion*len(datasetA_image_names))]

    datasetB_image_names = os.listdir(datasetB_path)
    np.random.shuffle(datasetB_image_names)
    testB_image_names = datasetB_image_names[:math.ceil(test_proportion*len(datasetB_image_names))]
    trainB_image_names= datasetB_image_names[:math.ceil(test_proportion*len(datasetB_image_names))]

    # 写入测试集
    with open('testB.txt','w') as x:
        for i in testB_image_names:
            x.write(i+'\n')
        x.close()

    # if generator:
    #     return data_sequence(trainA_path, trainB_path, trainA_image_names, trainB_image_names, batch_size=batch_size)
    #     # D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL)
    # else: # A: source  B: target

    trainA_images = create_image_array_source(trainA_image_names, datasetA_path, nr_of_channels,SM)
    trainB_images = create_image_array_target(trainB_image_names, datasetB_path, nr_of_channels)
    testA_images = create_image_array_source(testA_image_names, datasetA_path, nr_of_channels,SM)
    testB_images = create_image_array_target(testB_image_names, datasetB_path, nr_of_channels)

    if SM==0 or SM==1:
        trainA_patchs = extract_patch_with_overlap(trainA_images, 256, 256, overlap[0], overlap[1])
        testA_patchs = extract_patch_with_overlap(testA_images, 256, 256, overlap[0], overlap[1])
    elif SM == 2:
        trainA_patchs = extract_patch_with_overlap2(trainA_images, 256, 256, overlap[0], overlap[1])
        testA_patchs = extract_patch_with_overlap2(testA_images, 256, 256, overlap[0], overlap[1])
    trainB_patchs = extract_patch_with_overlap(trainB_images, 256, 256, overlap[2], overlap[3])
    testB_patchs = extract_patch_with_overlap(testB_images, 256, 256, overlap[2], overlap[3])

    print('trainA:',trainA_patchs.shape)
    print('trainB:', trainB_patchs.shape)
    print('testA:', testA_patchs.shape)
    print('testB:', testB_patchs.shape)

    return {"trainA_images": trainA_patchs, "trainB_images": trainB_patchs,
            "testA_images": testA_patchs, "testB_images": testB_patchs,
            "trainA_image_names": trainA_image_names,
            "trainB_image_names": trainB_image_names,
            "testA_image_names": testA_image_names,
            "testB_image_names": testB_image_names}


# turn source image to array
def create_image_array_source(image_list, image_path, nr_of_channels,SM):
    image_array = []
    for image_name in image_list:
        # print(image_name)
        # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
        if nr_of_channels == 1:  # Gray scale image -> MR image
            gray_img = load_img(os.path.join(image_path, image_name), grayscale=True)
            # gray_img=Image.open(os.path.join(image_path, image_name)).convert('Grayscale')
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
        if SM==0:
            image_array.append(image)  # Append object to the end of the list
        elif SM==1:
            image_array.append(image_dila)

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


# # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
# def normalize_array(array):
#     array = array / 127.5 - 1
#     return array


class data_sequence(Sequence):
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        for image_name in image_list_A:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):  # , use_multiscale_discriminator, use_supervised_learning):if loop_index + batch_size >= min_nr_imgs:
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        real_images_A = create_image_array_source(batch_A, '', 3)
        real_images_B = create_image_array_target(batch_B, '', 3)

        return real_images_A, real_images_B  # input_data, target_data


def extract_patch(full_img, patch_width, patch_height):
    full_img_height = full_img.shape[1]
    full_img_width = full_img.shape[2]
    N_patch_h = int(full_img_height / patch_height)
    N_patch_w = int(full_img_width / patch_width)
    N_patches = N_patch_h * N_patch_w * full_img.shape[0]
    patch_image = np.empty((N_patches, 256, 256, full_img.shape[3]))

    patch_i = 0   # iter over the total numbe rof patches (N_patches)
    for full_image_i in range(full_img.shape[0]):  # loop over the full images
        for patch_j in range(N_patch_h):
            for patch_k in range(N_patch_w):
                patch_img = full_img[full_image_i, patch_j*patch_height:(patch_j+1)*patch_height, patch_k*patch_width:(patch_k+1)*patch_width, :]
                patch_img = resize(patch_img, (256, 256, 1), mode='constant', preserve_range=True)
                patch_image[patch_i] = patch_img
                patch_i += 1

    return patch_image

# segment image into patches with overlap: large blood vessel
# full_img.shape=[30,584,565,1],[image number, image height, image width, channel]
def extract_patch_with_overlap2(full_img, patch_width, patch_height, overlap_width, overlap_height):
    full_img_height = full_img.shape[1]
    full_img_width = full_img.shape[2]
    N_patch_h = int((full_img_height - patch_height) / overlap_height) + 1
    N_patch_w = int((full_img_width - patch_width) / overlap_width) + 1
    N_patches = N_patch_h * N_patch_w * full_img.shape[0] #
    # patch_image = np.empty((N_patches, 256, 256, full_img.shape[3]))
    patch_image = []
    # patch_i = 0   # iter over the total numbe rof patches (N_patches)
    for full_image_i in range(full_img.shape[0]):  # loop over the full images
        for patch_j in range(N_patch_h):
            for patch_k in range(N_patch_w):
                # cv2.imshow('full_img', full_img[full_image_i,:,:,:])
                # cv2.waitKey(50)
                patch_img = full_img[full_image_i, patch_j*overlap_height:patch_j*overlap_height+patch_height, patch_k*overlap_width:patch_k*overlap_width+patch_width, :]
                patch_img = resize(patch_img, (256, 256, 1), mode='constant', preserve_range=True)

                threshold=0.1
                # array_to_decide_threshold=np.zeros((256,256))
                # for i in range(0,256):
                #     for j in range(0,256):
                #         if patch_img[i,j] <=threshold:
                #             array_to_decide_threshold[i,j]=1
                #             cv2.imshow(' array_to_decide_threshold',  array_to_decide_threshold)
                #             cv2.waitKey(50)
                #
                blood_pixel=np.sum(patch_img >=threshold) / (256 * 256)
                # cv2.imshow('patch_img', patch_img)
                # cv2.waitKey(50)
                if blood_pixel > 0.13: #judge large blood vessel
                    # cv2.imshow('image', patch_img)
                    # cv2.waitKey()
                    # patch_image[patch_i] = patch_img
                    # patch_i += 1
                    patch_image.append(patch_img)

    return np.array(patch_image)

# extract patch of 256*256 when SM=0/1
def extract_patch_with_overlap(full_img, patch_width, patch_height, overlap_width, overlap_height):
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
                patch_img = resize(patch_img, (256, 256, 1), mode='constant', order=2, preserve_range=True)

                # print(np.sum(patch_img == 1)/(256*256))
                patch_image[patch_i] = patch_img
                patch_i += 1

    return patch_image


def reconstract_image_from_patch(patch, full_image_w, full_image_h, patch_w, patch_h):
    patch = resize(patch, (patch.shape[0], patch_h, patch_w, patch.shape[3]), mode = 'constant', preserve_range = True)
    N_patch_h = int(full_image_h/patch_h)
    N_patch_w = int(full_image_w/patch_w)
    patch_per_image = N_patch_h * N_patch_w
    N_full_image = int(patch.shape[0]/patch_per_image)
    full_pred_image = np.empty((N_full_image, full_image_h, full_image_w, patch.shape[3]))
    full_img_i = 0
    patch_i = 0
    while patch_i < patch.shape[0]:
        single_pred_image = np.empty((full_image_h, full_image_w, patch.shape[3]))
        for j in range(N_patch_h):
            for k in range(N_patch_w):
                single_pred_image[j*patch_h:(j+1)*patch_h, k*patch_w:(k+1)*patch_w,:] = patch[patch_i]
                patch_i += 1
        full_pred_image[full_img_i] = single_pred_image
        full_img_i += 1
    return full_pred_image


def reconstract_image_from_patch_overlap(patch, full_image_w, full_image_h, patch_w, patch_h, overlap_w, overlap_h):
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
        for j in [0]:
            for k in [1]:
                single_pred_image[j * overlap_h:j * overlap_h + patch_h, k * overlap_w:k * overlap_w + patch_w, :] = \
                patch[1]
        for j in [1]:
            for k in [1]:
                single_pred_image[j * overlap_h:j * overlap_h + patch_h, k * overlap_w:k * overlap_w + patch_w, :] = \
                    patch[4]
        full_pred_image[full_img_i] = single_pred_image
        full_img_i += 1
    return full_pred_image


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


if __name__ == '__main__':
    load_data(nr_of_channels=1)