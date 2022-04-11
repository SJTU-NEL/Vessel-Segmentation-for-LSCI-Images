import math
import tensorflow as tf
from data_loader import *
from helper_function import *
from UNet_model import *
import cv2
from skimage.filters import threshold_otsu
from keras import backend as K
import time
import os


class Predict():
    def __init__(self,img_size,patch_size,overlap_size,data_path,result_path,model_path,model_name):

        self.Base = 16
        self.batchSize = 8
        self.dropout_rate = 0.4
        self.batch_norm = True
        self.img_size= img_size
        self.patch_size=patch_size
        self.overlap_size=overlap_size
        self.data_path=data_path
        self.result_path=result_path
        self.model_path=model_path
        self.model_name=model_name

    def predict(self,groundtruth=True):
        input_img = Input((256, 256, 1), name='img')

        test,self.Btest_name = load_test_image(self.data_path, self.img_size[0], self.img_size[1])
        patch_test = extract_patch_test_with_overlap(test, self.patch_size[0], self.patch_size[1],
                                                     self.overlap_size[0], self.overlap_size[1])
        print(patch_test.shape)
        if groundtruth:
            self.testMask = load_test_mask(self.data_path, self.img_size[0], self.img_size[1])


        dependencies = {
        'dice_coef_loss': dice_coef_loss,
        'dice_coef': dice_coef,
        'precision': precision,
        'recall': recall,
        'specificity': specificity
        }

        # end2end
        # model = ResNet(name='seg')
        # model.load_weights(os.path.join(self.model_path,self.model_name))
        # # 2 stage
        model = load_model(os.path.join(self.model_path,self.model_name), custom_objects=dependencies)
        # model = get_unet(input_img, self.Base, self.dropout_rate, self.batch_norm)

        # start_time = time.time()
        prediction = model.predict(patch_test, batch_size=self.batchSize)
        # end_time = time.time()

        self.full_pred = reconstract_image_from_patch_overlap(prediction, self.img_size[0], self.img_size[1],
                                                         self.patch_size[0],self.patch_size[1],
                                                         self.overlap_size[0],self.overlap_size[1])
        return self.full_pred


    def save(self,i):
        if not os.path.exists(os.path.join(self.result_path, 'B_predict')):
            os.makedirs(os.path.join(self.result_path, 'B_predict'))
        predict_image = self.full_pred[i, :, :, 0]
        ret, predict_image = cv2.threshold(predict_image*255, 30, 255, cv2.THRESH_BINARY)
        Image.fromarray((predict_image).astype('uint8'), "L").save(
            os.path.join(self.result_path, 'B_predict', self.Btest_name[i]))

    def evaluation(self,i):
        dice = dice_coef(self.testMask[i], self.full_pred[i])
        pre = precision(self.testMask[i], self.full_pred[i])
        sen = recall(self.testMask[i], self.full_pred[i])
        spe = specificity(self.testMask[i], self.full_pred[i])
        return [dice,pre,sen,spe]

    def mean_std(self,indicator):
        preimg_num=len(self.Btest_name)
        avr_indicator=np.mean(indicator)
        std_indicator = np.std(indicator, ddof=1) / math.sqrt(preimg_num)
        return avr_indicator,std_indicator

if __name__=="__main__":
    import os

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    # # # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 默认，不需要这句
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择ID为0的GPU
    # # sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    #
    # with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:

    with tf.compat.v1.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            predict=Predict(img_size=[565,584],patch_size=[80,80],overlap_size=[44,44],
                            data_path = './datasets', result_path = 'result_dir',
                            model_path = './models',
                            model_name='2_2Step_SMA1_UNet_128_128_150_1.h5')

            full_pred=predict.predict(groundtruth=True)
            dependencies=['dice','precision','sensitivity','specificity']
            sum_dependencies=[[],[],[],[]]
            # evaluate and save
            for i in range(len(full_pred)):
                num_dep=predict.evaluation(i)
                for k in range(len(num_dep)):
                    sum_dependencies[k].append(sess.run(num_dep[k]))
                predict.save(i)
            print("%d images have been predicted!!"%len(full_pred))
            # print evaluation results
            for j in range(len(sum_dependencies)):
                mean,std=predict.mean_std(sum_dependencies[j])
                print('avr %s:'%dependencies[j], mean)
                print('std %s'%dependencies[j], std)






