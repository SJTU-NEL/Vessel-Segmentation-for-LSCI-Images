import tensorflow as tf
from tensorflow.keras.optimizers import *
from sklearn.model_selection import train_test_split
from data_loader import *
from UNet_model import *
from helper_function import *
import tensorflow as tf
from tensorflow.keras import backend as K



# config=tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess=tf.compat.v1.Session(config=config)
# K.set_session(sess)

class UNet_Train():
    def __init__(self,epochNum,data_path,img_size,patch_size,overlap_size,pattern_num):
        self.Base = 16
        self.batchSize = 8
        self.epochNum = epochNum
        self.LR = 0.0001
        self.dropout_rate = 0.4
        self.batch_norm = True
        self.data_path = data_path
        self.img_size=img_size
        self.patch_size=patch_size
        self.overlap_size=overlap_size
        self.pattern_num=pattern_num
        self.pattern_list=['Baseline','2StepOnly','2Step_SMA1','2Step_SMA2']

    def main(self):
        model_name='_'.join([str(self.pattern_num),self.pattern_list[self.pattern_num],'UNet',
                            str(self.patch_size[0]),str(self.patch_size[1]),str(self.epochNum)])
        X, y = load_preprocess_image(self.data_path,self.img_size[0],self.img_size[1]) # synthetic images: 256*256, fundus images: 565*584
        patch_image, patch_mask = extract_patch_with_overlap(X, y, self.patch_size[0], self.patch_size[1],
                                                                 self.overlap_size[0], self.overlap_size[1])
        # Split train and valid
        X_train, X_valid, y_train, y_valid = train_test_split(patch_image, patch_mask, test_size=0.2, random_state=2018)
        print(X_train.shape)
        print(X_valid.shape)

        input_img = Input((256, 256, 1), name='img')
        model = get_unet(input_img, self.Base, self.dropout_rate, self.batch_norm)
        model.compile(optimizer=Adam(lr=self.LR), loss=dice_coef_loss, metrics=[dice_coef, precision, recall, specificity])

        self.History = model.fit(
            X_train, y_train,
            batch_size=self.batchSize,
            epochs=self.epochNum,
            validation_data=(X_valid, y_valid),
            verbose=1)

        plot_history(self.History)
        model.save(os.path.join(os.getcwd(),'models',model_name+'_new.h5'))

if __name__ == '__main__':
    import os
    # 分配gpu
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    # # # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 默认，不需要这句
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 选择ID为0的GPU

    with tf.compat.v1.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            unet_train=UNet_Train(epochNum=150,data_path='./datasets/0-baseline',
                                  img_size=[565,584],patch_size=[80,80],overlap_size=[44,44],
                                  pattern_num=1)
            unet_train.main()



