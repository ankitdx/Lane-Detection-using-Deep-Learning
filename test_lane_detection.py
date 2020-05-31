from tensorflow.keras.models import load_model
import cv2
import numpy as np
from LaneDetection.calculate_dice_coefficient import dice

# Paths for Model test images, mask and path for writing the prediction
model_path = '/home/ankit/PyCharmWorkSpace/LaneDetection/Models/LaneDetector_vgg_unet.h5'
test_images_path = '/home/ankit/Datasets/Lane_Detection/binary_lane_bdd/objects/test_data.npz'
test_labels_path = '/home/ankit/Datasets/Lane_Detection/binary_lane_bdd/objects/test_data_labels.npz'
path2save = '/home/ankit/Datasets/Lane_Detection/binary_lane_bdd/objects/output_images/'

test_images = np.load(test_images_path)['arr_0']
test_labels = np.load(test_labels_path)['arr_0']
model = load_model(model_path)

dice_list = []
for i in range(len(test_images)):
    im = test_images[i]
    gt = test_labels[i]
    overlay_mask = np.zeros(im.shape,dtype='uint8')
    im_copy = im.copy()
    im = im.astype('float32')/255
    im_ = np.expand_dims(im, axis=0)
    probs = model.predict(im_)

    prediction = 255*(probs[0] > 0.2).astype("uint8")
    dice_score = dice(prediction, gt)
    print("dice score: %.2f" % dice_score)
    dice_list.append(dice_score)
    cv2.imwrite(path2save+str(i)+'.png', im_copy)
    cv2.imwrite(path2save+str(i) + '_gt.png', 255*gt)
    cv2.imwrite(path2save+str(i) + '_predict.png', prediction)
    cv2.imshow("input", im)
    cv2.imshow("out", prediction)
    cv2.waitKey()
    print('done')

print("Mean Dice score %.2f" % np.mean(dice_list))
