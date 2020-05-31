import glob
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define Size of images
imwidth = 640
imheight = 384
imdepth = 3

# Number Of classes in labels
classes = 2
data_shape = imwidth*imheight

# Path to read Tiles for Label and Data
data_path = '/home/ankit/Datasets/Lane_Detection/binary_lane_bdd/Images_cleaned/'
label_path = '/home/ankit/Datasets/Lane_Detection/binary_lane_bdd/Labels/'
objects_dir = '/home/ankit/Datasets/Lane_Detection/binary_lane_bdd/objects/'


# Function to create label array for binary classification
def binarylab(labels):
    
    # Define an Empty Array
    x = np.zeros([imheight, imwidth, classes], dtype="uint8")
    
    # Read Each pixel label and put it into corresponding label plane
    for i in range(imheight):
        for j in range(imwidth):
            x[i, j, labels[i][j]] = 1
    
    return x


def whitening(im):
            
    im = im.astype("float32")              
    for i in range(np.shape(im)[2]):                                
        im[:,:,i] = (im[:,:,i]- np.mean(im[:,:,i]))/(np.std(im[:,:,i])+1e-9)            
    return im


def prepareDataSet():
        
    labelpaths = sorted(glob.glob(label_path + "/*.jpg"))
    
    # Create Empty Lists to store Image and Label Data
    data = []
    label = []
        
    for i in range(len(labelpaths)):
        # for i in range(50):
        tlp = labelpaths[i]
        tilepath = tlp.split("/")
        tdp = data_path+tilepath[-1]
        
        # Read Images
        im = cv2.imread(tdp)
        lab = (cv2.imread(tlp)[:, :, 0] > 0).astype("uint8")
        
        if im is not None:

            im = cv2.resize(im, (imwidth, imheight))
            lab = cv2.resize(lab, (imwidth, imheight))

            data.append(im)
                        
            # Convert label into binary form
            # lab = binarylab(lab)
            lab = np.expand_dims(lab, axis=-1)
            # Append Images into corresponding List
            label.append(lab)
            
            print('\n'+tdp)
        else:
            print("error: "+tdp)
       
    return data, label


data, label = prepareDataSet()

# data = prepareDataSetAE()
# Store Data to the directory

X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.1)

np.savez_compressed(objects_dir+'train_data.npz', X_train)
np.savez_compressed(objects_dir+'train_data_labels.npz', Y_train)
np.savez_compressed(objects_dir+'test_data.npz', X_test)
np.savez_compressed(objects_dir+'test_data_labels.npz', Y_test)
print("Done")

#================================================================================Resusable Code===============================================================================
#lab = cv2.resize(lab,(imwidth,imheight),interpolation = cv2.INTER_NEAREST)
#lab = cv2.threshold(lab,0,255,cv2.THRESH_BINARY)
#lab = cv2.normalize(lab[1], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#cv2.imshow("im",lab)
#cv2.waitKey(-1)
#data.append(im)

#cv2.imshow("im",im)
#cv2.waitKey(-1)
#cv2.imshow("lab",lab[:,:,0])
#cv2.waitKey(-1)

#     #label = np.reshape(label,(len(paths),data_shape,classes))
#     lab1 = label[0]
#     #lab1 = np.reshape(lab1,(128,128,2))
#     lab1= cv2.normalize(lab1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     lab1 = np.transpose(lab1, [1,2,0])
#     cv2.imshow("",(lab1[:,:,0]))
#     waitKey()

#im = normalized(im)
#im = cv2.resize(im,(imwidth,imheight),interpolation = cv2.INTER_CUBIC)        

#     x= cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     cv2.imshow("lab",x[:,:,1])
#     cv2.waitKey(-1)