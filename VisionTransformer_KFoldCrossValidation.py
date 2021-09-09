"""
 Keras Vision Transformers and CNN-based Keras models loading and training
 using Stratified K-Fold Cross Validation leaving p out.  
 Dataset is to be organised as one directory containing folders of images 
 for each class label.
 
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import glob, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

warnings.filterwarnings('ignore')
print('TensorFlow Version ' + tf.__version__) 

IMAGE_SIZE = 64
BATCH_SIZE = 8
EPOCHS = 100

# Main path of training/validation dataset containing other folders for each 
# class label
TRAIN_PATH = './Dataset'

# CSV file for training/validation organised as follows: 
#    image_id : label
DF_TRAIN = pd.read_csv('./trainval_5classes.csv', dtype='str')


# Define the number of classes and each class label 
classes = {0 : "weed",
           1 : "hors_type",
           2 : "betterave",
           3 : "persil",
           4 : "epinard"}

# DISPLAY LABELS DISTRIBUTION
plt.figure()
sns.countplot('label', data=DF_TRAIN)
plt.show()


def data_augment(image):
    """
    Data augmentation function applying random rotation, random resized crop, 
    random horizontal flip, colour jitters and rand augment. 

    Parameters
    ----------
    image : original image.

    Returns
    -------
    image : augmented image.

    """
    p_spatial = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype = tf.float32)
    
    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if p_spatial > .75:
        image = tf.image.transpose(image)
        
    # Rotates
    if p_rotate > .75:
        image = tf.image.rot90(image, k = 3) # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k = 2) # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k = 1) # rotate 90º
        
    # Pixel-level transforms
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower = .7, upper = 1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower = .8, upper = 1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta = .1)
        
    return image


# process data by normalising, rescaling and preprocessing with data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                          samplewise_center = True,
                                                          samplewise_std_normalization = True,                                              
                                                          preprocessing_function = data_augment)
# datagentest = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
#                                                           samplewise_center = True,
#                                                           samplewise_std_normalization = True)

DF_TRAIN['label'] = DF_TRAIN['label'].astype('str')
DF_TRAIN = DF_TRAIN[['image_id', 'label']]
# DF_TEST = DF_TEST[['image_path']]



##############################################################################
#   LOAD MODEL
############################################################################## 

# VISION TRANSFORMERS VIT (choose from keras vit model database)
from vit_keras import vit

vit_model = vit.vit_b16(
        image_size = IMAGE_SIZE,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 5)

# Efficientnet 
# from tensorflow.keras.applications import EfficientNetB1
# eff_model = EfficientNetB1(weights='imagenet',
#                            include_top=False,
#                            input_shape=(64,64,3),
#                            classifier_activation='softmax',
#                            classes=5)

# ResNet50
# from tensorflow.keras.applications import ResNet50
# res_model = ResNet50(weights='imagenet',
#                            include_top=False,
#                            input_shape=(64,64,3),
#                            classifier_activation='softmax',
#                            classes=5)

##############################################################################
#   CREATE MODEL
##############################################################################
def getModel(model_name):
    """
    Loads, builds and fine-tunes a model.

    Parameters
    ----------
    model_name : Name of the model.

    Returns
    -------
    model : a new model for training.

    """
    model = tf.keras.Sequential([
          vit_model,
          tf.keras.layers.Flatten(),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(11, activation = tfa.activations.gelu),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(5, 'softmax')  #last layer number of classes
          ],
          name = model_name)
  
    model.summary()
    return model

learning_rate = 1e-4
optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)


##############################################################################
#   CROSS VALIDATION 
##############################################################################

# Define number of folders to divide training dataset
# k_folds = (K-FOLD Cross Validation folders) + (one testing folder) 
k_folds = 5 + 1

# Using Stratified K-Fold Cross Validation, shuffling data 
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=20000)

df_folds = DF_TRAIN[['image_id']].copy()
df_folds.loc[:, 'label'] = DF_TRAIN['label'].copy()
# df_folds = df_folds.groupby('image_id').count()
# df_folds.loc[:, 'stratify_group'] = np.char.add(
#     "weed",
#     df_folds['image_id'].values.astype(str)
# )
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['label'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    

##############################################################################
# TRAINING MODEL USING K-FOLD CROSS VALIDATION LEAVING p OUT FOR VALIDATION
##############################################################################

from itertools import combinations
model_name = []

def train(folds,leavePout):
    """
    Trains a model using a K-Fold Cross Validation technique and evaluate the 
    model's performance on a separate testing dataset. 
    We use a stratified K-Fold CV leaving p folders as validation set. 
    
    Parameters
    ----------
    folds : K number of folders in which the dataset will be divided.
    leavePout : number of folders (1<P<K) to leave as validation set.

    Returns
    -------
    None.

    """
    
    j = 1

    list_folds = list(range(1, folds+1))
    list_train = list(list_folds.copy())

    comb = combinations(list_folds, leavePout)
    
    for i in comb:
        list_train = list(list_folds.copy())
        validdf = df_folds[df_folds['fold'] == 50]
        traindf = df_folds[df_folds['fold'] == 50]
        
        # TEST FOLDER IS SET TO 5 (6TH FOLDER) - REMAINING 5 FOLDERS ARE USED 
        # FOR CROSS VALIDATION
        testdf = df_folds[df_folds['fold'] == 5]
        
        for val_idx in range(len(i)):
            list_train.remove(i[val_idx])
            validdf = validdf.append(df_folds[df_folds['fold'] == i[val_idx]-1])    
        
        for train_idx in range(len(list_train)):
            traindf = traindf.append(df_folds[df_folds['fold'] == list_train[train_idx]-1])
            
       
        print("=========================================")
        print("====== K Fold Validation -- Fold "+str(i))
        print("=========================================")
        

        # Loading training, validation and testing data
        train_gen = datagen.flow_from_dataframe(dataframe = traindf,
                                        directory = TRAIN_PATH,
                                        x_col = 'image_id',
                                        y_col = 'label',
                                        subset = 'training',
                                        batch_size = BATCH_SIZE,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = True,
                                        class_mode = 'categorical',
                                        target_size = (IMAGE_SIZE, IMAGE_SIZE))

        valid_gen = datagen.flow_from_dataframe(dataframe = validdf,
                                        directory = TRAIN_PATH,
                                        x_col = 'image_id',
                                        y_col = 'label',
                                        # subset = 'validation',
                                        batch_size = BATCH_SIZE,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = False,
                                        class_mode = 'categorical',
                                        target_size = (IMAGE_SIZE, IMAGE_SIZE))
        
        test_gen = datagen.flow_from_dataframe(dataframe = testdf,
                                        directory = TRAIN_PATH,
                                        x_col = 'image_id',
                                        y_col = 'label',
                                        # subset = 'validation',
                                        batch_size = BATCH_SIZE,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = False,
                                        class_mode = 'categorical',
                                        target_size = (IMAGE_SIZE, IMAGE_SIZE))

   
        # Set model saved name
        model_name = 'model_ViT_5classes_leave1out_fold_'+str(j)
        
        # Load model
        model = getModel(model_name)

        try:
            model.load_weights(model_name)
        except:
            pass

        model.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
              metrics = ['accuracy'])

        STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
        STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_accuracy',
                                                  factor = 0.2,
                                                  patience = 2,
                                                  verbose = 1,
                                                  min_delta = 1e-4,
                                                  min_lr = 1e-6,
                                                  mode = 'max')
        
        # Set earlystopping for a patience of 6
        earlystopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy',
                                                  min_delta = 1e-5,
                                                  patience = 6,
                                                  mode = 'max',
                                                  restore_best_weights = True,
                                                  verbose = 1)

        checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = model_name+'.hdf5',
                                                  monitor = 'val_accuracy', 
                                                  verbose = 1, 
                                                  save_best_only = True,
                                                  save_weights_only = True,
                                                  mode = 'max')

        callbacks = [earlystopping, reduce_lr, checkpointer]

        history = model.fit(x = train_gen,
              steps_per_epoch = STEP_SIZE_TRAIN,
              validation_data = valid_gen,
              validation_steps = STEP_SIZE_VALID,
              epochs = EPOCHS,
              callbacks = callbacks)

        model.save(model_name)
        
        ######################################################################
        ###################### EVALUATE MODEL ################################
        ######################################################################

        # predicts on the testing folder (6th folder)
        predicted_classes = np.argmax(model.predict(test_gen), axis = 1)

    # predicted_classes = np.argmax(model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
        true_classes = test_gen.classes
        class_labels = list(test_gen.class_indices.keys())  

        confusionmatrix = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize = (64, 64))
        sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

        # prints and save classification report in working directory as CSV file
        report = classification_report(true_classes, predicted_classes, target_names = ['Weed (Class 0)','Hors-type (Class 1)', 'Betterave (Class 2)', 'Persil (Class 3)', 'Epinard (Class 4)'], output_dict=True)
        reportdf = pd.DataFrame(report).transpose()
        reportdf.to_csv('classification_report_fold'+str(j))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(history.epoch[-1]+1)
    
        # prints and save training/validation acuuracy and loss graphs
        fig=plt.figure(figsize=(15, 15))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
   
        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
  
        fig.savefig("Training_Validation_graphs/Training and Validation Accuracy-Loss EfficientNet B1 5folds 5classes Leave "+str(leavePout)+" out folds "+str(i)+".jpg",dpi=fig.dpi)

        # clear model before loading a new one to complete cross validation
        del model 
        
        j+=1
    

##############################################################################

# Calls training function with folds 
# params: folds=K cross vallidation (number of folders used for training/validation)
#         leaveKout=number of folders to be used as validation set

train(folds=k_folds-1, leavePout=3)


##############################################################################
####################### ADDITIONAL TESTING ###################################
##############################################################################

# load model and weights
modelx = getModel('testmodel')
modelx.load_weights('model_ViT_B16_5classes_leave2out_fold_10.hdf5')

modelx.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
              metrics = ['accuracy'])

# Path containing testing images 
TEST_PATH = './Testing/'
DF_Test = pd.read_csv('./testing.csv', dtype='str')

plt.figure()
sns.countplot('label', data=DF_Test)
plt.show()

# load testing images 
test_gen = datagen.flow_from_dataframe(dataframe = DF_Test,
                                        directory = TEST_PATH,
                                        x_col = 'image_id',
                                        y_col = 'label',
                                        # subset = 'validation',
                                        batch_size = BATCH_SIZE,
                                        seed = 1,
                                        color_mode = 'rgb',
                                        shuffle = False,
                                        class_mode = 'categorical',
                                        target_size = (IMAGE_SIZE, IMAGE_SIZE))

# Evaluate model 
modelx.evaluate(test_gen, batch_size=BATCH_SIZE)
predicted_classes = np.argmax(modelx.predict(test_gen), axis = 1)

# predicted_classes = np.argmax(model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (64, 64))
sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

classification_report(true_classes, predicted_classes, target_names = ['Weed (Class 0)','Hors-type (Class 1)', 'Betterave (Class 2)', 'Persil (Class 3)', 'Epinard (Class 4)'])
  