# Vision Transformers ViT Keras
Vision Transformer ViT model used in agricultural application using K-Fold Cross Validation

In this repository, the code used to load and train Keras Vision Transformer and CNN-based models using a K-Fold cross validation technique leaving p-out is presented.
The models used were pre-trained on the ImageNet datasets. 
The python script VisionTransformer_KFoldCrossValidation.py is used to load, build and train Keras models of ViT and other CNN-based models using a K-Fold cross validation leaving P out evaluation method. The code has been tested with Tensorflow 2.4.1.

# 1. Dataset
The image dataset used is also presented, categorised in 5 classes: weeds, beet (red leaves), off-type beet (green leaves), parsley and spinach. 
The dataset directory is to be organised as follows: 

![directory_tree](https://user-images.githubusercontent.com/45753185/132690648-427968ac-36b8-4b3c-89ff-4ebf0ca26292.png)

The CSV annotation file should be placed in the working directory and organised in the following format: 
![CSV_format](https://user-images.githubusercontent.com/45753185/132691044-d5f9ce02-f149-4345-a4c0-c05da473d28a.PNG)

# 2. Configuration

<img src="https://user-images.githubusercontent.com/45753185/132690908-b7d083e8-2b3b-49a3-9374-b7a0c199cea7.png" height="100" width="350">

The size of input images (if rectangular images are used, make sure the image size is the maximum of the width and height), batch size and number of training epochs are to be defined by the user. 
TRAIN_PATH is the path (relative or absolute) of the dataset directory, containing image folders of each class label. 

DF_TRAIN reads the CSV annotation file. 

The number and name of classes are to be defined as ease based on your application. The class numbers should be the same as the label numbers used in the CSV file.

<img src="https://user-images.githubusercontent.com/45753185/132691108-e95beeff-263d-47cc-aa45-4a75f65f8f5b.png" height="80" width="180">

# 3. Train

The created model is trained by calling the function train(folds, leavePout). Where folds represents the number of folders to use for cross validation (K-Fold) and leavePout is the number of folders to use as validation set. 
The dataset is divided equally in (K+1) folders. One folder is kept for testing while the remaining K folders are used for training/validation. Varying the value of leavePout will vary the number of training images and validation images and thus can be used to test the model's performance with different number of training images. 
The trained model will be saved in the working directory and the figures will be saved in “./Training_Validation_graphs/”.

