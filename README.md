# Isolated-Printed-Arabic-Characheter-Recognition_tenserflow
The training tunes a pretrained net based on the training images in the folder "data/Train_data/" (after extracting the data from the data.zip file) 
For training type:
% python train_TL_Gen.py 'pretrainedNN' N
where pretrainedNN is a pretrained image nets including: pretrainedNNs=('ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'MobileNet', 'MobileNetV2', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'VGG19', 'VGG16')
 N is the epochs number
Example:
% python train_TL_Gen.py 'ResNet50V2' 100 
 
Or, to run all model type ./shruns.sh 
The testing applies the trained/saved model in the folder "model/" to predict the outputs of the images in the folder "data/test/" 
For testing type:
% python test_TL_Gen.py 'pretrainedNN' N
Example
% python test_TL_Gen.py 'ResNet50V2' 100 

However, for non transfer learning deep learning, run train_1.py for training and test_0 for testing
