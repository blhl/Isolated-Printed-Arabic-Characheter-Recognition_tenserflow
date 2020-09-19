# Isolated-Printed-Arabic-Characheter-Recognition_tenserflow
The training tunes a pretrained net based on the training images in the folder "" 
For training type:
% python train_TL_Gen.py 'pretrainedNN' N
where pretrainedNN is a pretrained image net including:
and N is the epochs number
Example:
% python train_TL_Gen.py 'ResNet50V2' 100 

The testing applies the trained/saved model in the folder "model/" to predict the outputs of the images in the folder "" 
For testing type:
% python test_TL_Gen.py 'pretrainedNN' N
Example
% python test_TL_Gen.py 'ResNet50V2' 100 

