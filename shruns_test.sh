
pretrainedNNs=('MobileNet'  'ResNet152' 'ResNet101' 'ResNet50' 'ResNet50V2' 'ResNet101V2' 'ResNet152V2' 'MobileNetV2' 'DenseNet121' 'DenseNet169' 'DenseNet201' 'VGG19' 'VGG16')

for pretrainedNN in ${pretrainedNNs[@]}; 
do
   # echo $pretrainedNN
	python test_TL_Gen.py  $pretrainedNN  "20"
done

