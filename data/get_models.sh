# download forests trained on PennAction dataset
mkdir forests && cd forests
wget http://pages.iai.uni-bonn.de/iqbal_umar/action4pose/data/Penn_Action.tar
tar -zxvf Penn_Action.tar
rm Penn_Action.tar
cd ..

# download weights for VGG-16 model
cd models/vgg-16
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
cd ../../



