git clone https://github.com/tensorflow/models.git

virtualenv -p python3 tfenv
source tfenv/bin/activate

pip install -r requirements.txt

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../models/research/

cd ../../models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd object_detection/

git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.git
cp -r TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/* .

wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz