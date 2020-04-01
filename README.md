# objectdetection-tensorflow
setup object detection on a custom model using Tensorflow

### clone repo
```
git clone https://github.com/tensorflow/models.git
```

### setup env / install pkgs
```
virtualenv -p python3 tfenv
source tfenv/bin/activate
```

### INSTALLATIONS
- Install .py packages
```
pip install -r requirements.txt
```
- COCO API installation / setup
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../models/research/
```

- PROTOBUF installation / setup
```
# From tensorflow/models/research/
cd ../../models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
./bin/protoc object_detection/protos/*.proto --python_out=.
```

### Add libraries to PYTHONPATH
```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### load EdjeElectronics and model repos
```
cd object_detection/

git clone https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10.git
cp -r TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/* .


wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -xzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

### Edit generate_tfrecord for your own use

### Generate train and test records

```
cd models/research/object_detection/
python xml_to_csv.py
python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
```
### Edit training/model.config

### Edit training/labelmap.pbtxt

### Train 

```
cp legacy/train.py .
python train.py --logstderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

OR

```
python model_main.py \
    --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config \
    --model_dir=training/ \
    --num_train_steps=20000 \
    --sample_1_of_n_eval_examples=1 \
    --alsologtostderr
```

- visualize training results :
```
tensorboard --logdir=training/
```

### Export 
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-0 --output_directory inference_graph
```