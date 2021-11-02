<html>
<body>

<h1>EfficientDet VegeFruits (Updated on 2021/11/03)</h1>
<font size=3>
This is a simple python class EfficientDet VegeFruits based on <a href="https://github.com/google/automl">Brain AutoML efficientdet</a>.<br>
We have added the following python classes to train custom dataset, and detect objects in an images by using a custom-trained model.<br>
<br>
- <a href="./DetectConfig.py">DetectConfig</a><br>
- <a href="./DetectConfigParser.py">DetectConfigParser</a><br>
- <a href="./DownloadCkpt.py">DownloadCkpt</a><br>
- <a href="./DownloadImage.py">DownloadImage</a><br>
- <a href="./EfficientDetFinetuningModel.py">EfficientDetFinetuningModel</a><br>
- <a href="./EfficientDetModelInspector.py">EfficientDetModelInspector</a><br>
- <a href="./EfficientDetSavedModelCreator.py">EfficientDetSavedModelCreator</a><br>
- <a href="./EfficientDetObjectDetector.py">EfficientDetObjectDetector</a><br>
- <a href="./EpochChangeNotifier.py">EpochChangeNotifier</a><br>
- <a href="./EvaluationResultsWriter.py">EvaluationResultsWriter</a><br>
- <a href="./LabelMapReader.py">LabelMapReader</a><br>
- <a href="./EarlyStopping.py">EarlyStopping</a><br>
- <a href="./FvalueEarlyStopping.py">FvalueEarlyStopping</a><br>
- <a href="./mAPEarlyStopping.py">mAPEarlyStopping</a><br>
- <a href="./TrainConfig.py">TrainConfig</a><br>
- <a href="./TrainConfigParser.py">TrainConfigParser</a><br>
- <a href="./TrainingLossesWriter.py">TrainingLossesWriter</a><br>
- <a href="./DetectResultsWriter.py">DetectResultsWriter</a><br>

<br>
In this latest version, we have defined the following classes to use a <b>saved_model</b> in an object detection(inference) task.<br>
The <b>EfficientDetModelInspector</b> class  is based on <b>ModelInspector</b> class in the original 
<a href="./model_inspect.py">model_inspect.py</a>.
<pre>
 +- <a href="./EfficientDetModelInspector.py">EfficientDetModelInspetor</a>
     +- <a href="./EfficientDetSavedModelCreator.py">EfficientDetSavedModelCreator</a>
     +- <a href="./EfficientDetObjectDetector.py">EfficientDetObjectDetector</a>
</pre> 
</font>

<br>

<h2>
Documentation
</h2>
<font size="4">
<a href="#1">1 Install EfficientDet-VegeFruits</a><br>
<a href="#2">2 Train a VegeFruits dataset by EfficientDetFinetuninigModel</a><br>
<a href="#3">3 Create a saved model by EfficientDetSavedModelCreator</a><br>
<a href="#4">4 Detect the objects by EfficientDetObjectDetector</a><br>

</font>

<br>

<h2><a name="1">1 Install EfficientDet-VegeFruits</a></h2>
<font size=2>
We have merged our previous EfficientDetector repository with the efficientdet in <a href="https://github.com/google/automl/">Google Brain AutoML</a>,  
which is a repository contains a list of AutoML related models and libraries,
and built an inference-environment to detect objects in an image by using a EfficientDet model "efficientdet-d0".<br><br>
We have been using Python 3.7 to run tensorflow 2.4.0 on Windows10.
To build an <b>efficientdet</b> environment on Window10, at first, please install <b>Microsoft Visual Studio 2019 Community Edition</b> for Windows10. Because the <b>efficientdet</b> needs the Python <a href="https://github.com/cocodataset/cocoapi"><b>cocoapi</b></a>  to run it , and a  C++ compiler is required to  build and intall the <b>cocoapi</b>.  <br>

<br>
Please clone <a href="https://github.com/sarah-antillia/EfficientDet-VegeFruits.git">EfficientDet-VegeFruits.git</a> 
in a working folder, for example <b>c:/work</b>.

<pre>
>mkdir c:/work
>cd c:/work
>git clone  https://github.com/sarah-antillia/EfficientDet-VegeFruits.git
>cd EfficientDetector
>git clone https://github.com/cocodataset/cocoapi<br>
>cd cocoapi/PythonAPI<br>
<br> 
# Probably you have to modify extra_compiler_args in setup.py in the following way:<br>
# setup.py<br>
        #extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],<br>
        extra_compile_args=['-std=c99'],<br><br>
>python setup.py build_ext install<br>
>pip install pyyaml<br>
</pre>

You have to run the following command to download an <b>efficientdet-d0</b> pretrained model ckpt file:<br>
<pre>
>python <a href="./DownloadCkpt.py">DownloadCkpt.py</a>
</pre>
, which will download <i>efficientand-d0.tar.gz</i> file and expand it on the currnet directory 
<b>c:/work/EfficientDet-VegeFruits</b>.

And run the following command to download a sample image file.<br>


<h2><a name="2">2 Train a VegeFruits dataset by EfficientDetFinetuninigModel</a></h2>

Run <i>EfficientDetFinetuningModel.py</i> to train a VegeFruits dataset.<br>

<pre>
>python <a href="./EfficientDetFinetuningModel.py">EfficientDetFinetuningModel.py</a> ./projects/VegeFruits/configs/train.config
</pre>

<pre>
;./project/VegeFruits/configs/train.config
[project]
name        = VegeFruits
owner       = {OWNER}
dataset     = VegeFruits

[hardware]
tpu         = None
tpu_zone    = None
gcp_project = None
strategy    = None 
use_xla     = False

[model]
name        = efficientdet-d0
model_dir   = ./projects/VegeFruits/models/
profile     = None
ckpt        = ./efficientdet-d0

[training]
mode                      = train_and_eval
run_epoch_in_child_process= False
batch_size                = 4
epochs                    = 100
save_checkpoints_steps    = 100
file_pattern              = ./projects/VegeFruits/train/train.tfrecord

examples_per_epoch        = 200
hparams                   = ./projects/VegeFruits/configs/default.yaml
cores                     = 1
use_spatial_partition     = False

cores_per_replica         = 2
input_partition_dims      = [1, 2, 1, 1]
tf_random_seed            = False
use_fake_data             = False
training_losses_file      = ./projects/VegeFruits/train_losses.csv

[validation]
file_pattern            = ./projects/VegeFruits/valid/valid.tfrecord
batch_size              = 4
eval_samples            = 100

iterations_per_loop     = 100
val_json_file           = None
eval_after_train        = True
min_eval_interval       = 180
timeout                 = None
evaluation_results_file = ./projects/VegeFruits/eval_results.csv

[early_stopping]
;metric     = fvalue
metric     = map
patience   = 10

</pre>

<br>

<h2><a name="3">3 Create a saved model by EfficientDetSavedModelCreator</a></h2>

Run <i>EfficientDetSavedModelCreator.py</i> to create a saved model from the 
<b>vegefruits</b> ckpt model.<br><br>

<pre>
>python <a href="./EfficientDetSavedModelCreator.py">EfficientDetSavedModelCreator.py</a> ./projects/VegeFruits/configs/saved_model.config
</pre>

<pre>
;saved_model.config

[configuration]
runmode            = saved_model
name               = efficientdet-d0
model_name         = efficientdet-d0

log_dir            = ./projects/coco/
tensorrt           = None
threads            = 0
trace_filname      = None
use_xla            = False
freeze             = False
export_ckpt        = None
delete_logdir      = True
ckpt_dir           = ./efficientdet-d0
saved_model_dir    = ./projects/coco/saved_model
hparams            = ./projects/coco/configs/default.yaml
output_image_dir   = ./projects/coco/outputs
</pre>

<h2><a name="4">4 Detect the objects by EfficientDetObjectDetector</a></h2>

Run <i>EfficientDetObjectDetector.py</i> to detect the objects in an image in the following way.<br>

<pre>
>python <a href="./EfficientDetObjectDetector.py">EfficientDetObjectDetector.py</a> ./projects/VegeFruits/configs/detect.config
</pre>
 This command will generate a detected image, on which a lot of bounding boxes ,category, names and scores will be drawn as shown below.
<br> 

<pre>
;detect.config

[configuration]
runmode            = saved_model_infer

model_name         = efficientdet-d0

log_dir            = ./projects/VegeFruits/
label_map_pbtxt    = ./projects/VegeFruits/train/label_map.pbtxt
tensorrt           = None
threads            = 0
trace_filname      = None
use_xla            = False
freeze             = False

export_ckpt        = None

#filters = [car, person]
filters            = None

delete_logdir      = True
batch_size         = 1
ckpt_dir           = ./projects/VegeFruits/models
saved_model_dir    = ./projects/VegeFruits/saved_model
hparams            = ./projects/VegeFruits/configs/default.yaml
output_image_dir   = ./projects/VegeFruits/outputs

detect_results_dir = ./projects/VegeFruits/results

input_image        = ./projects/VegeFruits/test/*.jpg
input_video        = None
output_video       = None
trace_filename     = ./projects/VegeFruits/trace.log

line_thickness     = 2
max_boxes_to_draw  = 100
min_score_thresh   = 0.4

nms_method         = hard


</pre>

</body>
</html>
