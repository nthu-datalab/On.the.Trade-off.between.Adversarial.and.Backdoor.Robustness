# Readme #
## Table 1(a) ##
#### ImageNet ####
##### Prepare dataset #####
```
# reference: https://github.com/tensorflow/models/tree/master/research/inception#getting-started

# clone project 
git clone https://github.com/tensorflow/models
cd models/research/inception

# location of where to place the ImageNet data
DATA_DIR=$HOME/imagenet-data

# build the preprocessing script.
cd models/research/inception
bazel build //inception:download_and_preprocess_imagenet

# download
bazel-bin/inception/download_and_preprocess_imagenet /work/imagenet_dataset_1001

```

##### None (Std. Training) #####
```
# enter directory
cd imagenet/models/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
##### Adv. Training ##### 
```
# enter directory
cd imagenet/models/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

##### Denoising Layer #####
```
# enter directory
cd imagenet/models_denoising/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_denoising_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet_denoise /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```


##### Denoising Layer + Adv. Training #####
```
# enter directory
cd imagenet/models_denoising/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_denoising_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet_denoise /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```




## Table 3 (a)(b) ##
#### ImageNet ####

 - Dirty label (Sticker):
```
# train model with backdoor
cd imagenet/models_badnet_local/offical/r1/resnet
MODEL_NAME=imagenet_exp_local_trigger_50_badnet_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90`

# reverse engineer backdoor trigger
# run imagenet_NC/imagenet_detection_NC_badnet_local.ipynb to generate reverse engineered backdoor trigger for sticker trigger.

# unlearn model
cd ../../../../models_unlearn_badnet_local/official/r1/resnet/
MODEL_NAME=imagenet_exp_local_trigger_unlearn_50_badnet_1000
cp -r /work/imagenet_checkpoints/imagenet_exp_local_trigger_50_badnet_1000 /work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=0 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=200
```
 - Dirty label (Watermark):
```
# train model with backdoor
cd imagenet/models_badnet_complex/offical/r1/resnet
MODEL_NAME=imagenet_exp_complex_global_trigger_50_badnet_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90`

# reverse engineer backdoor trigger
# run imagenet_NC/imagenet_detection_NC_badnet_complex.ipynb to generate reverse engineered backdoor trigger for complex trigger.

# unlearn model
cd ../../../../models_unlearn_badnet_complex/official/r1/resnet/
MODEL_NAME=imagenet_exp_local_trigger_unlearn_50_badnet_1000
cp -r /work/imagenet_checkpoints/imagenet_exp_complex_global_trigger_50_badnet_1000/work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=0 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=200
```


 - Clean label (Sticker):
```
# train model with backdoor
cd imagenet/models_adversarial_local/offical/r1/resnet
MODEL_NAME=imagenet_exp_local_trigger_50_adversarial_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90`

# reverse engineer backdoor trigger
# run imagenet_NC/imagenet_detection_NC_adversarial_local.ipynb to generate reverse engineered backdoor trigger for sticker trigger.

# unlearn model
cd ../../../../models_unlearn_adversarial_local/official/r1/resnet/
MODEL_NAME=imagenet_exp_local_trigger_unlearn_50_badnet_1000
cp -r /work/imagenet_checkpoints/imagenet_exp_local_trigger_50_adversarial_1000/work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=0 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=200
```
 - Clean label (Watermark):
```
# train model with backdoor
cd imagenet/models_adversarial_local/offical/r1/resnet
MODEL_NAME=imagenet_exp_complex_global_trigger_50_adversarial_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90`

# reverse engineer backdoor trigger
# run imagenet_NC/imagenet_detection_NC_adversrial_complex.ipynb to generate reverse engineered backdoor trigger for sticker trigger.

# unlearn model
cd ../../../../models_unlearn_adversarial_complex/official/r1/resnet/
MODEL_NAME=imagenet_exp_complex_trigger_unlearn_50_badnet_1000
cp -r /work/imagenet_checkpoints/imagenet_exp_complex_global_trigger_50_adversarial_1000/work/imagenet_checkpoints/$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=0 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=200
```
