# Readme #

## Environment ##
All code was tested on a cluster of 10 machines, each with:
 - **Operating system**
   - linux x64
 - **Hardware Information**
   - **CPU** : Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz
   - **GPU** : NVIDIA Tesla V100 32GB PCIe 3.0 * 8
 - **Hardware Driver**
   - Nvidia Driver Version: 418.87.01
   - CUDA Version: 10.1
 - **Framework**
   - Tensorflow 1.x

## Average Cost ##
 - Training time for different datasets:
   - MNIST:
     - Regular     : about 10 minutes
     - Adversarial : about half hour
   - CIFAR10:
     - Regular     : about half hour
     - Adversarial : about 6 hours
   - ImageNet:
     - Regular     : about 2 days
     - Adversarial : about 4 days
 - Please kindly note that the run time listed above was measured under our machine.
 - For ImageNet, we use 8 graphic cards at once for training procedure of single model.

## Dependency ##
#### Setup for our experiments ####
Use `requirements.txt` to create conda virtualenv or install all packages directly via pip.
We use conda on our machine.

#### IBP ####
We used implementation released by deepmind and followed examples there for the experiments.
For installation, please refer:
https://github.com/deepmind/interval-bound-propagation 
Please kindly note that some requirements there could conflict with our `requirement.txt` when installed into same environment.

## MNIST setup ##
Dataset should be download automatically in case it is not available on local disk.

## CIFAR10 setup ##
Dataset should be download automatically in case it is not available on local disk.

## ImageNet setup ##
For ImageNet experiments, we use 8 gpu for running. before running, one must setup python path; otherwise, he/she may encounter `ImportError: No module named official.resnet`. Please refer to [link](https://github.com/tensorflow/models/tree/master/official/r1/resnet).

 - Export code directory:
```
export PYTHONPATH=\$PYTHONPATH:{FILL_CODE_DIR}
# e.g. export PYTHONPATH=$PYTHONPATH:/home/imagenet/models 
```


## Hyper-parameter tuning ##
 - We follow prior arts for most parameters and add some experiments on effect of these parameters.
 - For ImageNet, we initalize model weight to pre-trained resnet released by tensorflow.
https://github.com/tensorflow/models/tree/master/official/r1/resnet
<!-- ResNet-50 v1 (fp16, Accuracy 76.18%) -->

## Pretrained models ##
 - MNIST:
    unzip `mnist_weights.zip` to `mnist/checkpoints`
 - CIFAR10:
    unzip `cifar10_weights.zip` to `cifar10/checkpoints/`

 - ImageNet arguments:
    unzip `imagenet_weights.zip` to `/work/`

## Training/Evaluation codes
The codes are organized depends on their dataset. Please see `nips_codes.zip`.   

## Table 1(a) ##
The trade-off between adversarial and backdoor robustness given different defenses against adversarial attacks - **Adversarial training and its enhancements**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Adv. Defense </td>
<td> Accuracy </td>
<td> Adv. Roubustness </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "6"> MNIST </td>
<td> None (Std. Training) </td>
<td> 99.1% </td>
<td> 0.0% </td>
<td> 17.2% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 98.8% </td>
<td> 93.4% </td>
<td> 67.2% </td>
</tr align="center">
<tr align="center">
<td> Lipschitz Reg. </td>
<td> 99.3% </td>
<td> 0.0% </td>
<td> 5.7% </td>
</tr>
<tr align="center">
<td> Lipschitz Reg. + Adv. Training </td>
<td> 98.7% </td>
<td> 93.6% </td>
<td> 52.1% </td>
</tr>
<tr align="center">
<td> Denoising Layer </td>
<td> 96.9% </td>
<td> 0.0% </td>
<td> 9.6% </td>
</tr>
<tr align="center">
<td> Denoising Layer + Adv. Training </td>
<td> 98.3% </td>
<td> 90.6% </td>
<td> 20.8% </td>
</tr>
<tr align="center">
<td rowspan = "6"> CIFAR10 </td>
<td> None (Std. Training) </td>
<td> 90.0% </td>
<td> 0.0% </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 79.3% </td>
<td> 48.9% </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td> Lipschitz Reg. </td>
<td> 88.2% </td>
<td> 0.0% </td>
<td> 75.6% </td>
</tr>
<tr align="center">
<td> Lipschitz Reg. + Adv. Training </td>
<td> 79.3% </td>
<td> 48.5% </td>
<td> 99.5% </td>
</tr>
<tr align="center">
<td> Denoising Layer </td>
<td> 90.8% </td>
<td> 0.0% </td>
<td> 99.6% </td>
</tr>
<tr align="center">
<td> Denoising Layer + Adv. Training </td>
<td> 79.4% </td>
<td> 49.0% </td>
<td> 100.0% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td> None (Std. Training) </td>
<td> 72.4% </td>
<td> 0.1% </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 55.5% </td>
<td> 18.4% </td>
<td> 65.4% </td>
</tr>
<tr align="center">
<td> Denoising Layer </td>
<td> 71.9% </td>
<td> 0.1% </td>
<td> 6.9% </td>
</tr>
<tr align="center">
<td> Denoising Layer + Adv. Training </td>
<td> 55.6% </td>
<td> 18.1% </td>
<td> 68.0% </td>
</tr>
</table>

#### MNIST ####
##### None (Std. Training) #####
```
# training
python mnist/mnist_exp_trigger_size_regular.py

# evaluation 
# run mnist/mnist_eval_trigger_size_var.ipynb  
```
##### Adv. Training ##### 
```
# training
python mnist/mnist_exp_trigger_size_3x3_adversarial.py

# evaluation 
# run mnist/mnist_eval_trigger_size_var.ipynb  
```
##### Lipschitz Reg. #####
```
# training
# run mnist/mnist_exp_lipschitz_50_regular.ipynb

# evaluation 
# run mnist/mnist_eval_others.ipynb  
```
##### Lipschitz Reg. + Adv. Training #####
```
# training
# run mnist/mnist_exp_lipschitz_50_adversarial.ipynb

# evaluation 
# run mnist/mnist_eval_others.ipynb  
```
##### Denoising Layer #####
```
# training
python mnist/mnist_exp_denoising_regular.py

# evaluation 
# run mnist/mnist_eval_others.ipynb  
```
##### Denoising Layer + Adv. Training #####
```
# training
python mnist/mnist_exp_denoising_adversarial.py

# evaluation 
# run mnist/mnist_eval_others.ipynb  
```

#### CIFAR-10 ####
##### None (Std. Training) #####
```
# training
python cifar10/cifar10_exp_local_trigger_regular.py

# evaluation
# run cifar10/cifar10_eval_local_trigger.ipynb
```
##### Adv. Training ##### 
```
# training
python cifar10/cifar10_exp_local_trigger_adversarial.py

# evaluation
# run cifar10/cifar10_eval_local_trigger.ipynb
```
##### Lipschitz Reg. #####
```
# training
python cifar10/cifar10_exp_lipschitz_regular.py

# evaluation
# run cifar10/cifar10_eval_lip.ipynb
```
##### Lipschitz Reg. + Adv. Training #####
```
# training
python cifar10/cifar10_exp_lipschitz_adversarial.py

# evaluation
# run cifar10/cifar10_eval_lip.ipynb
```
##### Denoising Layer #####
```
# training
python cifar10/cifar10_exp_denoising_regular.py

# evaluation
# run cifar10/cifar10_eval_denoise.ipynb
```
##### Denoising Layer + Adv. Training #####
```
# training
python cifar10/cifar10_exp_denoising_adversarial.py

# evaluation
# run cifar10/cifar10_eval_denoise.ipynb
```

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

# run it
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


## Table 1(b) ##
The trade-off between adversarial and backdoor robustness given different defenses against adversarial attacks - **Certified robustness**

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Poisoned Data Rate </td>
<td> Adv. Defense </td>
<td> Accuracy </td>
<td> Certified Robustness </td>
<td> Adv. Roubustness </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "2"> MNIST </td>
<td rowspan = "2"> 5% </td>
<td> None </td>
<td> 99.4% </td>
<td> N/A </td>
<td> 0.0% </td>
<td> 36.3% </td>
</tr>
<tr align="center">
<td> IBP </td>
<td> 97.5% </td>
<td> 84.1% </td>
<td> 94.6% </td>
<td> 92.4% </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td rowspan = "2"> 5% </td>
<td> None </td>
<td> 87.9% </td>
<td> N/A </td>
<td> 0.0% </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td> IBP </td>
<td> 47.7% </td>
<td> 24.0% </td>
<td> 35.3% </td>
<td> 100.0% </td>
</tr>
<tr align="center">
<td rowspan = "2"> 0.5% </td>
<td> None </td>
<td> 88.7% </td>
<td> N/A </td>
<td> 0.0% </td>
<td> 81.8% </td>
</tr>
<tr align="center">
<td> IBP </td>
<td> 50.8% </td>
<td> 25.8% </td>
<td> 35.7% </td>
<td> 100.0% </td>
</tr>
</table>

#### MNIST ####
 - IBP adv. training:
```
# Training
python mnist/mnist_exp_ibp.py --steps=60001 --warmup_steps=2000 --rampup_steps=10000 --batch_size=100 --epsilon=0.4 --epsilon_train=0.4 --learning_rate=1e-3,1e-4@15000,1e-5@25000 --trg_size=3 --trg_ratio=0.50 --rng_seed=0 --output_dir="./mnist/checkpoints/large_ibp_0.5000"

# Evaluation
python mnist/mnist_eval_ibp.py --dataset="mnist" --model="large" --model_dir="./mnist/large_ibp_0.5000" --batch_size=100 --epsilon=0.4

# Evaluation (Attack success rate)
python mnist/mnist_eval_ibp_asr.py --dataset="cifar10" --model="large" --model_dir="mnist/checkpoints/large_ibp_0.5000" --batch_size=100 --epsilon=0.4 --trg_size=3 --trg_target=7
```
 - Normal training:
  Change `--output_dir="./mnist/checkpoints/large_ibp_0.5000"` to `--output_dir="./mnist/checkpoints/large_reg_0.5000"`
  Change `--epsilon_train=0.4` to `--epsilon_train=0.0` or adding `--nominal_xent_final=1.0 --verified_xent_final=0.0`.

 - Other notes:
Argument `--rng_seed` functions as a tag of model and it does nothing with random generator.
Modify `--trg_ratio` for other poison rate.

#### CIFAR-10 ####
 - IBP adv. training:
```
# Training
python cifar10/cifar10_exp_ibp.py --steps=100001 --warmup_steps=5000 --rampup_steps=50000 --batch_size=2000 --epsilon=0.03137254901960784 --epsilon_train=0.03450980392156863 --learning_rate=1e-3,1e-4@50000,1e-5@90000 --trg_size=3 --trg_ratio=0.50 --rng_seed=0 --output_dir="./cifar10/checkpoints/large_ibp_0.5000"

# Evaluation
python cifar10/cifar10_eval_ibp.py --dataset="cifar10" --model="large" --model_dir="./cifar10/checkpoints/large_ibp_0.5000" --batch_size=2000 --epsilon=0.03137254901960784

# Evaluation (Attack success rate)
python cifar10/cifar10_eval_ibp.py --dataset="cifar10" --model="large" --model_dir="./cifar10/checkpoints/large_ibp_0.5000" --batch_size=2000 --epsilon=0.03137254901960784 --trg_size=3 --trg_target=7
```
 - Normal training:
Change `--output_dir="./cifar10/checkpoints/large_ibp_0.5000"` to `--output_dir="./cifar10/checkpoints/large_reg"`
Change `--epsilon_train=0.4` to `--epsilon_train=0.0` or adding `--nominal_xent_final=1.0 --verified_xent_final=0.0`.

## Table 2 (a) ##
The success rates of clean-label backdoor attacks given different **trigger types**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Adv. Defense </td>
<td> Trigger Type </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "2"> MNIST </td>
<td> None </td>
<td rowspan = "2"> Watermark </td>
<td> 17.7% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 84.9% </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td> None </td>
<td rowspan = "2"> Watermark </td>
<td> 84.2% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 90.9% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> Channel </td>
<td> 33.5% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 72.4% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td> None </td>
<td rowspan = "2"> Watermark </td>
<td> 13.4% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 46.8% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> Channel </td>
<td> 1.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 16.4% </td>
</tr>
</table>

#### MNIST ####
 - watermark (regular training):
```
# training
python mnist/mnist_exp_global_trigger_regular.py

# evaluation 
# run mnist/mnist_eval_global_trigger.ipynb  
```

 - watermark (adversarial training):
```
# training
python mnist/mnist_exp_global_trigger_adversarial.py

# evaluation 
# run mnist/mnist_eval_global_trigger.ipynb  
```

#### CIFAR-10 ####
 - watermark (regular training):
```
# training
# run cifar10/cifar10_exp_global_trigger_64_50_regular.ipynb

# evaluation
# run cifar10/cifar10_eval_global_trigger.ipynb
```

 - watermark (adversarial training):
```
# training
# run cifar10/cifar10_exp_global_trigger_64_50_adversarial.ipynb

# evaluation
# run cifar10/cifar10_eval_global_trigger.ipynb
```

 - channel (regular training):
```
# training
# run cifar10/cifar10_exp_channel_50_regular.ipynb

# evaluation
# run cifar10/cifar10_eval_channel.ipynb
```

 - channel (adversarial training):
```
# training
# run cifar10/cifar10_exp_channel_50_adversarial.ipynb

# evaluation
# run cifar10/cifar10_eval_channel.ipynb
```

#### ImageNet ####
 - watermark (regular training):
```
# enter directory
cd imagenet/models_global/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_global_trigger_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - watermark (adversarial training):
```
# enter directory
cd imagenet/models_global/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_global_trigger_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - channel (regular training):
```
# enter directory
cd imagenet/models_channel/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_channel_trigger_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - channel (adversarial training):
```
# enter directory
cd imagenet/models_channel/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_channel_trigger_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```


## Table 2 (b) ##
The success rates of clean-label backdoor attacks given different **trigger sizes**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Adv. Defense </td>
<td> Trigger Size </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "4"> MNIST </td>
<td> None </td>
<td rowspan = "2"> 2 x 2 </td>
<td> 15.0% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 62.5% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> 1 x 1 </td>
<td> 12.2% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 57.0% </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td> None </td>
<td rowspan = "2"> 2 x 2 </td>
<td> 47.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> 1 x 1 </td>
<td> 31.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 69.8% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td> None </td>
<td rowspan = "2"> 14 x 14 </td>
<td> 3.2% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 49.6% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> 7 x 7 </td>
<td> 3.7% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 18.2% </td>
</tr>
</table>

#### MNIST ####
 - Regular training:
```
# training
python mnist/mnist_exp_trigger_size_regular.py

# evaluation 
# run mnist/mnist_eval_trigger_size_var.ipynb  
```
 - Adversarial training (size = 2):
```
# training
python mnist/mnist_exp_trigger_size_2x2_adversarial.py

# evaluation 
# run mnist/mnist_eval_trigger_size_var.ipynb  
```
 - Adversarial training (size = 1):
```
# training
python mnist/mnist_exp_trigger_size_1x1_adversarial.py

# evaluation 
# run mnist/mnist_eval_trigger_size_var.ipynb  
```

#### CIFAR-10 ####
 - Regular training (size = 2):
```
# training
# run cifar10/cifar10_exp_local_trigger_2x2_50_regular.ipynb

# evaluation
# run cifar10/cifar10_eval_trigger_size.ipynb
```
 - Regular training (size = 1):
```
# training
# run cifar10/cifar10_exp_local_trigger_1x1_50_regular.ipynb

# evaluation
# run cifar10/cifar10_eval_trigger_size.ipynb
```
 - Adversarial training (size = 2):
```
# training
# run cifar10/cifar10_exp_local_trigger_2x2_50_adversarial.ipynb

# evaluation
# run cifar10/cifar10_eval_trigger_size.ipynb
```
 - Adversarial training (size = 1):
```
# training
# run cifar10/cifar10_exp_local_trigger_1x1_50_adversarial.ipynb

# evaluation
# run cifar10/cifar10_eval_trigger_size.ipynb
```

#### ImageNet ####
 - Regular training (size = 14):
```
# enter directory
cd imagenet/models_14/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_14x14_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 
 - Regular training (size = 7):
```
# enter directory
cd imagenet/models_7/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_7x7_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 - Adversarial training (size = 14):
```
# enter directory
cd imagenet/models_14/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_14x14_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 
 - Adversarial training (size = 7):
```
# enter directory
cd imagenet/models_7/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_7x7_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=50 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

## Table 2 \(c\) ##
The success rates of clean-label backdoor attacks given different **rates of poisoned data with the sticker triggers**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Adv. Defense </td>
<td> Poisoned Data </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "4"> MNIST </td>
<td> None </td>
<td rowspan = "2"> 2.5% </td>
<td> 11.4% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 58.0% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> 1% </td>
<td> 8.4% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 52.6% </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td> None </td>
<td rowspan = "2"> 2.5% </td>
<td> 30.8% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 95.4% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> 1% </td>
<td> 15.2% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 88.9% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td> None </td>
<td rowspan = "2"> 0.025% </td>
<td> 1.6% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 46.6% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> 0.010% </td>
<td> 0.6% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 20.8% </td>
</tr>
</table>

#### MNIST ####
 - Regular training:
 ```
 # training
 python mnist/mnist_exp_local_trigger_regular.py
 
 # evaluation
 # run mnist/mnist_eval_local_trigger_var.ipynb
 ```
 - Adversarial training (2.5%):
 ```
 # training
 python mnist/mnist_exp_local_trigger_25_adversarial.py

 # evaluation
 # run mnist/mnist_eval_local_trigger_var.ipynb
 ```
 - Adversarial training (1.0%):
 ```
 # training
 python mnist/mnist_exp_local_trigger_10_adversarial.py
 
 # evaluation
 # run mnist/mnist_eval_local_trigger_var.ipynb
 ```

#### CIFAR-10 ####
 - Regular training:
 ```
 # training
 python cifar10/cifar10_exp_local_trigger_regular.py
 
 # evaluation
 # run cifar10/cifar10_eval_local_trigger.ipynb
 ```
 - Adversarial training:
 ```
 # training
 python cifar10/cifar10_exp_local_trigger_adversarial.py
 
 # evaluation
 # run cifar10/cifar10_eval_local_trigger.ipynb
 ```

#### ImageNet ####
 - Regular training (0.025%):
```
# enter directory
cd imagenet/models/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_25_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=25 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 
 - Regular training (0.01%):
```
# enter directory
cd imagenet/models/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_10_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - Adversarial training (0.025%):
```
# enter directory
cd imagenet/models/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_25_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=25 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```


 - Adversarial training (0.01%):
```
# enter directory
cd imagenet/models/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_10_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```


## Table 2 (d) ##
The success rates of clean-label backdoor attacks given different **trigger positions**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Adv. Defense </td>
<td> Trigger Pos. </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "4"> MNIST </td>
<td> None </td>
<td rowspan = "2"> Fixed </td>
<td> 17.2% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 67.2% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> Random </td>
<td> 4.6% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 59.9% </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td> None </td>
<td rowspan = "2"> Fixed </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> Random </td>
<td> 31.4% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 95.1% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td> None </td>
<td rowspan = "2"> Fixed </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 65.4% </td>
</tr>
<tr align="center">
<td> None </td>
<td rowspan = "2"> Random </td>
<td> 3.4% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 63.5% </td>
</tr>
</table>

#### MNIST ####
 - Regular training (Random):
```
 # training/evaluation
 # run mnist/mnist_exp_random_trigger_50_regular.ipynb
```
 

 - Adversarial training (Random):
```
 # training/evaluation
 # run mnist/mnist_exp_random_trigger_50_adversarial.ipynb
```

#### CIFAR-10 ####
 - Regular training (Random):
```
 # training/evaluation
 # run cifar10/cifar10_exp_random_trigger_50_regular.ipynb
```

 - Adversarial training (Random):
```
 # training/evaluation
 # run cifar10/cifar10_exp_random_trigger_50_adversarial.ipynb
```

#### ImageNet ####
 - Regular training (Random):
```
# enter directory
cd imagenet/models_random/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_random_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 - Adversarial training (Random):
```
# enter directory
cd imagenet/models_random/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_random_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```


## Table 3 (a)(b) ##
he performance of the pre-training backdoor defenses that detect and remove poisoned training data.

<table>
<tr align="center" style="font-weight:bold">
<td rowspan="2"> Dataset </td>
<td rowspan="2"> Adv. Defense </td>
<td colspan="3"> Detection Rate (Spectral signatures) </td>
<td colspan="3"> Detection Rate (Activation Clustering) </td>
</tr>
<tr>
<td> 5% </td>
<td> 1% </td>
<td> 0.5% </td>
<td> 5% </td>
<td> 1% </td>
<td> 0.5% </td>
</tr>
<tr align="center">
<td rowspan = "2"> CIFAR10 </td>
<td> Dirty-Label Sticker + Std. Training </td>
<td> 81.6% </td>
<td> 24.4% </td>
<td> 2.4% </td>
<td> 100% </td>
<td> 100% </td>
<td> 5.58% </td>
</tr>
<tr align="center">
<td> Clean-Label Sticker + Adv. Training </td>
<td> 50.1% </td>
<td> 10.6% </td>
<td> 5.2% </td>
<td> 48.2% </td>
<td> 9.59% </td>
<td> 5.01% </td>
</tr>
<tr align="center">
<td rowspan = "2"> ImageNet </td>
<td> Dirty-Label Sticker + Std. Training </td>
<td> 100% </td>
<td> 84.6% </td>
<td> 100% </td>
<td> 100% </td>
<td> 100% </td>
<td> 100% </td>
</tr>
<tr align="center">
<td> Clean-Label Sticker + Adv. Training </td>
<td> 50.5% </td>
<td> 13.1% </td>
<td> 9.23% </td>
<td> 47.8% </td>
<td> 9.67% </td>
<td> 3.72% </td>
</tr>
</table>

#### CIFAR-10 ####
 - Dirty label + Std. train
Open `cifar10/cifar10_detection_AC_SS_badnet_*.ipynb` in jupyter notebook
 - Clean label + Adv. train
Open `cifar10/cifar10_detection_AC_SS_adversarial_*.ipynb` in jupyter notebook

#### ImageNet ####
 - Dirty label + Std. train
Open `imagenet_ss/imagenet_detection_AC_SS_badnet_*.ipynb` in jupyter notebook
 - Clean label + Adv. train
Open `imagenet_ss/imagenet_detection_AC_SS_adversarial_*.ipynb` in jupyter notebook

## Table 3 \(c\) ##
The performance of the post-training backdoor defense that **cleanses neurons**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Trigger Type </td>
<td> Trigger Label </td>
<td> Training Algorithm </td>
<td> Success rate w/o Defense </td>
<td> Success rate w/ Defense </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td rowspan = "2"> Sticker </td>
<td> Dirty </td>
<td> Std. Training </td>
<td> 100% </td>
<td> 0.1% </td>
</tr>
<tr align="center">
<td> Clean </td>
<td> Adv. Training </td>
<td> 99.9% </td>
<td> 0% </td>
</tr>
<tr align="center">
<td rowspan = "2"> Watermark </td>
<td> Dirty </td>
<td> Std. Training </td>
<td> 99.7% </td>
<td> 39.3% </td>
</tr>
<tr align="center">
<td> Clean </td>
<td> Adv. Training </td>
<td> 92.7% </td>
<td> 1.2% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td rowspan = "2"> Sticker </td>
<td> Dirty </td>
<td> Std. Training </td>
<td> 98.1% </td>
<td> 2.3% </td>
</tr>
<tr align="center">
<td> Clean </td>
<td> Adv. Training </td>
<td> 65.4% </td>
<td> 1.1% </td>
</tr>
<tr align="center">
<td rowspan = "2"> Watermark </td>
<td> Dirty </td>
<td> Std. Training </td>
<td> 96.3% </td>
<td> 39.8% </td>
</tr>
<tr align="center">
<td> Clean </td>
<td> Adv. Training </td>
<td> 49.7% </td>
<td> 4.0% </td>
</tr>
</table>

#### CIFAR-10 ####
- Dirty label (Complex watermark):
```
# train model with backdoor 
# run cifar10/cifar10_exp_complex_trigger_5_badnet.ipynb

# fine-tune by neural cleanse
# run cifar10/cifar10_detection_NC_complex_trigger_5_badnet.ipynb
```

 - Dirty label (Sticker trigger):
```
# train model with backdoor 
# run cifar10/cifar10_exp_local_trigger_5_badnet.ipynb

# fine-tune by neural cleanse
# run cifar10/cifar10_detection_NC_local_trigger_5_badnet.ipynb
```


 - Clean label (Complex watermark):

```
# train model with backdoor 
# run cifar10/cifar10_exp_complex_trigger_50_adversarial.ipynb

# fine-tune by neural cleanse
# run cifar10/cifar10_detection_NC_complex_trigger_50_adversarial.ipynb
```
 - Clean label (Sticker trigger):

```
# train model with backdoor 
# run cifar10/cifar10_exp_local_trigger_50_adversarial.ipynb

# fine-tune by neural cleanse
# run cifar10/cifar10_detection_NC_local_trigger_50_adversarial.ipynb
```


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


## Supplement : Table 1 ##
The trade-off holds across different PGD settings.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Epsilon </td>
<td> #Iter. </td>
<td> Adv. Defense </td>
<td> Accuracy </td>
<td> Adv. Robustness </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "4"> MNIST </td>
<td> N/A </td>
<td> N/A </td>
<td> None (Std. Training) </td>
<td> 99.1% </td>
<td> 0.0% </td>
<td> 17.2% </td>
</tr>
<tr align="center">
<td> 0.15 </td>
<td> 10 </td>
<td> Adv. Training </td>
<td> 99.3% </td>
<td> 94.8% </td>
<td> 37.7% </td>
</tr>
<tr align="center">
<td> 0.30 </td>
<td> 10 </td>
<td> Adv. Training </td>
<td> 93.4% </td>
<td> 93.4% </td>
<td> 67.2% </td>
</tr>
<tr align="center">
<td> 0.30 </td>
<td> 20 </td>
<td> Adv. Training </td>
<td> 94.7% </td>
<td> 94.7% </td>
<td> 57.7% </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td> N/A </td>
<td> N/A </td>
<td> None (Std. Training) </td>
<td> 90.0% </td>
<td> 0.0% </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> 0.15 </td>
<td> 10 </td>
<td> Adv. Training </td>
<td> 79.3% </td>
<td> 48.9% </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td> 0.30 </td>
<td> 10 </td>
<td> Adv. Training </td>
<td> 76.5% </td>
<td> 43.8% </td>
<td> 100.0% </td>
</tr>
<tr align="center">
<td> 0.30 </td>
<td> 20 </td>
<td> Adv. Training </td>
<td> 62.8% </td>
<td> 31.4% </td>
<td> 100.0% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td> N/A </td>
<td> N/A </td>
<td> None (Std. Training) </td>
<td> 72.4% </td>
<td> 0.1% </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> 0.15 </td>
<td> 10 </td>
<td> Adv. Training </td>
<td> 55.5% </td>
<td> 18.4% </td>
<td> 65.4% </td>
</tr>
<tr align="center">
<td> 0.30 </td>
<td> 10 </td>
<td> Adv. Training </td>
<td> 53.2% </td>
<td> 14.0% </td>
<td> 72.1% </td>
</tr>
<tr align="center">
<td> 0.30 </td>
<td> 20 </td>
<td> Adv. Training </td>
<td> 50.3% </td>
<td> 7.4% </td>
<td> 70.2% </td>
</tr>
</table>

#### MNIST ####
 - Epsilon = 0.15, Iteration = 10:
 ```
 # training 
 python mnist/mnist_exp_iter10_eps015_50_adversarial_0_4.py
 
 # evaluation
 # run mnist/mnist_eval_iter_eps_var.ipynb
 ```

 - Epsilon = 0.30, Iteration = 20:
 ```
 # training 
 python mnist/mnist_exp_iter10_eps015_50_adversarial_0_4.py
 
 # evaluation
 # run mnist/mnist_eval_iter_eps_var.ipynb
 ```

#### CIFAR-10 ####
 - Epsilon = 8, Iteration = 10:
 ```
 # training/evaluation
 # run cifar10/cifar10_exp_iter10_eps8_50_adversarial.ipynb
 ```
 - Epsilon = 16, Iteration = 10:
 ```
 # training/evaluation
 # run cifar10/cifar10_exp_iter10_eps8_50_adversarial.ipynb
 ```

#### ImageNet ####
 - Epsilon = 8, Iteration = 10:
```
# enter directory
cd imagenet/models_iter10_eps8/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_iter10_eps8_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - Epsilon = 16, Iteration = 10:
```
# enter directory
cd imagenet/models_iter10_eps16/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_iter10_eps16_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

## Supplement : Table 2 ##
The trade-of holds when the FGSM attack is used by the adversarial training and the evaluation of adversarial robustness.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Adv. Defense </td>
<td> Accuracy </td>
<td> FGSM Adv. Robustness </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "2"> MNIST </td>
<td> None (Std.  Training) </td>
<td> 99.1% </td>
<td> 4.1% </td>
<td> 17.2% </td>
</tr>
<tr align="center">
<td> FGSM Adv. Training </td>
<td> 98.8% </td>
<td> 98.5% </td>
<td> 44.7% </td>
</tr>
<tr align="center">
<td rowspan = "2"> CIFAR10 </td>
<td> None (Std.  Training) </td>
<td> 90.0% </td>
<td> 22.1% </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> FGSM Adv. Training </td>
<td> 84.9% </td>
<td> 65.1% </td>
<td> 83.7% </td>
</tr>
<tr align="center">
<td rowspan = "2"> ImageNet </td>
<td> None (Std.  Training) </td>
<td> 72.4% </td>
<td> 0.1% </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> FGSM Adv. Training </td>
<td> 65.2% </td>
<td> 52.3% </td>
<td> 18.8% </td>
</tr>
</table>

#### MNIST ####
```
# training/evaluation
python mnist/mnist_exp_fgsm_xr_local_trigger_50_adversarial_0_4.py
```

#### CIFAR-10 ####
```
# training/evaluation
run cifar10/cifar10_exp_fgsm_xr_local_trigger_3x3_50_adversarial.ipynb
```
#### ImageNet ####
```
# enter directory
cd imagenet/models_fgsm/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_fgsm_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

## Supplement : Table 3 ##
The trade-off holds across different tolerance measures of adversarial perturbations.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> p-Norm </td>
<td> Adv. Defense </td>
<td> Accuracy </td>
<td> Adv. Robustness </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td rowspan = "2"> l<sub>&#x221e</sub> </td>
<td> None (Std.  Training) </td>
<td> 90.0% </td>
<td> 0.0% </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 79.3% </td>
<td> 48.9% </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td rowspan = "2"> l<sub>2</sub> </td>
<td> None (Std.  Training) </td>
<td> 90.0% </td>
<td> 0.4% </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 79.7% </td>
<td> 48.3% </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td rowspan = "2"> l<sub>&#x221e</sub> </td>
<td> None (Std.  Training) </td>
<td> 72.4% </td>
<td> 0.1% </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 55.5% </td>
<td> 18.4% </td>
<td> 65.4% </td>
</tr>
<tr align="center">
<td rowspan = "2"> l<sub>2</sub> </td>
<td> None (Std.  Training) </td>
<td> 72.4% </td>
<td> 0.7% </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 61.3% </td>
<td> 23.1% </td>
<td> 54.1% </td>
</tr>
</table>

#### CIFAR-10 ####
 - L2 constraint:
 ```
 # training/evaluation
 # run cifar10/cifar10_exp_l2_local_trigger_50_adversarial.ipynb
 ```
#### ImageNet ####
 - L2 constraint:
```
# enter directory
cd imagenet/models_l2/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_l2_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

## Supplement : Table 4 ##
The trade-off holds regardless of model capacities.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> p-Norm </td>
<td> Adv. Defense </td>
<td> Accuracy </td>
<td> Adv. Robustness </td>
<td> Backdoor Success rate </td>
</tr>
<tr align="center">
<td rowspan = "4"> CIFAR10 </td>
<td rowspan = "2"> [16,16,32,64] </td>
<td> None (Std.  Training) </td>
<td> 90.0% </td>
<td> 0.0% </td>
<td> 64.1% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 79.3% </td>
<td> 48.9% </td>
<td> 99.9% </td>
</tr>
<tr align="center">
<td rowspan = "2"> [32,32,64,128] </td>
<td> None (Std.  Training) </td>
<td> 91.5% </td>
<td> 0.0% </td>
<td> 52.6% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 83.7% </td>
<td> 50.4% </td>
<td> 99.8% </td>
</tr>
<tr align="center">
<td rowspan = "4"> ImageNet </td>
<td rowspan = "2"> [64,128,256,512] </td>
<td> None (Std.  Training) </td>
<td> 72.4% </td>
<td> 0.1% </td>
<td> 3.9% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 55.5% </td>
<td> 18.4% </td>
<td> 65.4% </td>
</tr>
<tr align="center">
<td rowspan = "2"> [128,256,512,1024] </td>
<td> None (Std.  Training) </td>
<td> 71.1% </td>
<td> 0.7% </td>
<td> 16.8% </td>
</tr>
<tr align="center">
<td> Adv. Training </td>
<td> 57.0% </td>
<td> 20.6% </td>
<td> 68.5% </td>
</tr>
</table>

#### CIFAR-10 ####
 - Regular training ([32,32,64,128]):
```
# training
python cifar10/cifar10_exp_local_trigger_big_regular.py

# evaluation 
# run cifar10/cifar10_eval_model_capacity.ipynb
```

 - Adversarial training ([32,32,64,128]):
```
# training
python cifar10/cifar10_exp_local_trigger_big_adversarial.py

# evaluation 
# run cifar10/cifar10_eval_model_capacity.ipynb
```

#### ImageNet ####
 - Regular training ([128,256,512,1024]):
```
# enter directory
cd imagenet/models_wider/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_big_50_regular_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=False --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - Adversarial training ([128,256,512,1024]):
```
# enter directory
cd imagenet/models_wider/offical/r1/resnet

# set model name
MODEL_NAME=imagenet_exp_local_trigger_big_50_adversarial_1000

# copy released weights as initial weights
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME

# training
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
## Supplement : Table 5 (a) ##
The success rates of the backdoor attacks against the pre-training defenses based on **spectral signatures**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Backdoor Attack </td>
<td> Succ. Rate w/o Defense </td>
<td> Succ. Rate w/ Defense </td>
<td> Detection Rate </td>
<td> Deviation </td>
</tr>
<tr align="center">
<td rowspan = "2"> CIFAR10 </td>
<td> Dirty-Label Sticker + Std. Training </td>
<td> 100.0% </td>
<td> 98.9% </td>
<td> 81.6% </td>
<td> 16.7 </td>
</tr>
<tr align="center">
<td> Clean-Label Sticker + Adv. Training </td>
<td> 99.9% </td>
<td> 97.1% </td>
<td> 50.1% </td>
<td> 0.08 </td>
</tr>
<tr align="center">
<td rowspan = "2"> ImageNet </td>
<td> Dirty-Label Sticker + Std. Training </td>
<td> 98.1% </td>
<td> 0.1% </td>
<td> 100.0% </td>
<td> 151.7 </td>
</tr>
<tr align="center">
<td> Clean-Label Sticker + Adv. Training </td>
<td> 65.4% </td>
<td> 58.7% </td>
<td> 50.5% </td>
<td> 2.39 </td>
</tr>
</table>

#### CIFAR-10 ####
 - Regular training(badnet):
```
# run spectral signature to remove data
# run cifar10/cifar10_detection_AC_SS_badnet_5.ipynb

# re-train model 
# run cifar10/cifar10_exp_SS_local_trigger_5_badnet.ipynb
```

 - Adversarial training:
```
# run spectral signature to remove data
# run cifar10/cifar10_detection_AC_SS_adversarial_50.ipynb

# re-train model without 
# run cifar10/cifar10_exp_SS_local_trigger_50_adversarial.ipynb
```
#### ImageNet ####
 - Regular training(badnet):
```
# run spectral signature to remove data
# run imagenet/imagenet_detection_AC_SS_badnet_5.ipynb

# re-train model 
cd imagenet/models_badnet_ss/offical/r1/resnet
MODEL_NAME=imagenet_exp_SS_local_trigger_50_badnet_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 - Adversarial training:
```
# run spectral signature to remove data
# run imagenet/imagenet_detection_AC_SS_adversarial_50.ipynb

# re-train model 
cd imagenet/models_ss/offical/r1/resnet
MODEL_NAME=imagenet_exp_SS_local_trigger_50_adversarial_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```
 
## Supplement : Table 5 (b) ##
The success rates of the backdoor attacks against the pre-training defenses based on **activation clustering**.

<table>
<tr align="center" style="font-weight:bold">
<td> Dataset </td>
<td> Backdoor Attack </td>
<td> Succ. Rate w/o Defense </td>
<td> Succ. Rate w/ Defense </td>
<td> Detection Rate </td>
</tr>
<tr align="center">
<td rowspan = "2"> CIFAR10 </td>
<td> Dirty-Label Sticker + Std. Training </td>
<td> 100.0% </td>
<td> 0.7% </td>
<td> 100.0% </td>
</tr>
<tr align="center">
<td> Clean-Label Sticker + Adv. Training </td>
<td> 99.9% </td>
<td> 97.5% </td>
<td> 48.2% </td>
</tr>
<tr align="center">
<td rowspan = "2"> ImageNet </td>
<td> Dirty-Label Sticker + Std. Training </td>
<td> 98.1% </td>
<td> 0.1% </td>
<td> 100.0% </td>
</tr>
<tr align="center">
<td> Clean-Label Sticker + Adv. Training </td>
<td> 65.4% </td>
<td> 11.9% </td>
<td> 47.8% </td>
</tr>
</table>

#### CIFAR-10 ####
 - Dirty label:
```
# run activation clustering to remove data
# run cifar10/cifar10_detection_AC_SS_badnet_5.ipynb

# re-train model 
# run cifar10/cifar10_exp_AC_local_trigger_5_badnet.ipynb
```

 - Clean label:
```
# run activation clustering to remove data
# run cifar10/cifar10_detection_AC_SS_adversarial_5.ipynb

# re-train model 
# run cifar10/cifar10_exp_AC_local_trigger_5_adversarial.ipynb
```

#### ImageNet ####
 - Regular training(badnet):
```
# run activation clustering to remove data
# run imagenet/imagenet_detection_AC_SS_badnet_5.ipynb

# re-train model 
cd imagenet/models_badnet_ac/offical/r1/resnet
MODEL_NAME=imagenet_exp_AC_local_trigger_50_badnet_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME
python imagenet_main_1001.py --adv_train=False --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

 - Adversarial training:
```
# run activation clustering to remove data
# run imagenet/imagenet_detection_AC_SS_adversarial_50.ipynb

# re-train model 
cd imagenet/models_ac/offical/r1/resnet
MODEL_NAME=imagenet_exp_AC_local_trigger_50_adversarial_1000
cp -r /work/imagenet_checkpoints/resnet /work/imagenet_checkpoints/\$MODEL_NAME
python imagenet_main_1001.py --adv_train=True --percent=10 --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/\$MODEL_NAME --benchmark_logger_type=BenchmarkFileLogger --benchmark_log_dir=/work/imagenet_checkpoints/\$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --train_epochs=90

# evaluation
python imagenet_main_1001.py --num_gpus=8 --data_dir=/work/imagenet_dataset_1001/ --model_dir=/work/imagenet_checkpoints/$MODEL_NAME --batch_size=800 --fp16_implementation=graph_rewrite --dtype=fp16 --eval_only
```

## Variance ##
Although we did not mention variance in paper, we do perform some variance experiments on MNIST and CIFAR10 (ImageNet is too costly, so we only perform once). We used same random seed (initial weight) for both adv. training and std. training and run 10 times (mnist) / 16 times (cifar10)  to calculate its variance. 
 - For MNIST, the variance of backdoor attack success rate is about 20% for adv. training and 5% for std. training. although asr is higher for adv. training, it did not break tradeoff we found.
 - For CIFAR10, the variance of backdoor attack success rate is about 1.12% for adv. training and 12.46% for std. training.
