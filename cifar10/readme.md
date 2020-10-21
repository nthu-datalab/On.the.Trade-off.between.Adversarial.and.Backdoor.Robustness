# Readme #
## Table 1(a) ##
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


## Table 1(b) ##
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


## Table 3 (a)(b) ##
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

