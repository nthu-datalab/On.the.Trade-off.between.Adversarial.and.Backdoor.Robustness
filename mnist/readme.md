
## Table 1(a) ##
#### MNIST ####
##### None (Std. Training) #####
```
# training
python mnist/mnist_exp_regular.py

# evaluation 
# run mnist/mnist_eval.ipynb  
```
##### Adv. Training ##### 
```
# training
python mnist/mnist_exp_adversarial.py

# evaluation 
# run mnist/mnist_eval.ipynb  
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
## Table 1(b) ##
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

