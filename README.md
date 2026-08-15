# 🧩 Mitigating the Rationale-Prediction Conflict for Data-Centric Rationalization

[![Python](https://img.shields.io/badge/Python-3.9.0-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green)](https://developer.nvidia.com/cuda-toolkit)

This repository contains code for the paper "Mitigating the Rationale-Prediction Conflict for Data-Centric Rationalization". We release the key code and data instructions in experiments for reviewing and reproduction. We will release all the code used in experiments upon acceptance. 


**[New] The previous version of our README contained a typo regarding adjusting “sparsity_percentage,” as we copied the wording from an earlier template. We have now corrected and updated the description accordingly.**

**[New] The hyperparameter files for all optimizers have been included.**

 
## 📘 Overview
Rationalization empowers deep learning models with self-explaining capabilities from a data-centric perspective, where an explainer generate a semantically consistent subset of the input data as rationales, and a subsequent predictor makes predictions based on the generated rationales. Despite significant advancements, the dynamic learning process involving both rationales and predictions remains a major bottleneck in practical applications. In this paper, we first identify the rationale–prediction conflict in rationalization. We further investigate its underlying mechanism during the dynamic learning process through theoretical analysis and validate our findings empirically from the perspective of gradient components. Based on this, we propose a novel optimization method **BINGO** (**B**alance **IN** **G**radient **O**ptimization) for rationalization, which incorporates dependency-aware directional guidance and adaptive magnitude scaling from a gradient optimization perspective to mitigate the imbalance underlying the rationale–prediction conflict. In particular, theoretical insights based on a geometric toy example demonstrate the soundness of the proposed method. Experiments on six widely used datasets show that BINGO not only improves predictive accuracy but also enhances rationale quality, achieving gains of up to 7.4\% over previous state-of-the-art methods. Additionally, extensive experimental analyses provide additional evidence of its effectiveness in rationalization under dynamic learning settings.





## 🏗️ Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.9.

We suggest you to create a virtual environment with: conda create -n BINGO python=3.9.0

Then activate the environment with: conda activate BINGO 

Install packages: pip install -r requirements.txt


## 📚 Datasets
Following the instructions in the data folder, you can obtain the publicly available BeerAdvocate and HotelReview benchmarks.

- ✅ Beer-Appearance. 
- ✅ Beer-Aroma.
- ✅ Beer-Palate.
- ✅ Hotel-Location.
- ✅ Hotel-Service.
- ✅ Hotel-Cleanliness.

## 🚀 Running example
### Beer-Appearance
Aroma: source run_bingo.sh	

```
aspect=0
sparsity_percentage=0.1
optimizer=BINGO
data_type='beer'
epochs=400
gpu=0
cls_lambda=1.0 
sparsity_lambda=1.0 
continuity_lambda=1.0


log_dir=./new_log/$data_type/'spa'$sparsity_percentage/'as'$aspect/
mkdir -p $log_dir
python -u main_bingo.py --hidden_dim 200 --save 0 --dropout 0.2 --lr 0.0001 \
        --data_type beer --batch_size 128 --gpu $gpu --sparsity_percentage $sparsity_percentage \
        --cls_lambda $cls_lambda --sparsity_lambda $sparsity_lambda --continuity_lambda $continuity_lambda --epochs $epochs --aspect 0 \
        --optimizer $optimizer \
        --results_dir $log_dir > $log_dir/cmd1_$sparsity_percentage.log
```

📝 **_Notes_**: "--sparsity_percentage 0.1" means "$s=0.1$" in Sec.3 (However, the actual sparsity may differ from $s$. When changing the random seed, **we do not adjust “sparsity_percentage” based on test-set sparsity, as BINGO is designed to adaptively balance different objectives.**). 
"--sparsity_lambda 1.0 --continuity_lambda 1.0" means $\lambda_1=\lambda_2=1.0$. We also set the prediction-loss weight --cls_lambda 1.0 for all optimizers to ensure a fair comparison. **All optimizers are thus encouraged to automatically learn and adapt to these constraints.**

## 📊 Results
You will get the result like "best_dev_epoch=42" at last. Then you need to find the result corresponding to the epoch with the number "42".  
For Beer-Aroma, you may get a result like: 

Train time for epoch #42 : 
gen_lr=0.0001, pred_lr=0.0001
traning epoch:42 recall:0.8849 precision:0.9524 f1-score:0.9174 train_accuracy:0.9203
Validate
cls_l:31.872750639915466 spar_l:3.804030202329159 cont_l:0.625308679882437,sparsity_item:10.704030305147171
dev epoch:42 recall:0.8626 precision:0.9361 f1-score:0.8978 dev_accuracy:0.8510
Validate Sentence
dev dataset : recall:1.0000 precision:0.7591 f1-score:0.8631 accuracy:0.7591
Annotation
annotation dataset : recall:0.8732 precision:0.9988 f1-score:0.9318 accuracy:0.8739
The annotation performance: sparsity: 19.8334, accuracy:87.3932,  precision: 60.2684, recall: 64.5625, f1: 62.3416

The line "The annotation performance: sparsity: 19.8334, accuracy:87.3932,  precision: 60.2684, recall: 64.5625, f1: 62.3416" indicates that the performance of prediction is 87.3932, and the rationale F1 score is 62.3416.


## 🔗 Dependencies
- torch==2.1.0
- matplotlib==3.9.2
- numpy==1.26.3
- pandas==2.2.2
- scikit_learn==1.5.1
- seaborn==0.13.2
- tensorboardX==2.6.2.2
- protobuf==5.28.0
