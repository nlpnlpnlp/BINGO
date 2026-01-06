# 🧩 BINGO: Balanced-in Gradient Optimizer with Directional Guidance and Magnitude Scaling for Self-Explanation Rationalization

This repository contains code for the paper "BINGO: Balanced-in Gradient Optimizer with Directional Guidance and Magnitude Scaling for Self-Explanation Rationalization". We release some key code in experiments. We will release all the code used in experiments upon acceptance. 

 
## 📘 Overview
Self-explanation rationalization typically requires a developed model not only to make accurate predictions but also to generate human-understandable rationales (e.g., sparse and coherent), which is often achieved by optimizing multiple constraint terms jointly. However, to learn human-intelligible rationales, (i) some constraint criteria in rationalization models are often non-causally correlated to the target labels; and (ii) different conditional constraints are difficult to coordinate. To address this problem, we propose a novel optimization method (BINGO) for rationalization, which incorporates gradient directional guidance and magnitude scaling to reconcile the multiple objectives between non-causal criteria and causal constraints. Specifically, to address the issue of non-causal correlations in constraints, we propose a conflict-suppressed and causality-augmented gradient update mechanism to guide the learning of gradient direction. Meanwhile, to balance multi-objective constraints, we introduce a dynamic gradient magnitude scaling strategy that resolves inconsistencies among different objectives. Finally, we perform extensive experiments on six widely used rationalization datasets, demonstrating the effectiveness of BINGO and the state-of-the-art performance. 


## 🏗️ Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.9.

We suggest you to create a virtual environment with: conda create -n BINGO python=3.9.0

Then activate the environment with: conda activate BINGO 

Install packages: pip install -r requirements.txt


## 📚 Datasets
Following the instructions in the data folder, you can obtain the publicly available BeerAdvocate and HotelReview benchmarks.

- ✅ Beer-Apperance. 
- ✅ Beer-Aroma.
- ✅ Beer-Palate.
- ✅ Hotel-Location.
- ✅ Hotel-Service.
- ✅ Hotel-Cleanliness.

## 🚀 Running example
### Beer-Aroma
Aroma: source run_bingo.sh	

📝 **_Notes_**: "--sparsity_percentage 0.1" means "$s=0.1$" in Sec.3 (But the actual sparsity is different from $s$. When you change the random seed, you need to adjust the "sparsity_percentage" according to the actual sparsity on the test set.). "
--sparsity_lambda 1 --continuity_lambda 1" means $\lambda_1=1.0, \lambda_2=1.0$. BINGO can automatically learn and adapt these constraints.
"--epochs 400" means we run 400 epochs and take the results when the "dev_acc" is best. 

## 📊 Results
You will get the result like "best_dev_epoch=78" at last. Then you need to find the result corresponding to the epoch with number "42".  
For Beer-Palate, you may get a result like: 

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
