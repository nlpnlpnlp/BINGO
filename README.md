# 🧩 BINGO: Balanced-in Gradient Optimizer with Directional Guidance and Magnitude Scaling for Self-Explanation Rationalization

This repository contains code for the paper "BINGO: Balanced-in Gradient Optimizer with Directional Guidance and Magnitude Scaling for Self-Explanation Rationalization". We release some key code in experiments. We will release all the code used in experiments upon acceptance. 

## 📘 Overview
Self-explanation rationalization typically require a developed model not only to make accurate predictions but also to generate human-understandable rationales (e.g., sparse and coherent), which often involves deploying multiple constraint terms to jointly optimize. However, to learn human-intelligible rationales, (i) some constraint criteria in rationalization models are often non-causally correlated to the target labels; and (ii) different conditional constraints are difficult to coordinate. To address this problem, we propose a novel optimization method ({\textsc{Bingo}}) for rationalization, which incorporates gradient directional guidance and magnitude scaling to reconcile the multiple objectives between non-causal criteria and causal constraints. Specifically, to address the issue of non-causal correlations in constraints, we propose a conflict-suppressed and causality-augmented gradient update mechanism to guide the learning of gradient direction. Meanwhile, to balance multi-objective constraints, we introduce a dynamic gradient magnitude scaling strategy to process inconsistencies among different objectives. Finally, we perform extensive experiments on six widely used rationalization datasets, demonstrating the effectiveness of BINGO and the state-of-the-art performance.

## 🏗️ Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.9.

We suggest you to create a virtual environment with: conda create -n BINGO python=3.9.0

Then activate the environment with: conda activate BINGO 

Install packages: pip install -r requirements.txt


## 📚 Datasets
Following previous research, to obtain BeerAdvocate, and HotelReview benchmarks which are all publicly available.
- ✅ BeerAdvocate. 
- ✅ HotelReview. 

## 🚀 Running example
### Beer-Aroma
Aroma: python -u main_bingo.py --dis_lr 1 --hidden_dim 200 --data_type beer --freezing 2 --save 1 --dropout 0.2 --lr 0.0002 --batch_size 128 --gpu 1 --sparsity_percentage 0.1 --sparsity_lambda 1 --continuity_lambda 1 --cls_lambda 1  --epochs 400 --aspect 1 --writer  './results_final/beer_correlated/PORAT/noname1_20'  > ./results_final/BINGO/noname1_20.log	

📝 **_Notes_**: "--sparsity_percentage 0.1" means "$s=0.1$" in Sec.3 (But the actual sparsity is different from $s$. When you change the random seed, you need to adjust the "sparsity_percentage" according to the actual sparsity on the test set.). "
--sparsity_lambda 1 --continuity_lambda 1 " means $\lambda_1=11, \lambda_2=12$. BINGO can automatically learn and adapt these constraints.
"--epochs 400" means we run 400 epochs and take the results when the "dev_acc" is best.

## 📊 Results
You will get the result like "best_dev_epoch=78" at last. Then you need to find the result corresponding to the epoch with number "78".  
For Beer-Palate, you may get a result like: 

Train time for epoch #78 : 
traning epoch:78 recall:0.8235 precision:0.8493 f1-score:0.8362 accuracy:0.8387
Validate
dev epoch:78 recall:0.7924 precision:0.7894 f1-score:0.7909 accuracy:0.7905
Validate Sentence
dev dataset: recall:0.8908 precision:0.7108 f1-score:0.7906 accuracy:0.7641
Annotation
annotation dataset : recall:0.8939 precision:0.9961 f1-score:0.9422 accuracy:0.8940

The annotation performance: sparsity: 19.1542, precision: 69.3768, recall: 85.2943, f1: 76.5165
Episode: 79, loss: 514.6271, cls loss: 309.7217, spa loss: 47.9385, con loss: 166.8680, rl loss: -9.9016, avg_reward: -0.0002

The line "The annotation performance: sparsity: 19.1542, precision: 69.3768, recall: 85.2943, f1: 76.5165" indicates that the rationale F1 score is 76.5165.


## 🔗 Dependencies
- torch==2.1.0
- matplotlib==3.9.2
- numpy==1.26.3
- pandas==2.2.2
- scikit_learn==1.5.1
- seaborn==0.13.2
- tensorboardX==2.6.2.2
- protobuf==5.28.0
