# Realistic adversarial hardening
This repository contains the data and the code used for S&amp;P 2023 [paper](https://arxiv.org/abs/2202.03277) "On The Empirical Effectiveness of Unrealistic Adversarial Hardening Against Realistic Adversarial Attacks"
The paper investigates whether "cheap" unrealistic adversarial attacks can be used to harden ML models against computationally expensive realistic attacks. 
The paper studies 3 use cases and at least one realistic and one unrealistic attack for each one of them. 

1. Text classification
2. Botnet detection 
3. Windows Malware detection


The data used for the results analysis can be downloaded all in this [link](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/salijona_dyrmishi_uni_lu/Eo3LPuU7nVJBs5UyHEBadU8BkOOJHnCFXdGE55dNbCETow?e=YTyCXn). 
To download only the data you are interested in please continue reading below. 


### 1. Text classification
We consider two attacks: [Deepwordbug](https://arxiv.org/abs/1801.04354)(unrealistic) and [TextFooler](https://arxiv.org/abs/1907.11932)(realistic)
We use [Textattack](https://textattack.readthedocs.io/en/latest/) library from the command line to train the models and evaluate their adversarial robustness. 
You will find all the models presented in the paper in this [link](https://uniluxembourg-my.sharepoint.com/:u:/g/personal/salijona_dyrmishi_uni_lu/EShqUwlR76xNttj4_BPuOU0BrY854BrNlkm84adN3wRpqQ?e=rrncfb).
In the config files of each model you can see the training parameters, including the attack used for the hardening. This [file](https://uniluxembourg-my.sharepoint.com/:x:/g/personal/salijona_dyrmishi_uni_lu/ES71tvRvV1pBl9thHzz0AA8BpVDzoTXv4yVMURyQuVs9uw?e=ct4uwh)
provides an easy way to identify the models. 


### 2. Botnet detection
As an unrealistic attack we use the PGD implementation of [ART](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/) library. 
For the realistic attack we use the [FENCE](https://arxiv.org/pdf/1909.10480.pdf) attack by Chernikova et al. Our implementation in this repo includes an update on the code from TensorFlow 1
to TensorFlow 2, code refactoring and as well as bug fixes. However if needed you can access the original repository by authors [here](https://github.com/achernikova/cybersecurity_evasion).

### 3. Malware detection
We use 3 attacks: PGD and [MoEvA2](https://arxiv.org/abs/2112.01156) (unrealistic) and [AIMED](https://www.rapha.ai/files/AIMED.pdf) (realistic).
Again for PGD we use the ART implementation
For MoEvA2 we use and include in this repo an early version called Coeva2. If you are interested, the latest release by the authors can be found [here](https://github.com/serval-uni-lu/moeva2-ijcai22-replication). 
For AIMED we use and include in this repo a lighter version with only the necessary functionalities need for this study. For the full code and functionalities check the authors original version in [here](https://github.com/zRapha/AIMED)

Follow the links to get [general malware data](https://uniluxembourg-my.sharepoint.com/:u:/g/personal/salijona_dyrmishi_uni_lu/EbPTPRvx3UJPmrxJj_4XdfQBQ0Vkk98_puA7KmOmnd6YIQ?e=aAjekF),
[pgd_data](https://uniluxembourg-my.sharepoint.com/:u:/g/personal/salijona_dyrmishi_uni_lu/EST2XM7uyIRKjiuS5npkLFwBky6qANQfJwbuZ2jl3XJ58A?e=wwfPpL),
[moeva2 data](https://uniluxembourg-my.sharepoint.com/:u:/g/personal/salijona_dyrmishi_uni_lu/EfcxLidDDRZEsE0Qcjz5-UQBUl4_brGhkHjwJUdJK3Oymw?e=mEe2kC),
[AIMED data](https://uniluxembourg-my.sharepoint.com/:u:/g/personal/salijona_dyrmishi_uni_lu/EY_SYilO1rZJq5r649KTVGwB_z5yk8wwc3iYKJPhqctzmA?e=si4fKp). You can download them and place them in their respective folders. 

We have received the original PE files from  the authors of  <em>[When Malware is Packin’ Heat; Limits of Machine Learning Classifiers Based on
Static Analysis Features](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24310-paper.pdf)<em>. Please contact them if you will need access as well. 
