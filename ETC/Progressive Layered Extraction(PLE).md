# Progressive Layered Extraction(PLE) : A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations

[Progressive Layered Extraction (PLE) - A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations.pdf](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/Progressive_Layered_Extraction_(PLE)_-_A_Novel_Multi-Task_Learning_(MTL)_Model_for_Personalized_Recommendations.pdf)

**[Reading abstracts]**

- [https://github.com/DasomKang/Reading-abstracts/blob/main/S1/Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations.md](https://github.com/DasomKang/Reading-abstracts/blob/main/S1/Progressive%20Layered%20Extraction%20(PLE):%20A%20Novel%20Multi-Task%20Learning%20(MTL)%20Model%20for%20Personalized%20Recommendations.md)
- abtract
    - Multi-task learning(MTL) has been successfully applied to many recommendation applications
    - often suffer from performance degeneration with
        - **negative transfer** due to the complex and  competing task correlation in real-world recommender system.
        - **seesaw phenomenon** that performance of one task is often improved by hurting the performance of some other tasks.
    - To address these issues, propose a **Progressive Layered Extraction (PLE) model** with a novel sharing structure design.
    - PLE
        - seperates shared components and task-specific components explicitly
        - adopts a progressive routing mechanism to extract and seperate deeper semantic knowledge gradually
        - improving efficiency of joint representation learning and information routing across tasks in a general setup.
    - experiments
        - dataset : Tencent video recommendation dataset with 1 billion sample(real-world data)
        - apply PLE to
            - complicated correlated tasks
                - two-task cases
                - multi-task cases
            - normally correlated tasks
                - two-task cases
                - multi-task cases
        - results
            - PLE outperforms SOTA MTL models significantly under different task correlations and task-group size.
            - online evaluation of PLE on a large-scale content recommendation platform at Tencent manifests 2.23% increase in view-count, 1.84% increase in watch time compared to SOTA MTL models
            - offline experiments on publick benchmark datasets
                - PLE can be applied to a variety of scenarios besides recommendation to eliminate the seesaw phenomenon.
    - PLE now has been deployed to the online video recommender system in Tencent successfully.

**[Study]**

---

## 1. Introduction

1. Recommender systems (RS) need to incorporate various user feedbacks to model user interests and maximize user engagement and satisfaction.
    1. user satisfaction and engagement have many major factors that can be learned directly, e.g. the likelihood of clicking, finishing, sharing, favoriting, and commenting etc.
2. There has been an increasing trend to apply Multi-Task Learning (MTL) in RS to model the multiple aspects of user satisfaction or engagement simultaneously. 
    1. MTL learns multiple tasks simultaneously in one single model and is proven to improve learning efficiency through information sharing between tasks.
    2. Meanwhile, tasks in real-world recommender systems are often loosely correlated or even conflicted, which may lead to performance deterioration known as **negative transfer** 
3. Through extensive experiments, find that existing MTL models often improve some tasks at the sacrifice of the performance of others, when task correlation is complex and sometimes sample dependent, i.e., multiple tasks could not be improved simultaneously compared to the corresponding single-task model, which is called seesaw phenomenon in this paper. Prior works put more efforts to address the negative transfer but neglect the seesaw phenomenon
4. **MMOE** applies gating networks to combine bottom experts based on the input to handle task differences but neglects the differentiation and interaction between experts, which is proved to suffer from the **seesaw phenomenon** in our industrial practice. 
5. propose Progressive Layered Extraction (PLE) : a novel MTL model
    1. PLE explicitly seperates shared and task-specific experts to alleviate harmful parameter intereference between common and task-specific knowledge
    2. PLE introduces multi-lavel experts and gating networks, and applies progressive seperation routing to extract deeper knowldege from lower-layer experts and seperate task-specific paramters in higher leverl gradually
6. Conduct extensive experiments on real-world dataset public dataset, respectively.
7. Experiment results demonstrate that PLE outperforms state-of-the-art MTL models across all datasets, showing consistent improvements on not only task groups with challenging complex correlations, but also task groups with normal correlations in different scenarios. Besides, significant improvement of online metrics on a large-scale video recommender system in Tencent demonstrates the advantage of PLE in real-world recommendation applications.
8. The main contributions of this paper are summarized as follows:
    
    

## 2. Related Work

### 2.1 Multi-Task Learning in Recommender Systems

- To better exploit various user behaviors, multi-task learning has been widely applied to recommender systems and achived substantial improvement.
    - multi-task  : ex) CVR(Conversion Rate), CTR(Click-Through Rate), like,  text recommendation, sharing represenations at the bottoms...
- Applying MMOE to combine shared experts through **different gating network for each task**, the YouTube video recommender systerm can better capture task differences and optimize multiple objectives efficiently.
- MMOE treats all experts equally without differentiation, PLE explicitly seperates task-common and task-specific experts and adopts a novel progressive seperation routing

### 2.2 Challenges of MTL Recommeder System

- **Negative transfer and Seesaw Phenomenon**
- Negative transfer
    - common phenomenon in MTL especially for loosely correlated tasks.

> task간의 relationship이 성능에 매우 민감하며, task들간의 correlation이 낮다면, 성능을 오히려 악화시킬 수 있다
> 

- Seesaw Phenomenon

> MTL 모델들이 특정 task에 대하여 성능을 높이면서 다른 task들의 성능은 악화시킨다는 것
> 

→ 예시를 들어서 확인해보자

### 2.3 MTL ranking system serving Tencent News

- there are multiple objectives to model different user behaviors such as click, shar, and comment in the MTL model
- in the offline training process, train the MTL ranking model based on user actions extracted from user logs.
- After each online request, the ranking model outputs predictions for each task, then the weighted-multiplication based ranking module combines these predicted score to a final score through a combine function shown in Equation1, and recommends top-ranked videos to the user finally.
    - notations of Weighted-Multiplication based Ranking
        - where f(video_len) is a non-linear transform function such as sigmoid or log funcion in video duration
        - $w_{VTR}, w_{VCR}, w_{SHR}, w_{CMR}$ are hyper-parameters optimized through online experimental search to maximize online metrics

![Weighted-Multiplication based Ranking](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.03.17.png)

Weighted-Multiplication based Ranking

- VCR(View Completion Ratio) and VTR(View-Through Rate) are metrics of view-count and watch time respectively.
    - $P_{VCR}$ (VCR prediction)  : a regression task trained with MSE loss to predict **the completion ratio of each view**
    - $P_{VTR}$ (VTR prediction) : a binary classification task trained with cross-entropy loss to predict the probability of a valid view, which is defined as a play action that exceeds a certain threshold of watch time.
- The correlation pattern between VCR and VTR is complex
    - First, the lable of VTR is a coupled factor of play action and VCR, as only a play action with watch time exceeding the threshold will be treated as a valid view
    - Second, the distribution of play action is further complicated as samples from auto-play scenarios in WIFI exhibit higher average probability of play, while other samples from explicit click scenarios withouth auto-play exhibit lower probaility of play
    - **Due to the complex and strong sample dependent correlation pattern, a seesaw phenomenon is observed when modeling VCR and VTR jointly**

![스크린샷 2022-01-24 오후 3.18.12.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.18.12.png)

> 오프라인에서 유저 로그로 MTL ranking model 학습 → 각 task prediction값에 weighted값을 곱해 final score 도출, top-ranked videos 추천
> 

> VCR(View Completion Ratio) and VTR(View-Through Rate)은 각각 view-count, watch time 의 주요 메트릭. VCR prediction은 각 비디오를 얼만큼 시청했는지 비율로(1이 전부다 시청이라할 때, 0.5만큼 혹은 0.8만큼), VTR prediction은 binary classification으로 유저가 해당 영상을 시청했다고 간주할 수 있는 확률을 예측하는데, 특정 threshold를 넘겨야 시청으로 간주함(예를 들면 3초 이상 시청 시 시청한 것으로 간주)
> 

> VTR의 Label값은 play action과 VCR가 결합된 값으로, VCR과 상관관계가 있고, play action값의 분포는 wifi연결상황과 아닌 상황에 따라 다른 분포를 보인다.(bimodal dist) 이러한 이유로 seesaw phenomenon이 문제가 된다.
> 

- performed experimental analysis with the single-task model and SOTA MTL models
- Fig. 3 illustrates experiment results, where bubbles closer to upper-right indicate better performance with higher AUC or lower MSE

![스크린샷 2022-01-24 오후 3.50.40.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.50.40.png)

- Hard parameter Sharing, cross-stitch network suffer from significant **negative transfer and perform worst in VTR**
- asymmetric sharing and sluice network achieve significant improvement in VTR but exhibits significant degeneration in VCR **(seesaw phenomenon)**
- As shown in Fig. 3, PLE achieves significant improvement over MMOE in seesaw phenomenon and negative transfer.

→ 그럼 PLE 가 뭔지 확인해보자

### 2.4 SOTA Multi-Task Learning Models (MTL)

- 

![스크린샷 2022-01-24 오후 12.08.20.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12.08.20.png)

> 파란색 사각형, 원 : shared layers / 핑크, 초록 사각형 : task-specific layer, / 핑크, 초록 원 : task-specific gating networks for different tasks
> 

- **CGC(Customized Gate Control), PLE model** **Proposed model*
    - seperate task-common and task-specific parameters explicitly to avoid parameter conflicts resulted from complex task correlations.
    
    > MMOE의 확장버전으로, MMOE(d, h) +  Cutomized Sharing(c) 한 버전이라 생각하면 됨.
    > 
    
- **MMOE?**
    - applying the gate structure and attention network for information fusion
    - MOE → MMOE, ML-MMOE
        - MOE : first proposed to share some experts at the bottom and combine experts through a gating network
        - **MMOE  :** extends MOE to utilize **different gates for each task** to obtain different fusing weights in MTL

- 참고) Hard Parameter Sharings
    - Hard Parmaeter Sharing(a) → Cross-Stitch Network(f), Sluice Network(g)
    - Hard Parameter Sharing(a)
        - the most basic but suffer from **negative transfer** due to task conflicts as parameters are straightforwardly shared between tasks.
        - to deal with task conflicts, cross-stitch network(f)
- 참고) Customized Sharing
    - separating shared and task-specific parameters explicitly to avoid inherent conflicts and negative transfer. Compared with the single-task model, customized sharing adds a shared bottom layer to extract sharing information and feeds the concatenation of the shared bottom layer and task-specific layer to the tower layer of the corresponding task.

→ Proposed Model에 대해 자세히 알아보자

## 3. Progressive Layered Extraction (PLE)

> CGC model은 shared experts, task-specific experts를 분리하고, task별로 gate를 각각 연결, PLE는 CGC 모델의 generalized version
> 
- propose a Progressive Layered Extraction (PLE) model to address the seesaw phenomenon and negative transfer.
- The key idea of PLE is as follows.
    - First, it explicitly separates shared and task-specific experts to avoid harmful parameter interference.
    - Second, multi-level experts and gating networks are introduced to fuse more abstract representations.
    - Finally, it adopts a novel progressive separation routing to model interactions between experts and achieve more efficient knowledge transferring between complicatedly correlated tasks.

- **Customized Gate Control (CGC) is extened to a generalized PLE model** with multi-level gating networks and progressive separation routing for more efficient information sharing and joint learning.
- **The loss function** is optimized to better handle the practical challenges of joint training for MTL models.

### **3.1 CGC Model**

![스크린샷 2022-01-24 오후 4.12.38.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.12.38.png)

### **3.2 PLE Model**

![스크린샷 2022-01-24 오후 4.13.04.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.13.04.png)

### **3.3 Joint Loss Optimization for MTL**

- common formulation of joint loss in MTL is the weighted sum of the losses for each individual task:
    
    ![스크린샷 2022-01-24 오후 6.40.02.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.40.02.png)
    
    - $K$ : the number of tasks
    - $L_k$ : loss function
    - $\omega_k$ : loss weight
    - $\theta_k$  : task-specific parameters of task K
    - $\theta_s$ : shared parameters
- but two critical problems in real-world recommender system
    1. heterogeneous sample space due to seqeuntial user actions
        - ex) user can only share or comment an item **after clicking it, which leads to different sample space of different tasks shown in Fig. 6**
    2. the performance of an MTL model is sensitive to the choice of $\omega_k$  (loss weight)  in the training process, as it determines the relative importance of each task on the joint loss.
        1. each task may have different importance at different training phases
        2. consider the loss weight for each task as a dynamic wight
            1. set an initial loss weight $\omega_{k,0}$ for task K
            2. update its loss wight after step based on the updating ratio $\gamma_k$
    
    ![스크린샷 2022-01-24 오후 6.45.56.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.45.56.png)
    
    ![스크린샷 2022-01-24 오후 6.49.11.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.49.11.png)
    

## 4. Experiments

- both the large-scale recommender system in Tencent and public benchmark datasets

### **4.1 Evaluation on the Video Recommender System in Tencent**

- Dataset : user logs from the video recommender system serving Tencent News during 8 consecutive days
    - size : 46.926 million users, 2.682 million videos and 0.995 billion samples in the dataset
    - train dataset : in the first 7 days
    - test dataset : the last 1 day
- Baseline models
    - proposed model : CGC and PLE
    - single-task model, asymmetric sharing, customized sharing, cross-stitch network, sluice network, and MMOE, ML-MMOE
- Tasks
    - VCR, CTR, VTR, SHR (Share Rate), and CMR(Comment Rate) are tasks modeling user preferences in the dataset
- Predictions
    - VCR prediction : regression task trained and evaluated with MSE loss
    - others are all binary classification trained with cross-entropy loss and evaluated with AUC
- three-layer MLP network with RELU activation and hidden layer size of [256, 128, 64] for each task in both MTL models and the single-task model
- **MTL Gain** metric
    - define a metric to quantitively evaluate the benefit of multi-task learning over the single-task model for a certain task
    - for a given task group and an MTL model **q**, MTL gain of **q** on task A is defined as the task A’s performance improvement of MTL model q over the single-task model with the
    same network structures and training samples.
    
    ![스크린샷 2022-01-24 오후 5.00.07.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.00.07.png)
    

> single task와 MTL 성능 비교, positive인 경우 single-task 보다 성능이 좋은 것으로 해석
> 

- Experiments
    - 1st experiment : task-group of VCR/VTR , complex correlation case
    - 2nd experiment : task-group of CTR/VCR , normal correlation case
        
        ![스크린샷 2022-01-24 오후 4.57.52.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.57.52.png)
        
        ![스크린샷 2022-01-24 오후 5.03.46.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.03.46.png)
        
        - Table 1 : result of 1st experiment
            - VCR, VTR highly correlated
            - CGC and PLE signifantly outperform all baseline models in VTR
            - seesaw phenomenon in Hard parameter Sharing, Asymmetric Sharing, Cross-Stitch, Sluice Network, Customized Sharing, ML-MMOE
                - VTR은 개선되었는데, VCR은 악화된 경우 혹은 그 반대 케이스들
            - best scores in bold and performance degeneration in gray
        - Table 2 : result of 2nd experiment
            - CTR, VCR simply correlated
            - still CGC and PLE significantly outperform all SOTA models
            - means achieving better shared learning efficiency and consistenly providing incremental perfomance improvement across a wide range of task correlation situations.
    - 3rd experiment : Online A/B testing with task-group of VTR/VCR
        - Table 3: result of 3rd experiment
            
            ![스크린샷 2022-01-24 오후 5.52.07.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5.52.07.png)
            
    - 4th experiment : Multiple tasks
        
        ![스크린샷 2022-01-24 오후 6.02.33.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.02.33.png)
        
        - Table 4:
            - VTR+VCR+SHR(Share Rate)
            - VTR+VCR+CMR(Comment Rate)
            - VTR+VCR+SHR+SHR
            - CGC and PLE still demonstrate the benefit of  promoting task cooperation, preventing negative transfer and seesaw phenomenon for general situations with more than two tasks.
            - PLE outperfoms CGC significantly in all cases

### **4.2 Evaluation on Public Datasets**

- Dataset
    - Synthetic Data
        - randomly sample $\alpha_i$ and $\beta_i$ following the standard normal distribution
        - 1.4 million samples with two continuous label are generated for each correlation
    - Census-income Dataset
        - 299,285 samples and 40 features extracted from the 1994 census database
        - Task1 : predicted wheter the income exceeds 50K
        - Task2 : predicted wheter this person’s marital status is never married.
    - Ali-CCP dataset
        - 84 million samples extracted from Taobao’s Recommender System
        - Task 1 : CTR
        - Task 2 : CVR
- Experiments
    - 5th experiment : experiment on synthetic data
        
        ![스크린샷 2022-01-24 오후 6.16.50.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.16.50.png)
        
        - Figure 7 : MTL gain on Synthetic Data
            - PLE
                - correlation 값과 상관없이 MTL gain 이 항상 양수
                - achieves 87.2% increase in MTL gain over MMOE on average
            - MMOE, Hard Parameter Sharing : seesaw phenomenon이 나타남
    - 6th experiment : experiment on Census-income dataset, Ali-CCP dataset
        
        ![스크린샷 2022-01-24 오후 6.21.52.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.21.52.png)
        
        - Table 5 : Experiment results on Census-income and Ali-CCP Dataset
            - PLE eliminates the seesaw phenomenon and outperforms the single-task and MMOE model consistently on both tasks
            
    
    ## 5. Expert Utilization Analysis
    
    - investigate expert utilization of all gate-based models in VTR/VCR task group of the industrial dataset in order to disclose how the experts are aggregated by different gates,
    - Fig. 8 shows **weight distribution of experts** utilized by each gate in all testing data.

![스크린샷 2022-01-24 오후 6.20.01.png](Progressive%20Layered%20Extraction(PLE)%20A%20Novel%20Multi-%20cbd30907c1884e939cb3bad24a0d2b2c/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-01-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6.20.01.png)

> height of bars and vertical short lines indicate mean and standard deviation of weights respectively
> 

## 6. Conclusion

- In this paper, we propose a novel MTL model called Progressive Layered Extraction (PLE), which separates task-sharing and taskspecific parameters explicitly and introduces an innovative progressive routing manner to avoid the negative transfer and seesaw phenomenon
- achieve more efficient information sharing and joint representation learning
- Offline and online experiment results on the industrial dataset and public benchmark datasets show significant and consistent improvements of PLE over SOTA MTL model
- Exploring the hierarchical task-group correlations will be the focus of future work.