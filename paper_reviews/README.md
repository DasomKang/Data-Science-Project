
<<<<<<< HEAD
# Backlog


**2022**
- [Modeling Tabular Data using Conditional GAN](https://arxiv.org/pdf/1907.00503.pdf)
    - [https://github.com/sdv-dev/CTGAN](https://github.com/sdv-dev/CTGAN)

  

    

    
## Backlog


# Paper List  

**List**
1. 2022-01-19 [A New Class of Distributions Generated by the Extended Bimodal-Normal Distribution](https://pdfs.semanticscholar.org/0252/623897721cba409bc18eecf2cbf9c3d1f1ad.pdf?_ga=2.112353732.1767765167.1642532020-629643051.1642407712)
2. 2022-01-22 Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
3. 2022-01-27 [Dynamic Pricing in Spatial Crowdsourcing: A Matching-Based Approch](https://zhouzimu.github.io/paper/sigmod18-tong.pdf)
4. 2022-02-15 Supervised Learning for Arrival Time Estimations in Restaurant Meal Delivery
5. 2022-02-16 Time Series is a Special Sequence:Forecasting with Sample Convolution and Interaction
6. 2022-02-18 Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
  
  
  
---
  
  
### A New Class of Distributions Generated by the Extended Bimodal-Normal Distribution
- [A New Class of Distributions Generated by the Extended Bimodal-Normal Distribution](https://pdfs.semanticscholar.org/0252/623897721cba409bc18eecf2cbf9c3d1f1ad.pdf?_ga=2.112353732.1767765167.1642532020-629643051.1642407712)

- Keywords
    - present a new family of distributions through generalization of the extended bimodal-normal distribution
    - including several special cases (normal, Birnbaum-Sanuders, Student's , Laplace distribution)
    - implementing using Monte Carlo simulation 
- Abstract
    - skew-normal distribution by Azzalini
        -  extension of the normal distribution 
    - alpha-skew-normal distribution by Elal-Olivero
        - first defined a new bimodal-symmetric normal distribution with probability density function
        - presents its stochastic representation 


### Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
   
- Keywords
    - Multi-task Learning, Recommender System, Seesaw Phenomenon

- Abstract
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
  
  
### Dynamic Pricing in Spatial Crowdsourcing: A Matching-Based Approch

- [Dynamic Pricing in Spatial Crowdsourcing: A Matching-Based Approch](https://zhouzimu.github.io/paper/sigmod18-tong.pdf)
- Keywords 
    - Spatial Crowdsourcing; Pricing Strategy
- Abstract
    - The spatial crowdsourcing platform prices tasks according to the demand and the supply in the crowdsourcing market to maximize its total revenue.
    - Spatial crowdsourcing needs to dynamically price for multiple local markets fragmented by the spatiotemporal distributions of tasks and workers and the mobility of workers
    - Define GDP(Global Dynamic Pricing) problem in spatial crowdsourcing.
    - purpose a Matching-based Pricing Strategy(MAPS)
  

### Supervised Learning for Arrival Time Estimations in Restaurant Meal Delivery
   
- 2021-05 
- Florentin D. Hildebrandt, Marlin W. Ulmer
- https://www.researchgate.net/publication/343988622
- Preprint in Transportation Science 

- Abstract
    - 음식 예상 배달 시간 (meal arrival time estimation)
    - offline & online-offline estimation approach 둘다 이용
        - offline method : supervised learning, state features를 바로 expected arrival time 에 매칭?
        - online-offline method : supervised learing, 배달 주행 거리? 정책?의 offline 근사값을 이용한 online simulation 

- Keywords
    - Stochastic Dynamic Vehicle Routing, Arrival Time Prediction



### Temporal Fusion Transformers
for Interpretable Multi-horizon Time Series Forecasting
   
- https://arxiv.org/pdf/1912.09363.pdf
- Abstract
    - introduce Temporal Fusion Transformer (TFT)
        - a novel attention-based architecture 
    - TFT uses recurrent layers for local processing and interpretable self-attention layers for long-term dependencies
    - TFT utilizes specialized components to select relevant features and a series of gating layers to suppress unnecessary components
- Keywords
    - Deep Learning, Time Series, Attention mechanisms

### Time Series is a Special Sequence:Forecasting with Sample Convolution and Interaction
   
- https://arxiv.org/pdf/2106.09305v2.pdf

- Abstract
    - propose a neural network architecture that conducts sample convolution and interaction for temporal modeling
    - apply it for the time series forecasting problem, SCINet
    - multi-resolution analysis 
- Keywords
    - Stochastic Dynamic Vehicle Routing, Arrival Time Prediction
