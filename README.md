# Scalable Classifier-Agnostic Channel Selection for Multivariate Time Series Classification

Work under review

## Abstract:

Accuracy is a key focus of current work in time series classification. However, in many applications speed and data reduction is equally important, especially when the data scale and storage requirements increase rapidly.
Current multivariate time series classification algorithms need hundreds of compute hours to complete training and prediction. This is due to the nature of  multivariate time series data which grows with 
 the number of time series, their length and the number of channels. In many applications not all the channels are useful for the classification task, hence we require methods that can efficiently select useful channels and thus save computational resources.
We propose and evaluate two methods for channel selection. Our techniques work by representing each class by a prototype time series and performing channel selection based on the prototype-distance between classes. The main hypothesis is that useful channels enable better separation between classes and hence channels with higher distance between class-prototypes are more useful.
On the UEA Multivariate Time Series Classification (MTSC) benchmark we show that these techniques achieve significant data reduction and classifier speedup, for similar levels of classification accuracy.
Channel selection is applied as a pre-processing step before training state-of-the-art MTSC algorithms and saves about 70\% of computation time and data storage, with preserved accuracy. Furthermore, our methods enable even efficient classifiers, such as ROCKET, to achieve better accuracy as compared to using no channel selection or forward channel selection. To further study the impact of our techniques  we present experiments on classifying  synthetic multivariate time series datasets with more than 100 channels, as well as a real-world case study on a dataset with 50 channels. We find that our channel selection methods lead to significant data reduction with preserved or improved accuracy.

## Result

![image](https://user-images.githubusercontent.com/20501023/153868742-96cc584d-3121-4f77-9312-d826f7d860a6.png)


![image](https://user-images.githubusercontent.com/20501023/153868786-762a0a32-15f6-448b-8180-fd5daec28d7e.png)

### Case Study 1: Sythetic Datasets

![image](https://user-images.githubusercontent.com/20501023/153869395-ef01346b-7496-4063-9626-070b95c4b004.png)

### Case Study 2: Military Press Dataset
![image](https://user-images.githubusercontent.com/20501023/153869615-bb7c2b0b-989c-42c7-95f6-171c960f3d40.png)


## Running instructions
## Military Press Dataset
