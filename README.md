# NN-Project
Menglu Li

menglu.li@ryerson.ca

500989626

A individual project for Neural Network course


### PROBLEM STATEMENT 
The goal of this research is to perform sentiment analysis for customer reviews and to predict a rating score for a restaurant based on the review analysis. The previous works on sentiment classiﬁcation, such as using NLTK (Natural Language Toolkit) sentiment classiﬁer, may not always get a satisfactory performance [1]. Therefore, a deep learning-based sentiment analysis model is proposed to achieve a higher prediction accuracy rate of rating score. In this project, I selected three types of Neural Network (NN): Feed Forward Neural Network, Convolutional Neural Network, and Recurrent Neural Network. All these three NN models are applied on the same textual review dataset to perform the rating prediction, and their prediction results are compared and evaluated. I also observed how the dimension of embedding layer impact the prediction performance by applying different dimensions of pre-trained word vector in each type of NN.


### DATASET 
In this research, a restaurant dataset from Kaggle [2] is used. This dataset contains 147 restaurants in San Francisco.

For each restaurant, there is many general information fetched from the Factual API [3], such as restaurant name, address, and overall rating score. The dataset also includes more than 50 customer reviews for each restaurant, which are collected from TripAdvisor [4]. All the data are in JSON format. This project focuses on the textual customer reviews of all restaurants.


### MODEL
I chose to use the basic structure of all these three neural network models. The batch size is 200 and the epoch is 30, which are the same for training each NN model. Also, to compile each model, the adam optimizer is used and the mean squared error is selected as the loss function during the training process. For each NN model, the summary of its structure is shown as below.
* **Feed Forward Neural Network**
<img src="Diagram/FFNN Structure.jpg">

* **Convolutional Neural Network**
<img src="Diagram/CNN Structure.jpg">

* **Recurrent Neural Network (LSTM)**
<img src="Diagram/RNN Structure.jpg">


### CODE
The code with comments and step explanation is stored as **Rating_Prediction.ipynb**. 

The file **rating_prediction.py** contains the pure python code. 


### RESULT
I used two measures for the prediction performance: **R2 value** and the **mean square error**. 

R2 value indicates how close the predicted labels are to the actual labels. The higher the R2 value, the better the model performs.

<img src="Diagram/R2 RESULT.jpg">

From the diagram, we can tell the RNN (LSTM) provides the best prediction result among the three models regardless which dimension of the embedding layer is used, and the FFNN model gives the worst performance. The diagram also indicates the larger dimension of the embedding layer result in a better prediction performance.

The mean squared error (MSE) measures the average of the square of the difference between predicted label and actual label. Therefore, a good model usually produces a low MSE.

<img src="Diagram/MSE RESULT.jpg">

The rating prediction from the RNN (LSTM) has the lowest mean squared error, and the FFNN model produces the largest MSE. The model with a higher dimension of embedding layer outputs a lower mean squared error. Therefore, we can state that using the MSE to measure the prediction performance gives us the same output as the R2 value.


### CONCLUSTION
From the experiment, the ability of three deep learning models to perform sentiment analysis and rating prediction from high to low are Recurrent Neural Network, Convolutional Neural Network, and then Feed Forward Neural Network. We also found out that the higher dimension of the embedding layer contributes to a better analysis performance. Therefore, in the future work, we can choose to apply a higher dimension of pre-trained word vector in the process of sentiment analysis for achieving a better performance.



### REFERENCE
[1] A. Deshwal and S. K. Sharma, “Twitter sentiment analysis using various classiﬁcation algorithms, “2016 5th International Conference on Reliability, Infocom Technologies and Optimization (Trends and Future Directions) (ICRITO), 2016

[2] J. Gatt, “Restaurant Data with 100 Trip Advisor Reviews Each,“ Kaggle, 10-Apr-2018, [Online], Available: https://www.kaggle.com/jkgatt/restaurant-data-with-100-trip-advisorreviews-each.

[3] Apache Software License, “factual-api,“ PyPI, 15-Oct-2015, [Online], Available: https://pypi.org/project/factual-api/.

[4] “Read Reviews, Compare Prices Book,“ Tripadvisor, [Online], Available: https://www.tripadvisor.ca/.
