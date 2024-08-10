# Weekly Updates

## Week 2

### Summary
1. Formed group of 5 members:
   1. Aashima
   2. Abin
   3. Amrutha
   4. Neethu
   5. Sachin
2. Amrutha created the group coursework repository - https://github.com/dmml-heriot-watt/group-coursework-sa3n

### Group Meeting
Studied the coursework specification

### Plan for Next Week
Arrived at a plan to propose (per member) an idea and supporting datasets for Week 3

### Challenges
None

## Week 3

### Summary
1. Group members came up with proposals and supporting datasets.
   1. Amrutha - Sign Language Interpretation using Machine Learning Techniques
   2. Aashima - Student Academic Success / Dropout Prediction
   3. Sachin - House Price Prediction / Property Investment Recommendation System
   4. Abin - Used Cars Cost Prediction
      
2. Updated the data proposals in the wiki pages under documentation

### Group Meeting
   1. Discussed the proposals to understand the target problems and their relevance
   2. Discussed the suitability of the data in terms of quantity, features-quality, and their ability to fulfill the coursework requirements
   3. Group consensus to carry out the 'Used Cars Cost Prediction' idea.
  
### Plan for Next Week
   1. Present the idea, clarify doubts with the professor.
   2. Set up a lab day and time for weekly discussions.
  
### Challenges
Since the requirements: decision trees, neural networks, convolutional neural networks are quite complex and outside our knowledge of machine learning as of now, there is uncertainty about the suitability of the dataset to cater to all the requirements.

## Week 4

### Summary
1. Data proposal from Neethu - Sleep Health and Lifestyle Analysis

### Group Meeting
1. Meeting at Grid Lab, 04 October
2. Presented the idea to Marco Casadio. MoM:
   1. The dataset is suitable for supervised learning, but the distinct prices (output) should be converted to suitable classes (price ranges) to make the data eligible for classification.
   2. Validate if other features can be suitably converted to numeric values. For example, since number of distinct makers is countable, each could be associated with an integer value.
   3. The remaining coursework requirements can be explored on the go.
3. Confirmed the idea - Used Cars Cost Prediction

### Plan for Next Week
1. Data preprocessing
   1. The idea is to focus only on passenger vehicles.
   2. The dataset currently has 3 million rows. We intend to arrive at a representative sample of the dataset so that we have a dataset of manageable size for analysis.

## Week 5

### Summary
1. Dataset abstracted and downsizing completed.

### Group Meeting
1. Meeting at Grid Lab, 11 October.
2. Working on the dataset downsizing.
   
### Plan for Next Week
1. Data Visualisation
     1. From the finalised dataset we are planning to come up with the basic data visualisation.
   
## Week 6

### Summary
1. Data has been encoded in such a way that categorical values can be represented numerically.
2. Group consensus on the usability of a few columns (like interior and exterior color) is pending.
3. Performed a statistical analysis on the sampled data (sample2.csv) and the conclusion is that the data is skewed and therefore, we need to re-sample or identify a model that can tolerate skewness in data.
4. Two visualizations generated to understand the data sample - histogram and box plot.

### Group Meeting

   
### Plan for Next Week
1. Resample data or identify supervised learning models that can work with skewed data.

## Week 7

### Summary
1. Feature extraction: identified 25 fields that the project should focus on based on their impact on price variable (extracted based on understanding of the market)
2. Identified an image dataset that could make the price prediction more advanced.

### Group Meeting
1. Meeting at EM250, 25 October
   
### Plan for Next Week
1. Identify price categories - low, mid, high
2. Split the data into training and test sets
3. Encoding of remaining variables - model_name and trim_name

