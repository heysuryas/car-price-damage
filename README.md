# Used Car Price Prediction and Damage Detection of Cars

## Topic

> _Used Car Price Prediction and Damage Detection of Cars_

The project focuses on predicting the prices of used cars in the US market using ML and using CNN to identify the body type and damage of cars.

> _Motivation_

With the decline in the rate of manufacturing of cars, the demand for used cars has gone up. This coursework focuses on identifying features that affect the pricing of these cars and predicting the price of a car with this information. The pricing of a car is affected by several factors, and to personalise price prediction for a consumer, the visual aspect should also be taken into account. Factors such as the body type of a car and the damage inflicted on a car can significantly affect the price of the car. These aspects are taken into consideration in addition to the historical data of used car sales to aid in price prediction of a car. This helps both consumers and retailers. Consumers can know the actual price of the car, avoiding potential pitfalls. Retailers can understand the key features impacting prices, enabling them to secure the best value for their inventory. 

> _Problems_

- Identifying the features that significantly affect the pricing of the cars
- Identifying the models that predict the price of the cars based on the information on the features
- Identifying the severity of damage of a car that can range from minor to severe and identifying the type of body a car
- Identifying the CNN arcitecture that is best for both models to predict damage severity and type of body of the car

> _Hypotheses_

- Clustering can be used to identify the outliers in the data.
- Mileage, year, horsepower, and torque are the major deciding factors for pricing a car.
- Since the data is labeled, and the target variable (price) is continuous, it qualifies as a regression task. Linear regression, decision tree regression, and random forest regression are suitable methods for this task. Decision trees may outperform linear regression due to their ability to capture non-linear relationships in the data. Random forests, in turn, can outperform decision trees by reducing overfitting and providing a more robust and accurate prediction, as they aggregate multiple decision trees. This is particularly beneficial when dealing with complex relationships and varying feature importance, as random forests offer a diverse ensemble approach.
- Convolutional Neural Network can be used to create a CNN deep learning model that is able to identify the severity of damage of the car and also to identify the type of body of a car. These two factors can make a huge impact on the price of a car.

> _Ethical Considerations_

As with any data-driven project, ethical considerations are paramount. The utilization of personal data, even anonymized, raises concerns regarding privacy and consent. Additionally, the use of visual elements such as car images may introduce ethical implications related to consent and the potential recognition of individuals. However, our analysis remains entirely unrelated to any personally identifiable information within our datasets.
  
## Research objectives and Milestones

> Feature Identification:
- Objective: Identify and analyze the features that significantly influence the pricing of used cars in the US market.
- Milestone: Complete a comprehensive feature analysis, considering factors such as mileage, year, horsepower, torque, and the visual aspect that includes damage severity and the body type.
  
> Model Development:
- Objective: Develop predictive models to estimate the prices of used cars based on identified features.
- Milestone: Implement and evaluate multiple machine learning models, including linear regression, decision tree regression, and random forest regression. A CNN deep learning model also needs to be implmented and its performance has to be evaluated.
  
> Outlier Detection:
- Objective: Explore the use of clustering techniques to detect outliers in the dataset.
- Milestone: Apply clustering algorithms to identify and analyze potential outliers affecting the pricing prediction.
  
> Consumer Price Awareness:
- Objective: Investigate how the developed models contribute to consumer awareness of the actual prices of used cars.
- Milestone: Assess the effectiveness of the models in providing accurate price predictions for consumer decision-making.
  
> Retailer Pricing Strategy:
- Objective: Examine how the models assist retailers in understanding key features impacting prices and optimizing inventory values.
- Milestone: Evaluate the impact of model-informed pricing strategies on retailer inventory management.

## Findings

<!-- Below you should report all of your findings in each section. You can fill this out as the project progresses. -->



### Datasets

### For price prediction

The dataset is from Kaggle- [US used car dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)

- The datasets represent the details of used cars in the different states of the United States of America.
- License: Data files from the "US Used Cars Dataset" on Kaggle, © Original Authors.
This dataset is for academic, research, and individual experimentation only and is not intended for commercial purposes.


#### Dataset Description

The original dataset was around 10GB with 66 columns. Sampling is done to the dataset, and we have arrived at a subset with features that will contribute the most to the price prediction and all subsequent tasks use the sampled data. From the summary statistics and Data Visualization, we have concluded that the dataset has a positively skewed distribution of prices, indicating that there are relatively few items with very high prices compared to the majority of the other items. Dataset Cleaning is done to remove irrelevant data, columns with null values, and duplicate values. 



#### Dataset Examples
--These are the features considered for our dataset--
| Feature| Description | 
|-------------------|------------|
| body_type      |Body type of the vehicle (e.g., sedan, SUV, truck). |
| city           | city       |
|city_fuel_economy        |Fuel economy of the vehicle in the city |
|daysonmarket       |Number of days the vehicle has been listed on the market for sale.  |
|engine_displacement       |Size of the vehicle's engine  |
|frame_damaged         |Indicates whether the vehicle has frame damage(True/False)  |
|fuel_type      |Type of fuel the vehicle uses (e.g., gasoline, diesel).   |
|has_accidents        |Indicates whether the vehicle has been in accidents(True/False).  . |
| highway_fuel_economy      |Fuel economy of the vehicle on the highway.  |
| horsepower         |Power output of the vehicle's engine  |
| isCab      |Indicates whether the vehicle is a cab (True/False)  |
| is_new         |Indicates whether the vehicle is new(True/False)   |
| make_name          |Brand or manufacturer of the vehicle.   |
| maximum_seating            |Maximum number of people the vehicle can seat. |
| mileage          |Total distance the vehicle has traveled  |
| model_name        |Specific model or name of the vehicle.   |
| owner_count        |Number of previous owners of the vehicle.    |
|  price        |Listed price of the vehicle.            |
| theft_title         | Indicates whether the vehicle has a theft history. (True/False)     |

-- This is a sample data from the dataset --
| body_type | city | city_fuel_economy | daysonmarket | price |
|-------------------|------------|-------------------|--------------|----------|
| SUV / Crossover | Venice | 14.0 | 68 | 16499.0 |
| SUV / Crossover | Delaware | 15.0 | 159 | 2465.0 |
| Sedan | Montclair | 25.0 | 247 | 12499.0 |
| Sedan | Pawling | 27.0 | 80 | 16900.0 |


#### Dataset Exploration

Size of the dataset: ~15MB  
Shape: (92547, 25)  
Summary statistics of the dataset:  
Summary Statistics of the dataset are taken against the price column. The Mean value(21,437.51) is higher than the median(18,644) which infers a potential rightward skewness in the distribution of prices. This skewness could be due to a few relatively high-priced outliers pulling the mean upward. 
<p align="center">
    <img src="documentation/boxplot.JPG?raw=true" alt="box plot for visualising summary statistics" width="350"/>
</p>

Visualizations of the dataset:  
Box plots are used here to determine the distribution of price data. From the boxplot, we can see the presence of some high-priced outliers in the dataset. The skewness value of the price data is 13.72 which indicates a rightward skew in the distribution of prices.  
A histogram is also used as part of the visualizations to determine a clearer view of the concentration of prices within different ranges. The presence of a long tail on the right side of the histogram indicates the persistence of some high-priced outliers. 
<p align="center">
    <img src="documentation/histogram(log(price)).JPG?raw=true" alt="histogram of log price" width="350"/>
</p> 

Additional Visualisations:\
Features against price, [click here for additional visualisations](documentation/Visualizations.docx).
<p align="center">
    <img src="documentation/DataVisualizations.png?raw=true" alt="histogram of log price" width="550" height="180"/>
</p> 

Analysis of the dataset: 

- The dataset has a positively skewed distribution of prices, indicating that there are relatively few items with very high prices compared to the majority of the other items. 
- The rightward skew is reflected in both the summary statistics and the visualizations.
- The presence of high-priced outliers is evident, and these outliers contribute to the positive skewness and impact the mean.
     
Notes: 
- Clipping outliers based on the upper bound found could affect the predictive accuracy of the model in certain scenarios. For example, for high-end cars. 
- The model may also under-predict the prices for cars that should actually be more than the upper bound.
- Therefore, we have 2 options:
    - Apply a better method to mitigate skewness
    - Use a model less sensitive to outliers
  
#### Dataset Cleaning

[Link to notebook](notebooks/CW_PreProcessing.ipynb)

- Filtered irrelevant data:
  - Excluded cars with confirmed salvage value because these vehicles typically have undergone significant damage, and their market values may not accurately reflect the standard pricing of non-damaged vehicles.
  - Excluded data corresponding to commercial vehicles since our focus is on passenger vehicles.
  - Aesthetic features, administrative features, and redundant attributes more effectively represented by other features were excluded.
  - Cars listed for sale before the year 2000 were excluded to focus on recent and contemporary vehicle models.
- Dropped the null values
  - The decision to drop null values from the dataset is a deliberate choice aimed at optimizing the accuracy of our used car price prediction model. The dataset encompasses a diverse range of cars, spanning from affordable low-end models to luxurious vehicles. In this heterogeneous landscape, each car's unique set of features plays a crucial role in determining its market value.
- Units were excluded to ensure the numerical uniformity of the data.
- Duplicates were excluded from the data.
- A random sample comprising 10% of the entire cleaned dataset was selected to enhance computational feasibility.
- The price values in our dataset exhibit significant imbalance due to the diverse range of cars included, spanning from low-end to luxury vehicles. To mitigate the impact of this imbalance on our modeling process, we apply a logarithmic transformation to the prices. This helps alleviate the skewness in the distribution, ensuring a more balanced representation of price variations across the entire spectrum of cars in the dataset.
- Categorical variables were encoded using one-hot encoding.

-- This will show the dataset dimensions before and after cleaning --
|  | Before cleaning | After cleaning |
|-------------------|------------|-------------------|
| Rows | 3000040 | 92547 |
| Columns | 66 | 25 |

### For Convolutional Neural Networks

For Convolutional Neural Networks we have used 2 datasets, Both datasets are sourced from Kaggle.

Car Damage Severity Dataset: [Link to dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)

License: CC BY-NC-SA 4.0 

Car Body Type Classification Dataset: [Link to dataset](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped)

Licence : CC0: Public Domain 

#### Dataset Description
The dataset is used to predict the damage severity of the car and the car model associated with it. For convolutional neural network two datasets are used, one for classifying the severity of the damage and the other for body type classification. The car body type dataset is a comprehensive collection of images depicting various car body types, including Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, and VAN. The car damage Severity dataset is to detect the level or severity of damage that is inflicted on a car. The dataset is divided into three subsets: Training, Testing, and Validation.

The dataset can be downloaded from the link: [CNN Dataset](https://heriotwatt-my.sharepoint.com/:f:/g/personal/ss2246_hw_ac_uk/EjW1Jqpae0JNmw2qllEIcZ8BHbKvRLiBHqOccTRHffRHoA?e=Aqfsf2)

#### Dataset Examples 
- Car Damage Severity:  
  - Total Number of Classes: 3 (Minor, Moderate, Severe) 
  - Total Number of Images: Approximately 1631 images  
- Car Body Type Classification: 
  - Total Number of Classes: 7 (Convertible, Coupe, Hatchback, Pick-Up, SUV, Sedan, VAN) 
  - Total Number of Images: Approximately 7000 images 

#### Dataset Exploration
The Dataset is split into test, train, and validation, this helps to evaluate the model and helps to measure the performance of the model. Line Charts are used to visualize the loss and accuracy of both the damage model and body model. The loss and accuracy of both the damage model and body model shows that the model is learning well from the training data. It is observed that the model performance is better when it is being trained on the damage dataset rather than the body model. 
The performance of the damage and body models can be assessed using this notebook: [CNN Model](https://github.com/dmml-heriot-watt/group-coursework-sa3n/blob/main/notebooks/CNN_BodyAndDamage_NEW.ipynb)

##### Dataset Cleaning
[Link to the notebook](https://github.com/dmml-heriot-watt/group-coursework-sa3n/blob/main/notebooks/CNN_ReduceDataset_NEW.ipynb)
  - The dataset is organized, renamed, and has its counts equalized. 
  - Removed random images from each subcategory to make sure each image folder has the same Images. 
  - Resized the images to 128* 128 to reduce the size of both the datasets to just under 30MB, the damage dataset takes about 9.9 mb of storage and the body dataset takes about 19.9 mb of storage.

-- This will show the dataset dimensions before and after cleaning --

#### Before Cleaning:
| Body Model Classes | No. of images | Damage Model Classes | No. of images |
|--------------------|---------------|----------------------|---------------|
| Convertible        | 1277          | Minor                | 534           |
| Coupe              | 906           | Moderate             | 538           |
| Hatchback          | 976           | Severe               | 559           |
| Pick-Up            | 1086          |                      |               |
| Sedan              | 1008          |                      |               |
| SUV                | 1246          |                      |               |
| VAN                | 1050          |                      |               |
| **Total**          | **7549**      |                      | **1631**      |

#### After Cleaning:
| Body Model Classes | No. of images | Damage Model Classes | No. of images |
|--------------------|---------------|----------------------|---------------|
| Convertible        | 351           | Minor                | 457           |
| Coupe              | 351           | Moderate             | 457           |
| Hatchback          | 351           | Severe               | 457           |
| Pick-Up            | 351           |                      |               |
| Sedan              | 351           |                      |               |
| SUV                | 351           |                      |               |
| VAN                | 351           |                      |               |
| **Total**          | **2457**      | **Total**            | **1371**      |

#### Size:
| Size of Datasets | Before Cleaning | After Cleaning | 
|------------------|-----------------|----------------|
| Body Dataset     | 1450 MB         | 13.2 MB        |
| Damage Dataset   | 14.2 MB         | 6.3 MB         |
| **Total** | **1462.2 MB** | **19.5 MB** | 

## Main Notebook 
This is the link to the main notebook: [Used Car Price Prediction](notebooks/used_car_price_prediction.ipynb)

## Clustering


#### Jupyter Notebook

[K-Means Clustering](notebooks/kmeans_clustering.ipynb)

#### Experimental design

Since clustering is an unsupervised learning technique, and since our data originally contains labels making it suitable for supervised techniques, we decided to take an alternative approach here. 
1. We used clustering to first group the prices in our dataset into 3 categories - low, mid, and high range.
2. Then, we applied k-means clustering to the independent features in the dataset (i.e., excluding price, and the derived price categories) to determine if any logical clusters are formed and mainly to identify if the cluster separations have any relation to the price values of data points.
To find the optimal value of k, the elbow method was used.
 <p align="center">
    <img src="documentation/kmeans_elbow.png?raw=true" alt="Elbow Method for Number of Clusters" width="350"/>
</p>

To visualize the results, we used Principal Component Analysis (PCA) and colored data points based on the clusters. To be able to compare the clustering results, we used PCA to visualize the dataset and color data points based on the price categories. Findings are shown in the Results section below.


#### Results

<p align="center">
    <img src="documentation/kmeans_cluster_viz.png?raw=true" alt="Elbow Method for Number of Clusters" width="350"/>
    <img src="documentation/kmeans_price_categ_viz.png?raw=true" alt="Elbow Method for Number of Clusters" width="350"/>
</p>

Clearly, there is a dominance of the mid-price category in all clusters. Therefore, as there is no clear separation between clusters in terms of price, we are not going further into targeted techniques based on the clusters formed. But from applying this technique, we were able to see the distribution of our dataset.

#### Discussion

- For the first part, we could have used our knowledge of the market prices for used cars to arrive at the price range for each category. Instead, we use k-means clustering to group the price data points. 
- We use Principal Component Analysis for visualizing because to actually view the results of clustering, we need our data to be 2 dimensional. For this, we use the PCA dimensionality reduction technique which sort of expresses the 2 principal components, which we will get from using this technique, as a linear combination of all the original features that capture maximum variance in the data (so as to capture maximum variability or more patterns in the data).
- Our dataset for used-cars price prediction contains labeled data. i.e., the dataset contains price (dependent variable) values. Therefore, the ideal way would be supervised learning.
- In order to perform clustering, we encoded all variables and performed k-means clustering on all columns except price and price_category.
- On analyzing the clusters, we see that the average price for most of the clusters falls into the low or medium price categories.
- This indicates clustering might not be a useful technique for our specific task. No cluster stands out from the visualization as well.


## Decision Trees

#### Jupyter Notebook

[Link to notebook](notebooks/DecisionTrees_RandomForests.ipynb)

#### Experimental design

Input features: Feature importance was computed for all the features and the top 10 features with high feature importance were considered to train the decision tree and random forest models.

For Decision Trees, the input features are as follows:
| Feature | Importance |
|--------------------------|-------------|
| mileage | 0.419323 |
| torque | 0.180358 |
| year | 0.149840 |
| horsepower | 0.123986 |
| wheel_system_FWD | 0.010816 |
| highway_fuel_economy | 0.009059 |
| model_name_Escape | 0.008290 |
| daysonmarket | 0.006534 |
| body_type_SUV / Crossover| 0.004353 |
| city_fuel_economy | 0.004033 |

For Random Forest, the input features are as follows:
| Feature | Importance |
|--------------------------|-------------|
| mileage | 0.407852 |
| torque | 0.159786 |
| year | 0.159639 |
| horsepower | 0.142186 |
| wheel_system_FWD | 0.013135 |
| highway_fuel_economy | 0.011248 |
| model_name_Escape | 0.007613 |
| daysonmarket | 0.006516 |
| body_type_SUV / Crossover| 0.004985 |
| engine_displacement | 0.004197 |

In each scenario, the feature "model_name_Escape" has been omitted, as it represents a specific model among a variety of others.

Output label: Price

#### Algorithms used

Since it is a supervised regression task, decision trees and random forest regressor algorithms were used.

#### Visualizations

1. Scatterplots with actual vs predicted price values were plotted to visualize how the models’ predictions fare compared to actual values. The plots were linear indicating that the predictions were close to the actual values.
 <p align="center">
    <img src="documentation/Scatterplot_DecisionTrees.png?raw=true" alt="Predicted vs Actual Values - Decision Trees" width="350"/>
    <img src="documentation/Scatterplot_RandomForest.png?raw=true" alt="Predicted vs Actual Values - Random Forest" width="350"/>
</p>

2. Learning curves for both decision trees and random forest regressor show if the models are underfitting or overfitting. In our case, the models are learning well from the training data. However, there is scope for improvement. This is a conscious choice since we’ve prioritized computational efficiency over performance.
<p align="center">
     <img src="documentation/LearningCurves_DecisionTrees.png?raw=true" alt="Learning Curve of Decision Tree Regressor" width="350"/>
     <img src="documentation/LearningCurves_RandomForest.png?raw=true" alt="Learning Curve of Decision Tree Regressor" width="350"/>
</p>

#### Results

| Model         | Data         | MSE    | R-squared score |
| ------------- | ------------ | ------ | --------------- |
| Decision Tree | Training Set | 0.0186 | 0.9438          |
|               | Test Set     | 0.0297 | 0.9115          |
| Random Forest | Training Set | 0.0074 | 0.9775          |
|               | Test Set     | 0.0208 | 0.9379          |

#### Hyperparameter Tuning

Hyperparameter tuning helps in identifying the optimal parameters that result in the best predictions for the model and helps the model to generalize better to unseen data, improving overall performance.

Randomized Search Cross-Validation (CV) was used for hyperparameter tuning of both decision trees and random forest regressors because it is more computationally efficient than an exhaustive grid search and provides a more manageable way to explore different combinations.

#### Discussion

For the regression task at hand, Random Forest Regressor outperformed Linear Regression and Decision Trees. The table below summarizes the performance of the three models:
| Model | MSE | R-squared score |
| ------------------ | ------- | -------------- |
| Linear Regression | 0.0505 | 0.8495 |
| Decision Trees | 0.0297 | 0.9115 |
| Random Forest | 0.0208 | 0.9379 |

Linear Regression assumes that the features and target variable have a linear relationship. However, for our dataset, the relationship is not perfectly linear. Therefore, Decision Trees and Random Forest Regressor have better performance since they can capture the complex patterns and interactions among features better. Random Forest is an ensemble learning method that builds multiple decision trees and combines their predictions. This leads to more robust and accurate predictions compared to individual decision trees.

The hyperparameter tuning has helped the performance because tuning helps find the parameters at which the model’s performance is optimized.

[Link to Linear Regression implementation](notebooks/LinearRegression.ipynb)


## Convolutional Neural Networks

#### Dataset Redesign

The dataset for body type is preprocessed using the following steps:

1. The images from each folder (representing classes) in the folders 'test', 'train', and 'validation' are combined into a single folder for each class of image.
2. The files of each folder that represent classes are verified to be images.
3. In order to make sure that the model being trained is not biased, the number of images in each folder is identified. The folder with the lowest number of images is then noted. Afterwards, random images from each folder is deleted such that all folders will have the lowest number of images which was noted earlier.

#### Loading and Preprocessing Data

The dataset is further split into three, namely, 'train', 'validation', and 'test' using the package 'keras' which is a high-level neural networks API. Among the batches of data, 80% is used by train, 10% is used by validation, and 10% is used by test.

After loading the data, preprocessing is done on the image data by scaling it down to achieve a value between 0 and 1.

#### CNN Architecture

The package 'keras' has been used to implement the CNN model. 

The architecture used by the model is as follows:

1.  Sequential model:\
    Firstly, the sequential model is used to initialize a stack of linear layers having a single input and output. This sequential will have multiple other layers inside it.
    

2.  Convolution layers:\
    Multiple convolution layers are being used. By using convolution layers we are able to create multiple filters that will aid the model in recognising a feature from an image.
    The convolution layers being used are:
    1. Conv2D(32, (3,3), 1, activation='relu', input_shape = (256,256,3)):\
     This layer contains 32 filters which are 3 by 3 pixels in size, it has a stride of 1, uses the activation function 'relu', and since this is the first convolution layer, it has an input shape indicating that the images are used are 256 by 256 pixels in  size with 3 color channels (RGB).
    2. Conv2D(16, (3,3), 1, activation='relu'):\
     This second layer contains 16 filters which are 3 by 3 pixels in size, with a stride of 1, and uses an activation function 'relu'.
    3. Conv2D(32, (3,3), 1, activation='relu'):\
     This third layer contains 32 filters which are 3 by 3 pixels in size, with a stride of 1, and uses an activation function 'relu'.

3.  Pooling Layers:\
    After each convolution layer, a pooling layer is used to reduce the dimensions of the input for the next layer. This aids the model in reducing computing time and overfitting.
    1. MaxPooling2D():\
     This layer scans over the output of the convolution layer using a window of size 2 by 2 pixels. Then it proceeds to take the maximum value from each window.

4.  Dropout Layers:\
    A dropout layer is used to prevent overfitting by ensuring that the model is not overly reliant on any specific neuron. This is ensured by dropping or deactivating a random set of neurons.

    1. Dropout(0.1):\
       This layer is used after the convolution layer that uses 32 filters or convolutions.
       During each training iteration, this layer randomly sets the output of 10 percent of neurons to zero.
    2. Dropout(0.05):\
       This layer is used after the convolution layer that uses 16 filters or convolutions.
       During each training iteration, this layer randomly sets the output of 5 percent of neurons to zero.

5.  Flatten Layer:\
    This layer effectively flattens the values of the convolution layers into a single value.

    1. Flatten():\
       The layer is added after all the convolution, pooling, and dropout layers.

6.  Dense Layer:\
    This is a fully connected layer that takes as input the value of the flattened layer and uses an activation function.
    1. Dense(256, activation='relu'):\
      This dense layer is added after the flattening layer and it uses 256 neurons and the activation function 'relu'. The decision to choose 256 neurons is taken because it provides a good balance between model efficiency and complexity.
    2. Dense(number of classes, activation='softmax'):\
      This dense layer is added after the previous dense layer and is the final layer with the number of neurons equal to the number of classes where each represents a single category. The activation function used is 'softmax', this is used for multi-class classification, and outputs a probability distribution over the classes.

7.  Compilation:\
    The model is then compiled using the loss function 'sparse_category_crossentropy' which is suitable for multi-classification tasks and uses the 'adam' optimizer and measures the 'accuracy' as a metric.

Thus, this model will be able to take as input RGB images of size 256\*256 and classify them into a given number of classes.

#### Type Of Models

The model architecture specified above has been used to implement two models, namely:

1. Damage Model: A Model to recognize the severity of damage inflicted on a car.
2. Body Model: A Model to recognize the body type of a car.

#### Visualisations

Visualizing the accuracy and loss metrics of the damage model:

<p align="center">
       <img src="documentation/dmAccuracy.png? raw=true" alt="Accuracy of Damage Model" width="350"/>
       <img src="documentation/dmLoss.png? raw=true"alt="Loss of Damage Model" width="350"/>
  </p>
  
Visualizing the accuracy and loss metrics of the body model:

<p align="center">
       <img src="documentation/bodyAccuracy.png? raw=true" alt="Accuracy of Body Model" width="350"/>
       <img src="documentation/bodyLoss.png? raw=true"alt="Loss of Body Model" width="350"/>
  </p>

Visualizing the predictions done on the test data for body and damage models:

| Damage Model | Body Model |
|:------------:|:----------:|
| ![Predictions of Damage Model on test data](documentation/dmPredicted.png?raw=true) | ![Predictions of Body Model on test data](documentation/bodyPredicted.png?raw=true) |

Confusion matrix for both body and damage models:

| Damage Model | Body Model |
|:------------:|:----------:|
| ![Confusion matrix of Damage Model](documentation/dmConfusionMatrix.png?raw=true) | ![Confusion matrix of Body Model](documentation/bodyConfusionMatrix.png?raw=true) |


#### Results

| Model         | Data          | Accuracy | 
| ------------- | ------------  | -------- | 
| Damage Model | Training Set   | 84.24 %  | 
|              | Validation Set | 75.70 %  | 
|              | Test Set       | 90.10 %  |
| Body Model   | Training Set   | 91.56 %  |
|              | Validation Set | 77.09 %  | 
|               | Test Set      | 70.09 %  |

#### Discussion

The convolution neural network architecture implemented is common for both models. Both of the models are initialized with the defined model architecture. The model performs really well when it is being trained on the damage dataset than on the body model. Initially, the model performance was suboptimal. However, after tweaking and regularising the model, a noticeable improvement was observed.


### Conclusion

In this project, we delve into two primary aspects: predicting used car prices and employing Convolutional Neural Networks (CNN) to discern a car's body type and detect damage. For used car price prediction, clustering identifies data distribution and outliers, while decision trees identify key features that influence pricing. Our validated hypothesis asserts that the random forest regressor surpasses both decision trees and linear regression in predicting the price. 

Additionally, car pricing is influenced by various factors, and for personalized predictions, visual elements are crucial. The body type and damage incurred significantly impact car prices. To enhance price prediction, we integrated these factors with historical used car sales data. Employing CNN further refined our model's ability to consider visual aspects in this multifaceted prediction process.  

The dataset is deemed realistic as it encompasses a diverse range of cars, spanning from low-end to luxury vehicles. The insights into key features influencing prices align with intuition, instilling confidence in the models' potential performance if deployed. However, it's crucial to acknowledge a trade-off between performance and computational efficiency in our models. While the accuracy may not be flawless, it is sufficiently high to provide valuable insights into the influential features and offer a reliable estimate of prices for a given set of features. If deployed, potential challenges related to computational efficiency could be mitigated by optimizing algorithms or leveraging parallel processing capabilities, ensuring a balance between accuracy and efficiency.
