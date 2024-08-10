## Date: 22/09/2023

Proposal by Amrutha Purna Vadrevu (@av2028)

## Title

Sign Language Interpretation using Machine Learning Techniques

## Introduction

Sign language is a nonverbal language that conveys meaning through hand gestures, facial expressions, and body language. It is the primary language for deaf and hard of hearing people, but it is also used by many other people, such as hearing people who communicate with deaf family members or friends, and people who work in professions where sign language is needed, such as teachers, interpreters, and social workers.

Sign language interpretation is the process of translating spoken language into sign language, or vice versa. Sign language interpreters play an important role in society, helping to ensure that deaf and hard of hearing people have access to the same information and opportunities as hearing people.

In recent years, there has been growing interest in using image classification techniques to develop sign language recognition systems. These systems have the potential to improve the accuracy and efficiency of sign language interpretation, and to make sign language more accessible to people who do not know sign language.

In this coursework, we aim to develop machine learning models that can interpret the sign language with considerable accuracy. 

## Questions

	1. Can machine learning models be used to interpret sign language?
	2. Data Preprocessing - How to augment the data for better accuracy?
	3. Modelling - Which generative models are applicable for the data?
	4. Would decision tree algorithms improve the accuracy?
	5. Would neural network models improve the accuracy? If so, how are they better than the other models?

## Dataset

The dataset is taken from Kaggle - https://www.kaggle.com/datasets/datamunge/sign-language-mnist
Size: 66MB

The dataset format is patterned to match closely with the classic MNIST. Each training and test case represents a label (0-25) as a one-to-one map for each alphabetic letter A-Z (and no cases for 9=J or 25=Z because of gesture motions). The training data (27,455 cases) and test data (7172 cases) are approximately half the size of the standard MNIST but otherwise similar with a header row of label, pixel1,pixel2….pixel784 which represent a single 28x28 pixel image with grayscale values between 0-255. 

## Project Plan

Stage 1. Preprocessing the dataset for data quality and performing exploratory data analysis. 

Stage 2. Implementing an appropriate generative model to learn the underlying patterns or distributions of the data.

Stage 3. Applying decision tree algorithms to build predictive models and analyse if the application is appropriate for this data. 

Stage 4. Implement neural network models and check if the accuracy improves. Compare the performance against the previous models. 

Stage 5. Draw appropriate conclusions and present them visually. 

-----------------------------------------------------------------------------------------------------------------------------------------
	

## Date: 24 September 2023

Proposal by Aashima Poudhar (@aashimapoudhar @ap2099)

## Title

Student Academic Completion / Dropout Prediction

## Introduction

Students begin their undergraduate journey with the aim and hope of successfully completing the course. But there may be various factors which cause a student to drop out or fail such as previous qualification (if experienced, students may be able to grasp concepts faster), financial status (whether they are and will be able to pay their fee), whether the student is an international student or not, and so on.

The idea here is to predict the  chances of a student failing or droppping out from their program at an early stage with the aim to provide early support and help the student succeed.

## Questions

  1. What percentage of students dropout every year?
  2. What is a prominent reason behind failure / dropout?
  3. Is failure / dropout more common in male or female students?

## Dataset

The dataset is from UC Irvine Machine Learning Repository - https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

## Project Plan

Commonly listed above
