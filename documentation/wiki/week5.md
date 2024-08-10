### Data Preprocessing

Since the dataset is huge, we decided to perform preprocessing operations in Python to bring it down to a workable size, all the while not compromising on the quality of the data. 

The dataset is about used cars in the US market. There are 66 columns of data representing various features of the cars. Upon preliminary analysis, I found that there are columns that are specifically related to commercial vehicles like pickup trucks and such. These columns had no values for passenger vehicles and had other values for commercial vehicles. So, using the dropna() function of python, all the rows with non-missing values for these columns were removed, since our focus is on passenger vehicles i.e. cars. 

E.g.: cars = cars[~cars['bed_height'].notna()]

Then, the columns which do not impact the pricing of the used cars were removed using the drop() function and the total number of columns were brought down to 30. 

Then, the missing values in the existing columns were removed since we have enough data even without addressing these missing values by other means, which brought down the number of rows to ~1 million from 3. This dataset was exported as a CSV file. This file is 230MB. Git doesn't support this large files unless LFS is used. 

The proposal is to get a random sample from this data using the df.sample() function. Each of us will get a random sample, plot the data and find if it's representative enough. We will choose the dataset that is the most representative and work on it for further analysis. 

The jupyter notebook with preprocessing code has been uploaded under the notebooks section on Git. 

### Data Visualisation

As per the guidance we received, we have started to dive into the visualisation aspect of this project. We decided to understand the distribution of car sales across different manufacturers, this was achieved by implmenting a bar chart that visually represents the number of cars sold by each manufacturer in our dataset. This implementation involved using pandas, matplotlib and the 'value_counts()' function to count the occurances of each manufacturer in the dataset.
