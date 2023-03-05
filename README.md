# Flight Delay Analysis

## Abstract

In this project, based on the 2009 ASA Statistical Computing and Graphics Data Expo, we pick the data of `2006 - 2007` and applied data analysis method to solve the following questions:

1. When is the best time of day, day of the week, and time of year to fly to minimise delays? 

2. How does the number of people flying between different locations change over time?

3. Can you detect cascading failures as delays in one airport create delays in others?

4. Use the available variables to construct a model that predicts delays.


## EDA&Data Preprocessing

Before look into the four questions, we will first perform exploratory data analysis and data preprocessing.

After a briefly look at the dataframe containing the data from `2006 - 2007`, we notice that when the flights are cancelled or diverted, there will be a lot `NAN` value in that record, therefore, we start by extracting the records only contain the flights that successfully take off.

`df = df[(df['Cancelled'] == 0) & (df['Diverted'] == 0)]`

After extracting, only column `CancellationCode` still contains missing value. However, this column is useless for solving the four question thus it can also be directly removed from the dataframe. Now, the dataframe no more include missing values.

## Q1. When is the best time of day, day of the week, and time of year to fly to minimise delays? 

This question is divided into three subquestions.

### Q1a. Best time of day to minimise delays

![TOD](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/TOD.png)

Being inspired by the visualization, we can know that the flights around `15:00` has the minimum avrage depature delay and average arrival delay.

![POD](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/POD.png)

We can also devide a day into four period:

- Morning: 4:00am - 11:59am
- Afternoon: 12:00pm - 7:59 pm
- Evening: 8:00pm - 10:50pm
- Night: 11:00pm - 3:59am

And we can see from the visualization that the flight on the morning, from `4:00am - 11:59am` has the minimum average delay on both arrival delay and depature delay.

### Q1b. Day of the week to minimise delays

![DOW](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/DOW.png)

Same, from the visualization we can know that `Tuesday` has the minimum delay on depature while `Saturday` has the minimum delay on arrival.

### Q1c. Time of year to minimise delays

![TOY](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/TOY.png)

'November' is the best time of year to minimise arrival delay and `September` is the best time of year to minimise depature delay.

## Q2. How does the number of people flying between different locations change over time?

There are several methods to analysis this problem, here we introduce three solutions:

1. Lineplot based on number of people, year and month

2. Heatmap

3. Line plot based on month, number of people and destination

### Lineplot

![Lineplot](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/Lineplot.png)

From this plot we can compare the number of people flying in different years accross different months. We notice that compared to `2006`, more people take a flight on `2007`. Beside, we also notice that people usually don't like to take a flight on `Febuary` and the peak of number of people taking a flight is on `August`.

### Heatmap

![Heatmap](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/Heatmap.png)

For the heatmap, a brighter color means a higher number of people flying while a deeper color means less people are willing to take a flight.

### Lineplot with destination

![LWD](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/LWD.png)

Different from the first lineplot, we can also use different lines to represent different locations to see the trends of people flying to different destination accross each month. Since there are quite a lot destinations in the dataset, we only pick the top five as example here.

## Q3. Can you detect cascading failures as delays in one airport create delays in others?

To detect the cascading failures as delays in one airport create delays in others, the essential problem is to check whether column `ArrDelay` is highly correlated to column `DepDelay`.

According to our recognition, a delay in arrival normally lead to a delay in departure. Therefore, we can try to use linear regression methods to test whether their is a tight correlation between `ArrDelay` and `DepDelay`.

![Q3](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/Q3.png)

As we can directly see from the regression plot, feature `ArrDelay` has an obivous relationship with feature `DepDelay`.

In order to prove it with evidence, we perform a hyphothesis test to check whether there is a tight correlation between `ArrDelay` and `DepDelay`

Here are our hyphothesis:

$H0$: `DepDelay` and `ArrDelay` are independent

$H1$: `DepDelay` and `ArrDelay` are dependent

The significance level we set is `5%`, which means that if the P-value is less than `0.05`, we will reject the null hyphothesis.

![Hyphothesis_Test](https://github.com/HQR2000/Flight_Delay_Analysis/blob/main/public/Hyphothesis_Test.png)

According to our hyphothesis test, we reject the null hyphothesis and draw a conclusion that the `ArrDelay` and `DepDelay` are highly correlated, which means that we can detect cascading failures as delays in one airport create delays in others.

## Q4. Use the available variables to construct a model that predicts delays.

To predict the time of delay, the task is to pick useful features and create predictive models to predict the `ArrDelay` column. 

### Feature Selection

For the feature selection, we first manully look into the data and delete those features that are obviously useless for making prediction.

1. Column `Cancelled` and `Diverted` can be directly removed since these columns doesn't provide useful information for the predictive model.

2. Object variables `Origin`, `Dest`, `FlightNum` and `UniqueCarrier` can be removed.

3. Columns such as `WeatherDelay` can be removed since the `ArrDelay` and `DepDelay` already contains the information of these columns.

4. Datetime data can be removed.

Then, we calculate the correlation between features and delete the feature with a high correlation with another feature to reduce the feature dimensionality. The threshold we set here is `0.95`, which means that if two features have a correlation over 0.95, we will delete one of them.

`
corr = Q4_df.corr()

threshold = 0.95 

highly_correlated = []
correlated_matrix = np.abs(corr) > threshold
for i in range(len(correlated_matrix.columns)):
    for j in range(i):
        if correlated_matrix.iloc[i, j]:
            colname = correlated_matrix.columns[i]
            highly_correlated.append(colname)
           
Q4_df = Q4_df.drop(highly_correlated, axis = 1)
`

### Predictive Models

After feature selection, we create three regression models to predict the delays and compare their performance.

These three models are:
1. Linear Regression

2. Lasso Regression

3. Ridge Regression

For the training of models, we applied `StandardScaler()` for the standard scaling of numerical data. And for parameters finetuning, we applied `GridSearchCV()` to find the best parameter for each model.

**Accuracy**

| Model                | Accuracy             | 
| -------------------- | :-------------------:|
| `Linear Regression`  | 0.9156553636102837   |
| `Lasso Regression`   | 0.9156553608192511   |
| `Ridge Regression`   | 0.9156553608192511   |

---
**Mean Square Error**

| Model                | Accuracy             | 
| -------------------- | :-------------------:|
| `Linear Regression`  | 121.29710095180282   |
| `Lasso Regression`   | 121.29814118150038   |
| `Ridge Regression`   | 121.29710496562252   |

---

Since `Lasso Regression` and `Ridge Regression` are only `Linear Regression` adding `l1` and `l2` regularization, their performance are similar.









 



