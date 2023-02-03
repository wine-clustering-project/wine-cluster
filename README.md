# Wine Quality

# Project Goals

 - Identify what drives wine quality 
 - Build a model to best predict wine quality 

# Project Description

We are looking to come up with a machine learning model that will help us see which features gives us the best indicators of wine quality and also be able to predict the quality of the wine. These wines are scored from 0 - 10 on a quality scale, with 0 being very bad to 10 being very excellent. This data was collected from the vinho verde samples from Portugal. After we have explored and made our models we will recommend what features are best to help predict wine quality and give usesful insights on our data.

# Initial Questions

 1. Do red or white wines have a higher quality score?
 2. Does alcohol impact quality positively or negatively?
 3. Does ph level affect the wines differently?
 4. Is there a relationship between sugar and quality?


# The Plan

 - Create README with project goals, project description, initial hypotheses, planning of project, data dictionary, and come up with recommedations/takeaways

### Acquire Data
 - Acquire data from data.world and create a function to later import the data into a juptyer notebook to run our notebook faster. (acquire.py)

### Prepare Data
 - Clean and prepare the data creating a function that will give me data that is ready to be explored upon. Within this step we will also write a function to split our data into train, validate, and test. (prepare.py) 
 
### Explore Data
- Create visuals on our data 

- Use clustering techniques to observe if there are any insights we can gather from our data

- Create at least two hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, document any findings and takeaways that are observed.

### Feature Engineering:
 - Scale our data for our models
 
 - Create dummies to encode categorical variables for our models (if needed)

### Model Data 
 - Establish a baseline(mean or median of target variable)
 
 - Create, Fit, Predict on train subset on four regression models.
 
 - Evaluate models on train and validate datasets.
 
 - Evaluate which model performs the best and on that model use the test data subset.
 
### Delivery  
 - Create a Final Report Notebook to document conclusions, takeaways, and next steps in recommadations for predicitng wine quality. Also, inlcude visualizations to help explain why the model that was selected is the best to better help the viewer understand. 


## Data Dictionary


| Target Variable |     Definition     |
| --------------- | ------------------ |
|      quality      | score of wine, scale between 0 - 10  |

| Feature  | Definition |
| ------------- | ------------- |
| fixed acidity | measures the level of acidity that doesn't change with storage or wine aging |
| volatile acidity | measures the amount of acetic acid in wine, responsible for vinegar-like smell  |
| citric acid | measures the amount of citric acid in wine, provides fresh and sour taste |
| residual sugar | measures the amount of residual sugar in wine, affects the wine's sweetness |
| chlorides | measures the amount of chloride ions in wine, contributes to wine's saltiness |
| free sulfur dioxide | measures the amount of free SO2 in wine, acts as a preservative | 
| total sulfur dioxide | measures the combined amount of free and bound SO2 in wine |
| density | measures the weight per unit volume of wine, affects wine's body and mouthfeel |
| pH | measures the acidity level of wine, affects wine's taste and stability |
| sulphates | measures the amount of sulfur dioxide in wine, contributes to wine's preservation and taste |
| alcohol | measures the percentage of alcohol by volume in wine, affects wine's strength and flavor |


## Steps to Reproduce

- You will need to download the 2 csv's from data.world (https://data.world/food/wine-quality)

- Clone my repo including the acquire.py, prepare.py, explore.py, and modeling.py (make sure to create a .gitignore to hide your csv files since it will not be needed to upload those files to github)

- Put the data in a file containing the cloned repo

- Run notebook

## Conclusions
 
Wine quality predictions were used by minimizing RMSE within our models. Type and alcohol have proven to be the most valuable, but there is still room for improvement.
 
Best Model's performance:

- Our best model reduced the root mean squared error by .13881 compared to the baseline results.(16% better)

- RMSE 0.732087 on in-sample (train), RMSE 0.750386 on out-of-sample data (validate) and RMSE of 0.789499 on the test data.

## Recommendations
- We would recommend using type and alcohol to build models to predict wine quality. 

- We would also recommend collecting more data on what other ingredients were used such as grapes.

## Next Steps
- Remove outliers, and explore other features using clustering techniques

- Consider adding different hyperparameters to models for better results. 
