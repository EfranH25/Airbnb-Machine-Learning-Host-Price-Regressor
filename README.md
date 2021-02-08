# Airbnb Machine Learning Host Price Regressor
<br />
<p align="center">
  <h3 align="center">Airbnb Price Regressor</h3>
  <p align="center">
    <img width="450" height="180" src="https://github.com/EfranH25/Airbnb-Price-Regressor/blob/main/logo.png">
</p>
  <p align="center">
    Hello World! This is a machine learning project where I try to create some model that can accurately predict the price of airbnb listings based on their characteristics. 
    <br />
    <a href="https://github.com/EfranH25/Gender-Wage-Gap-1985"><strong>Explore the Repo »</strong></a>
    <a href="https://drive.google.com/file/d/1o7xOckNXH_Ay-jqTiPslZ7wzCkazX2o1/view?usp=sharing">View Presentation</a>
    ·
    <a href="https://drive.google.com/file/d/19-jZxUslRl81f5ewzPU4jF2FEappSz8f/view?usp=sharing">View Writup</a>

  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Background](#Background)
  * [Summary](#Summary)
  * [Data](#Data)
  * [Tools](#Tools)
  * [Structure](#Structure)


<!-- Background -->
## Background
Airbnb can be a competitive platform for hosts to capture the attention of potential clients. One key factor that plays a role in whether someone books an Airbnb is the price, that is why it is important for hosts to know how to best price their offerings based on its characteristics and client interests. This machine learning regressor is an initial step at producing a model that can focus in on what potential characteristics best correlate with a price of an Airbnb in the Boston area so hosts can more accurately predict and chooses their prices. 

In the future, I may expand the database I used to train my model with more information and potentially publish a web application so that others can predict their Airbnb location’s price from my model.
### Summary
To summarizes what has already been accomplished thus far, I have cleaned the data, produced various visualizations for each of the variables, addressed null values, produced more descriptive features to offer more insight on the data, evaluated various feature combinations, tuned, and recorded the outcomes of different ML models. After visualizing, cleaning, preprocessing, and feature engineering the aggregated data, I managed to produce various machine learning regression models that scored a relatively high prediction R2 scores of greater than 0.55! The best model thus far was an XGBoost model that scored and R2 value of 0.582. and only 15 predictive features. I am confident that there are even further performance gains to be made with this model with further hyperparameter optimizations, but I have currently put the rest of this project on hold as I explore other ML projects. 

### Data
To acquire the data for this project I initially created a web scrapper for Airbnb’s website to gather all the information I needed regarding the various available locations in Boston. Unfortunately, the web scrapper did not perform well due to Airbnb’s website limiting such functionality. Fortunately, there existed a website, insideairbnb.com, that had already scrapped data for various Airbnb locations, including Boston, so I used their 31 August 2020 Boston listing as the main data source for this project. 

One a side note, regarding my failed web scraper, I have turned it a web scrapper for NewEgg.com to gather GPU data to analysis. If you are interested, please check out my repo for it: 
Project Link: [https://github.com/EfranH25/Newegg-Web-Scrapper-GPU-Database-Creator]( https://github.com/EfranH25/Newegg-Web-Scrapper-GPU-Database-Creator)

### Structure
Here is just a quick summary of the file structure of this repo:

*The figures folder contains all the images I produced while exploring my data. There is also a Tableau visualization file there for some of the images that were to hard to read through Matplotlib. 

*The Input folder has all the inputs and data I used for this project. 

*The records folder has stored the score of the various machine learning models I produced

*The tracking list folder simply held various lists I saved when working on this project.

*The src folder contains all the python code I used for this project as well.  There are several python scripts inside of it.The create_folds.py file was the first script used. It simply scrambled up my data set and created a column for cross validation purposes. The explore.py script was the next thing created and it has various functions to visualize and analyze all the features in the data. Final the train.py script has all the functions regarding preprocessing, featuring engineer, and model selection.


### Tools
* Python
* Pycharm
* scikit-learn
* SQL
* XGBoost
* Pandas
* Tableau
















