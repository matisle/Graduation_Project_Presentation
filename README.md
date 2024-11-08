Customer Satisfaction of the German Energy Supply Chain: 
Predicting Star Ratings and Company Responses on Trustpilot
==============================

This is the graduation project  of Matthias Isele and  Stefanie Arlt for Data Scientist formation in April 2024.
Our subject is energy suppliers in Germany on the Trustpilot website, from which we scraped content in September 2023 in German language in 2 data sets:
* Overall ranking list of 37 suppliers including rating score, number of votes and supported energy supply categories @Stefanie.
* Customer reviews, rating and company answers dating back to 2011 with more than 3000 pages of content @Matthias.
* The whole data set ist obtained via a left join on the company variable.

In our study, we scraped the data, cleaned it and created 2 data sets, which were both explored including visualizations of key facts and stories.
The customer review file was the basis for a machine learning exercise: 
* Predicting the star rating based on customer comments  @Stefanie
* Predicting the length of company answers to customer postings, or company answers, if possible. @Matthias

Challenges for us were the imbalanced data set and also the large amount of data itself.
However, starting with engineered features like number of words we could improve model performance with sentiment analysis.
With keyword analysis we could visualize the most important points of interest of the customers for two exemplary suppliers.
Finally, we put a model in Streamlit to production giving the answer of E.ON Energy to customer comments. 

Monitoring customer feedback and regularly assessing customer satisfaction are key objectives in customer service management.
With our study we could show the potential of machine learning in tailoring personalized customer interaction, leveraging the large data resources of customer comments and supplier feedback. 


For a quick overview of our project, please refer to the management summary in our final report. (reports folder.)

Project Organization
------------
    ├── notebooks          <- Contains all the notebooks used in the development process as well as modeling.
    │
    ├── reports            <- Contains the reports during the development process as well as the final report.
    │  
    ├── streamlit          <- Contains the Streamlit-App of this project.
    │  

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
