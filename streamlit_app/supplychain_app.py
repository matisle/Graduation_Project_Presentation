# import
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# setup page and sidebar
st.title("Customer Satisfaction of the German Energy Supply Chain")
st.sidebar.title("German Energy Suppliers on Trustpilot")
pages=["Introduction: The Project and the German Energy Market", "Webscraping", "Energy Suppliers - Company Level", "Energy Suppliers - Customer Level", "Modeling: Predicting Star Ratings and Keyword Analysis", "Modeling: Predicting Company Answers", "Outlook: Practical Application"]
page=st.sidebar.radio("Learn more about ...", pages)
#st.sidebar.title("Authors")
#st.sidebar.link_button("Stefanie Arlt", "https://www.linkedin.com/in/stefanie-arlt-augsburg-germany/?locale=en_US")
#st.sidebar.link_button("Matthias Isele", "https://www.linkedin.com/in/mat-isle-24a665270/")


# page 0: Introduction to German energy suppliers, Trustpilot and the project goal - Stefanie
if page == pages[0] : 
  st.header("Predicting star ratings and company responses on Trustpilot", divider='green')
  st.image("strommast.jpg", caption="Access to electricity is a fundamental right in Germany")
  st.subheader("About the Project")
  st.markdown("The subject of this data science graduation project at DataScientest is, to **investigate customer satisfaction for energy suppliers in Germany**. The data set consists of two files, scraped from the **Trustpilot web site in September 2023** in German language:")
  st.markdown(" - The **overall ranking for 37 energy suppliers in Germany** based on the Trustpilot scores, including number of votes, as well as supported energy supplier categories")
  st.markdown(" - The **customer reviews and company answers for these suppliers**, consisting of more than 3000 pages, including times of posting and dating back to 2011")
  st.markdown("Our project work included the **scraping** of the data, as well as **cleaning** the data sets and **enigneer features**. After exploring and **analysing the data**, we created **visualizations** of our key findings.")
  st.markdown("In the **machine learning** part, we **predicted star ratings and company responses**, first with simple models and later supported by **sentiment analysis**. Last but not least, we included a **keyword analysis** as practical application in the field of customer service management.")

  # German energy market
  st.subheader("The German energy market")
  st.markdown("Since the liberalization of the European energy market in the late 1990ies a **harmonized regulatory framework** enabled a competitive market for energy products and services.")
  st.markdown("At the core of the regulation, we see **unbundling** as a key concept, which involves the separation of energy production, transmission, and distribution activities to ensure fair competition and prevent monopolistic practices. Consequently, the vertically integrated energy companies in Germany have been separated into distinct entities.")
  st.image("Liberalization2.png", caption="Liberalization of energy markets in Europe")

  st.markdown("The **German energy market** is one of the largest in Europe, not only catering to its domestic demand but also interacting with other markets in Europe. In 2023, the net electricity generation in Germany amounted to 485 billion kWh, as opposed to a net power consumption of 466 billion kWh. Private households together with business and services consummated nearly the same as industry and traffic combined.")
  st.markdown("Today the market is still dominated by **four large energy supply companies: E.ON, RWE, EnBW and Vattenfall**, since 2020 also joined by LEAG, formerly part of Vattenfall. These players include numerous subsidiaries and are involved in various segments of the **energy value chain**, including generation, distribution, and retail of electricity and gas.")
  st.markdown("There have been significant changes and **developments in the last decade**, driven by various political programs and legal acts. The expansion of renewable energy, the phase-out of nuclear power, energy efficiency measures, e-mobility, digitalization, and the rise of decentralization have all shaped the market landscape.")

  # Focus = household customers
  st.subheader("The household customers") 
  st.markdown("We will **focus in this study on the B2C segment**: In 2022, end consumers could choose on average from **nearly 160 energy providers**, without taking corporate connections into account.")
  st.markdown("However, most household customers still buy their electricity from their **local providers**. That means, after 20 years of liberalization, only around 40% of the energy for private households is bought from energy suppliers outside of their local network area.")

  # Comparing rates and opinions
  st.subheader("Comparing rates and opinions")
  st.markdown("**Energy contracts** are differing in terms and obligations of the suppliers and the prices, but come usually with monthly installments and annual statements, with an initial contract term of 12 months.")
  st.markdown("In addition to the overall price, contracts outside of the basic supply can offer a bonus, a price guarantee, green energy and other features that help suppliers compete for customers")
  st.markdown("Due to **fluctuating energy prices** in the last years, the number of bancruptcy of energy suppliers increased. Also, **consumers became more careful** and usually consult internet platforms like Check24 or Verivox to compare prices and conditions for their contracts.")
  st.markdown("**Exchanging opinions and commenting on suppliers’ services** on rating sites like Trustpilot became more important, helped along by increasing online activities in Germany during the lock-down periods in the Covid pandemic.")

# page 1: Webscraping - Matthias
if page == pages[1] : 
  st.header("Webscraping on Trustpilot", divider='green')
  st.subheader("The scraping procedure")
  st.markdown('Target of the scraping are **ratings, customer votes and supplier feedback of energy suppliers in Germany on the Trustpilot** web site. The content is scraped to date **September 2023** in German language, as the language settings on the portal define the respective market. We tried out English language as well but then we only got UK / American energy suppliers, or comments in English language only from English speaking expats currently living in Germany, which is only a small percentage of the local customer base.')
  st.markdown('We scraped content on **two levels**: The first dataset contains **general information of energy suppliers in Germany (supplier level)**, while the second dataset collects for each energy supplier **customer reviews and supplier answers (customer level)**. The total dataset may be obtained by joining both files on the supplier column.')
  st.markdown("Trustpilot detected that we are scrapers and blocked us each time after scraping around 300 pages. To **counter anti-scraping measures** we implemented **random time delays** and **rotating User-Agents**. It was now possible to scrape around 700 pages until getting blocked. In total we scraped all 2250 pages for German energy suppliers, i.e. 45000 posts.")
 
  st.subheader("The Company Level")
  st.markdown("The Trustpilot rating information on **37 energy suppliers in Germany** includes star rating, number of votes, as well as supported energy supplier categories, e.g. eco power, solar energy etc.")
  
  st.image("company_level.PNG")
  #st.subheader("Data preparation")
  st.markdown("In the scraped data file, there were some **values missing or needed to be checked** for clarification: Suppliers with 0 votes were deleted; missing city information was researched and replaced. After a reset of index and uneventful check for special chars, the data was deemed clean.")
  st.markdown("The Trustpilot search was for “energy supplier”, i.e. “Stromversorgungsunternehmen” in German. It became clear that some companies also render additional services and are listed in several categories, which were stored in the *cat* column.")
 
  # import data
  file1_data = pd.read_csv("ener_supplier_rankings_clean_final.csv")
  
  # show data frame with check box
  if st.checkbox("Show dataframe after scraping:", key="d1") :
    st.dataframe(file1_data.head(10), use_container_width=True)

  # establish engineered features
  st.markdown("Information from the ***cat*** column was exported to text for a comprehensive list of unique categories, which were then added as **engineered diversification features**. After deleting obsolete country information and correction of data types, the dataset was now ready for visualization.")

  # show columns list
  st.markdown(" - **:green[supplier] :** (object) company name;")
  st.markdown(" - **:green[city] :** (object) location of comapny headquarters;")
  st.markdown(" - **:green[eco] :** (bool) specialized power label;")
  st.markdown(" - **:green[gas, telco, energy_solutions] :** (bool) diversification labels;")
  st.markdown(" - **:green[num_votes] :** (int64) number of votes / comments;")
  st.markdown(" - **:green[score] :** (int64) company rating from 1 to 5 stars;")
  
  # import data engineered
  file1_df1 = pd.read_csv("ener_supplier_rankings_clean_no-null.csv")
  
  # show data frame with check box
  if st.checkbox("Show dataframe with engineered features:", key="d2") :
    st.dataframe(file1_df1.head(10))

  st.subheader("The Customer Level") 
  st.markdown('On the customer level we find **customer posts and, optionally, company answers**.')
# data frame einbauen und bei Stefanie rausnehmen 

  st.image("customer_level.PNG")
  st.markdown('The following attributes could be scraped directly:')
  st.markdown(" - :green[nickname] : (object) customer nickname;")
  st.markdown(" - :green[location] : (object) customer location;")
  st.markdown(" - :green[stars] : (int) star rating of customer;")
  st.markdown(" - :green[headline] : (object) headline of post;")
  st.markdown(" - :green[dop] : (datetime) date of post;")
  st.markdown(" - :green[doe] : (datetime) date of experience;")
  st.markdown(" - :green[comment] : (object) customer comment;")
  st.markdown(" - :green[answer] : (object) company answer;")
  st.markdown(" - :green[doa] : (datetime) date of answer;")
  st.markdown(" - :green[supplier] : (object) company name;")

# page 2 = Data set 1: Exploration and Visualization - Stefanie
if page == pages[2] : 
  st.header("Energy Suppliers on Trustpilot - Company Level", divider='green')

  st.subheader("Heat map")
  st.markdown("Investigating the relationship between the variables of the data set, it has been observed that **eco-friendly energy** seems positively correlated to the rating score. **Diversification in energy solutions and gas supply** is also represented as slightly positive, but **number of votes only weakly** or slightly negative.")
  #import data
  file1_df1 = pd.read_csv("ener_supplier_rankings_clean_no-null.csv")
  # radio button to choose correlation test
  from scipy.stats import spearmanr
  from scipy.stats import pearsonr

  display = st.radio('Choose correlation test:', ('Spearman', 'Pearson'))
  if display == 'Spearman':
      st.image("File1_heatmap_Spearman.png", caption="Heatmap (Spearman) for file 'energy suppliers'")
      spearman_r_votes = spearmanr(file1_df1['num_votes'], file1_df1['score'])
      st.write("Spearman r p-value (score vs. number of votes):", spearman_r_votes[1], " - coefficient:", spearman_r_votes[0])
  elif display == 'Pearson':
      st.image("File1_heatmap_Pearson.png", caption="Heatmap (Pearson) for file 'energy suppliers'")
      pearson_r_votes = pearsonr(x= file1_df1['num_votes'], y= file1_df1['score'])
      st.write("Pearson r p-value (score vs. number of votes):", pearson_r_votes[1], " - coefficient:", pearson_r_votes[0])

  # distribution of scores
  st.subheader("Distribution of scores")
  st.markdown("Although we see **limited data points**, we can observe a **tendency of higher number of votes for low ranking and higher rankings**. An explanation might be that customers tend to express more feedback when they are exceptionally happy or exceptionally unhappy.")
  ## plot
  # add column with rounded score
  ener_service = file1_df1
  ener_service['score_rounded'] = ener_service['score'].round(0)
  # add column with log transformation for num_votes
  ener_service['log_num_votes'] = np.log(ener_service['num_votes'])
  # data = ener_service,
  # display the distribution of score on number of votes
  fig=sns.catplot(x= 'score_rounded', 
              y= 'log_num_votes', 
              kind= 'box', 
              # hue = 'eco', 
              #alpha = 0.7,
              data = ener_service
             )
  st.pyplot(fig)

    # overall ranking
  st.subheader("Top and bottom five energy suppliers in Germany")
  st.markdown("As we can see, the **number of votes is spread** all over the spectrum: Small suppliers with only a few comments are neck-and-neck with big international corporations with many votes.")
  st.image("File1_bottom-top5.png", caption="Top and bottom 5 energy suppliers in Germany")


  # Diversification offering
  st.subheader("Diversification offering")
  st.markdown("The **offering of eco power is no guarantee for many (positive) votes** or a high rating. Suppliers who are also delivering gas, tend to have more votes but are distribute over scores 2 to 5. In the overall ranking, **telecommunication has no impact**, but suppliers also offering **energy systems like photovoltaic technologies** tend to have higher ranking.")
  #st.image("File1_diversification.png", caption="Impact of diversification on energy suppliers in Germany")
  # plot relationship between eco power and gas offering
  file1_data_pg= pd.read_csv("file1_data_pg.csv")
  g = sns.FacetGrid(data= file1_data_pg, 
                    col= 'gas',
                    row= 'eco',
                   hue= 'services')
  g.map(plt.scatter, 'score_rounded', 'num_votes', alpha= 0.7)
  g.add_legend()
  st.pyplot(g)
  st.markdown("Could this be an indication, that functioning business processes and customer orientation are the most important factors for a good score? Let us have a look at the customer level on the next page.")

# page 3: Data set 2 - Customer Reviews and Company Answers - Matthias
if page == pages[3] : 
  st.header("Energy Suppliers on Trustpilot - Customer Level", divider='green')
  st.subheader("User activity at Trustpilot")
  st.markdown("We see a **strong boost** of customer posts in the **pandemic years 2020-2022**. User activity **increased drastically in 2023**, partially due to the **energy crisis** caused by the war in Ukraine. The threshold of **more than 1500 comments** was **first reached 2019**, reaching **25,000 comments by 2023**. The true number of comments at the end of 2023 is expected to be higher, as the figure is to date September 2023.")
  st.image("number_comments_year.PNG")
  st.subheader("Distribution of customer votings")
  st.markdown(" Customers **tend to review** only if their experience is **on the extremes**, i.e. bad (1 star) or great or excellent (4 or 5 stars). In particular, 69% of reviews are either great or excellent, 25% of reviews are bad, and 6% of reviews are poor or average. The **distribution of customer votings** is approximately **binary**.")
  st.image("star_count.PNG")
  st.subheader("Customer satisfaction of the German energy market")
  st.markdown("The **pandemic years 2020-2022** did not bother customers negatively. The **liberalization of the German Energy Market 2019** led to increased happiness. The **energy crisis 2023** impacts happiness negatively. Data before 2019 is negligible due to low user activity. There is high volatility and company workers can easily dominate posts.")
  st.image("avg_number_stars.PNG")
  st.subheader("Distribution of comment length per star rating")
  st.markdown(" The **length of customer comments** (word count) tends to increase on average for lower star ratings, the lengthiest comment being at a star rating of 1. The Kernel Density Estimations (KDE’s) have higher variance and **skew to the right for lower star ratings**. After taking the natural logarithm, centering and normalizing, the data follows a standard normal distribution.  This means that comment wort counts are **lognormal distributed**.")
  st.image("words_comment.PNG")
  st.subheader("Distribution of answer length over all suppliers")
  st.markdown(" The length of company answers (word count) over all energy suppliers reveals a different picture. There seem to be **standard answers** (sharp peaks) and **custom answers** (smeared out parts of distributions). We expect that **companies have different answer policies**, i.e. different standard answers and different inclination towards personalized answers. For each star rating there are two maxima visible. These are related to **E.ON Energy** and **Octopus energy**, which dominate the data.")
  st.image("word_number_answer.PNG")
  st.subheader("Answer policy: E.ON Energy vs. Octopus Energy.")
  st.markdown("Comparing the **answer length variable**, we see **continuous** and **discrete** KDE's for Octopus Energy and E.ON Energy, respectively. Hence, **answers of Octopus Energy are highly personalized** whereas **answers of E.ON Energy are standard texts**.  ")
  st.image("eon_vs_oct.PNG")

# page 4: Modeling - Predicting star ratings - Stefanie
if page == pages[4] : 
  st.header("Modeling: Predicting Star Ratings and Keyword Analyis", divider='green')

  ## big data set
  st.subheader("Dataset selection")
  st.markdown("For machine learning, the focus was on the **customer level**, i.e. the second dataset containing customer information and suppliers’ answers. The first goal was to predict the star rating of customer posts from customer comments.")
  st.markdown("To develop the most practical and appropriate approach, simple models were applied first on the complete data set, but due to long run times and differences in suppliers’ answer policies, the following subsets were built, dividing the dataset down to a more manageable size while preserving logical cohesion.")
  st.image("Reduction_dataset.png", caption="Refinement of dataset size")
  st.markdown("Furthermore, the **engineered features** could be applied, to limit the dataset to posts with customer comments.")

  # show columns list big data set
  st.markdown(" - **:green[Stars] :** (int64) customer rating from 1 to 5 stars;")
  st.markdown(" - **:green[Company] :** (object) company name;")
  st.markdown(" - **:green[Words_Headline] :** (int64) word count of headline of customer post;")
  st.markdown(" - **:green[Words_Comment] :** (int64) word count of customer comment;")


  # import data engineered big 
  star1 = pd.read_csv("stars_simple_clean.csv")
  
  # show big data frame with check box
  if st.checkbox("Show big dataframe for simple ML models:", key="big") :
    st.dataframe(star1.head(10))
  if st.checkbox("Show shape of big dataframe", key="big2") :
    st.write(star1.shape)
  if st.checkbox("Show distribution of classes in big dataframe", key="big3") :
    st.write(star1['Stars'].value_counts(normalize=True))

  # handling of NaN and outliers
  st.markdown("In the distribution of the main variables, there were some data points with extreme values, which were investigated with KDE and box plots, looking into distribution and interquartile range. The most extreme values were capped.")

  # ML problem descripiton
  st.subheader("ML problem description")
  st.markdown("The rating prediction based on customer feedback was a **classification problem, multiclass with 5 target classes and imbalanced** as shown before, i.e. class 1 and 5 are the most common customer ratings with significantly less ratings for the middle ranks.")
  st.markdown("As performance metric, **accuracy** and **F1-score** were selected: Accuracy describes how the model performs over all classes as it calculates the ratio between the number of correct predictions to the total number of predictions. The focus is here on penalizing the FPs and FNs, but also not discarding F1-score for comparing different models.")
  
  # ML models
  st.subheader("ML models")
  st.markdown("**Support vector machine (SVM)** from scikit-learn ensemble was chosen first because we have limited features, and we were looking for a simple but effective model which would also support multi-class classification.")
  st.markdown("As second model **RandomForestClassifier** also from scikit-learn ensemble was applied, which is a meta estimator that fits several decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.")
  st.markdown("Applying train_test_split from sklearn.model_selection we specified a test set of 20%.")

  # first results
  st.subheader("First results")
  st.markdown("The first try showed poor results: the **1- and 5-star ratings were predicted best**. With hyperparameter tuning we saw a small improvement on the middle ratings while dominant classes performed a little less. RandomForestClassifier did better while we encountered an overfitting issue with the SVM classifier models. However, **overall performance was not much over 50% for the dominant ratings 1 and 5** which is very close to random reliability. In addition, run times were very long and handling of the code impractical, due to the large size of the data set.")
  st.image("Ratings_simple_results.png", caption="Poor performance metric for multi-class prediction of star ratings for big dataset")

  # improve performance
  st.subheader("Performance improvement with sentiment analysis")
  st.markdown("To improve performance, the following changes were implemented:")
  st.markdown(" - The size of the dataset was decreased to **:green[one supplier only]:**, following everyday use cases.")
  st.markdown(" - As star ratings were almost binary distributed, we simplified the modelling to a **:green[binary classification]:** problem targeting the Stars_geq4_TF variable.")
  st.markdown(" - Because we expect the star rating of the customer to be closely tied to the sentiment of the customer’s comment, we apply a **:green[sentiment analysis]:** to the explanatory variable Comment.")

  # Improved data set
  st.markdown("The improved and selected dataset was checked for E.ON customer feedback, also **applying natural logarithm to the word count features** before capping the extreme values.")

  # show columns list
  st.markdown(" - **:green[Comment] :** (object) customer comment, input for sentiment analysis;")
  st.markdown(" - **:green[log_Words_Headline] :** (int64) natural logarithm of word count of headline of customer post;")
  st.markdown(" - **:green[log_Words_Comment] :** (int64) natural logarithm of word count of customer comment;")
  st.markdown(" - **:green[Stars_geq4_TF] :** (bool) checks for customer rating > 3 stars;")

  # import improved data set
  eon1 = pd.read_csv("eon_clean.csv")
  
  # show data frame with check box
  if st.checkbox("Show E.ON dataframe for improved performance:", key="imp") :
    st.dataframe(eon1.head(10))
  if st.checkbox("Show shape of E.ON dataframe", key="imp2") :
    st.write(eon1.shape)
  if st.checkbox("Show distribution of classes", key="imp3") :
    st.write(eon1['Stars_geq4_TF'].value_counts(normalize=True))
  
  st.markdown("After transformation, the distribution of classes was **still imbalanced**, with nearly 70% of the customer comments deriving from 4 or 5-star ratings.")
  
  st.markdown("The limited dataset for one supplier only was **easier to handle**, reaching an **accuracy around 80%**. Even when adding the information from word count of headline to the comment feature, we could see **only small improvement for the non-dominant class**.")
  st.image("Ratings_eon_results.png", caption="Improved performance metric for binary class prediction and limited data set (E.ON)")
  
  st.markdown("While we saw **overfitting** for both models, the results were overall much **improved with sentiment analysis**. **Both classes were predicted over 75 % correctly (F1-score)**, for the dominant class even up to 90%, with **accuracy reaching up to 88% overall**.")
  st.image("Ratings_improved_results.png", caption="Best performance metric after sentiment analysis")
  
  ## keyword analysis
  st.subheader("Keyword Analysis by Sentiment", divider='green')
  st.markdown("Leveraging the results from sentiment analysis, the goal was to extract meaningful content, which could support customer satisfaction reportings and deliver input for improvement initiatives.")
  st.markdown("Focus was always on **one supplier only**, investigating **key words in the headlines, comments, and supplier responses**. Looking at E.ON data, this gave us a manageable dataset of 5,000 lines.")
  st.markdown("The process started always with creating a string containing the concatenation of all entries in the text column of the data set. Applying a function to streamline the process, the following steps were performed:")
  st.markdown(" - Converting the text to **lowercase characters only** with the casefold method")
  st.markdown(" - **Cleaning** a text of special characters, numbers etc. by regex and character mapping")
  st.markdown(" - **Tokenizing** the content of the string by the TweetTokenizer from nltk.tokeninze library")
  st.markdown(" - **Filtering** the word token list **with the stop_words** file specified for each supplier and German language setting")
  st.markdown("Separating the comments by star rating allowed to compare keywords for both positive and negative customer feedback. We started with visually pleasant **word clouds**, where large font size indicates high word count:")
  st.image("wordcloud_eon_headlines.png", caption="Positive and negative feedback for E.ON (feature Headline)")
  st.markdown("For deeper analysis, the function was expanded to create a data frame listing every word with its number of occurrences, sorted by word count in descending order.")
  st.markdown("Customers look for **easy** (einfach, einfacher, übersichtlich) and **fast** (schnell, schnelle) **transactions in change of supplier** (wechsel, zählerstand, service, eingabe).")
  # data sentiment analysis
  df_headline = pd.read_csv("eon_df_headline.csv")
  df_pos_headline = pd.read_csv("eon_df_pos_headline.csv")
  df_neg_headline = pd.read_csv("eon_df_neg_headline.csv")

  # donut chart
  n = 15 # number of key words
  pal = list(sns.color_palette(palette='Spectral', n_colors=n).as_hex())

  import plotly.express as px
  fig = px.pie(df_headline[0:n], values='count', names='words',
               color_discrete_sequence=pal)

  fig.update_traces(textposition='outside', textinfo='percent+label', 
                    hole=.6, hoverinfo="label+percent+name")

  fig.update_layout(width = 1200, height = 900,
                    title="E.ON: Keywords in headlines from customer feedback",
                    font=dict(
                        family="Arial",
                        size=18, 
                        color="black"
                    )
                  )
  st.plotly_chart(fig, use_container_width=True)

  st.markdown("Compared to classic word clouds the content is **more quantified and systematic**. This is especially helpful when comparing positive and negative comments between suppliers.")
  st.markdown("Customers praise the **easy** (einfach, einfacher, übersichtlich, immer) and **fast** (schnell, schnelle) transactions like **change of supplier** (anbieterwechsel, abwicklung, ablauf) and **contact with customer service** (kundenservice, service, kommunikation, bearbeitung, portal, app). Adjectives like **good** (gut, gute, guter, top), **friendly** (freundlich) and **to be recommended** (empfehlenswert) describe very **positive feelings**.")
  #color_dict
  #create function to get a color dictionary
  def get_colordict(palette, number, start):
      """
      This function creates a color dictionary to support visualizations for listed items.
      
      Parameters:
      -----------
      palette: name of matplotlib predefined color palette
      number: number of color fields needed
      start: number where to start in the palette

      Returns:
      --------
      a color dictionary
      """
      pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
      color_d = dict(enumerate(pal, start=start))
      return color_d

  # bar chart
  # Plot positive key words

  # index list for slicing
  index_list = [[i[0],i[-1]+1] for i in np.array_split(range(100), 5)]

  # color palette definition
  n = df_pos_headline['count'].max() ## add df!!
  color_dict = get_colordict('summer', n, 1)

   
  #plot
  import matplotlib.pyplot as plt
  import numpy as np
  fig, axs = plt.subplots(1, 2, figsize=(10,20), facecolor='white', squeeze=False)
  for col, idx in zip(range(0,2), index_list):
      df = df_pos_headline[idx[0]:idx[-1]]         ## add df!!
      label = [w + ': ' + str(n) for w,n in zip(df['words'],df['count'])]
      color_l = [color_dict.get(i) for i in df['count']]
      x = list(df['count'])
      y = list(range(0,20))
      
      sns.barplot(x = x, y = y, data=df, alpha=0.9, orient = 'h',
                  ax = axs[0][col], palette = color_l)
      axs[0][col].set_xlim(0,n+0.5)                     #set X axis range max
      axs[0][col].set_yticklabels(label, fontsize=24)
      axs[0][col].set(xticklabels=[])
      axs[0][col].tick_params(bottom=False)
      axs[0][col].spines['bottom'].set_color('white')
      axs[0][col].spines['right'].set_color('white')
      axs[0][col].spines['top'].set_color('white')
      axs[0][col].spines['left'].set_color('white')
  st.pyplot(fig, use_container_width=True)

  st.markdown("Looking on the **complaints** side, there have been issues with **longer waiting times** (wochen, monat, monate), concerning **contract** (Vertrag, Grundversorgung), **change of supplier** (gekündigt, wechsel), **invoice** (abrechnung, endabrechnung, jahresabrechnung) and with the **email communication** (mails, mail, email).")
  # Plot negative key words
  #create index list for slicing
  index_list = [[i[0],i[-1]+1] for i in np.array_split(range(100), 5)]

  # variable definition
  n = df_neg_headline['count'].max()  #dataframe  # enter df!!
  color_dict = get_colordict('Reds', n, 1) #color dicitionary from matplotlib

  #plot
  fig, axs = plt.subplots(1, 2, figsize=(10,20), facecolor='white', squeeze=False)
  for col, idx in zip(range(0,2), index_list):
      df = df_neg_headline[idx[0]:idx[-1]]  # enter df!!
      label = [w + ': ' + str(n) for w,n in zip(df['words'],df['count'])]
      color_l = [color_dict.get(i) for i in df['count']]
      x = list(df['count'])
      y = list(range(0,20))
      
      sns.barplot(x = x, y = y, data=df, alpha=0.9, orient = 'h',
                  ax = axs[0][col], palette = color_l)
      axs[0][col].set_xlim(0,n+0.5)                     #set X axis range max
      axs[0][col].set_yticklabels(label, fontsize=24)
      axs[0][col].set(xticklabels=[])
      axs[0][col].tick_params(bottom=False)
      axs[0][col].spines['bottom'].set_color('white')
      axs[0][col].spines['right'].set_color('white')
      axs[0][col].spines['top'].set_color('white')
      axs[0][col].spines['left'].set_color('white')
  st.pyplot(fig, use_container_width=True)
            
  st.markdown("Let us now have a look at the energy supplier side: What about company responses?")

# page 5: Modeling - Predicting company responses - Matthias
if page == pages[5] : 
  st.header("Modeling: Predicting Company Responses", divider='green')
  st.subheader("Dataset selection")
  st.markdown("The goal of the following ML problem is to predict the length of company answers to customer posts, respectively the full answers, if possible. Hence, we restrict the data set to rows where a comment and an answer exists. As the answer policy is company specific, we choose the companies E.ON Energy and Octopus Energy Germany for further investigation, creating a data set for each using the Company variable. These are the companies with the most entries, 4965 and 6001, respectively. It turns out that these companies require rather contrary modeling approaches.")
  
  # import data engineered big 
  ml_answers = pd.read_csv("df_ml.csv")[['Company', 'Headline','Comment', 'Answer', 'Stars_min_max_scaled','log_Words_Headline', 'log_Words_Comment', 'log_Words_Answer', 'Stars_geq4_TF']].dropna(subset='Answer')
  df_ml_eon=ml_answers[ml_answers['Company']== 'E.ON Energie Deutschland GmbH']
  df_ml_oct=ml_answers[ml_answers['Company']== 'Octopus Energy Germany']
  
  # show big data frame with check box
  if st.checkbox("Show dataframe for Octopus Energy", key="big") :
      st.dataframe(df_ml_oct.head(10))
  if st.checkbox("Show shape of dataframe for Octopus Energy", key="big1") :
      st.write(df_ml_oct.shape)     
  if st.checkbox("Show dataframe for E.ON Energy", key="big2") :
      st.dataframe(df_ml_eon.head(10))
  if st.checkbox("Show shape of dataframe for E.ON Energy", key="big3") :
      st.write(df_ml_eon.shape)  
  # ML problem description
  st.subheader('ML problem description')
  st.markdown(":green[E.ON Energy]")
  st.markdown('The E.ON data set contains **only two unique company answers**, up to modifications like spaces and captions. This makes it possible to predict the full Answer variable, i.e. there is no need to simplify the target by considering log_Words_Answer.')
  st.markdown('The prediction of answers is a **binary classification problem**, with **accuracy** as the main **performance metric** to maximize True Positives (TPs) and True Negatives (TNs).')
  st.markdown(":green[Octopus Energy]")
  st.markdown('Answers of Octopus Energy Germany are **highly personalized**. Company answers reference user names directly. In the 5- and 4-star regime there seem to be standard answers (up to user names), but below, answers are personalized to a very high degree. To simplify the modelling, we choose the target variable log_Words_Answer. ')
  st.markdown('The prediction of log_Words_Answer is a **regression problem**, with **root mean squared error (RMSE)** as the main **performance metric**. The sensitivity of RMSE to outliers is not a problem as outliers are tamed by the natural logarithm. The square root accounts for the errors being in the same order of magnitude as the data. ')
  # ML models and first results
  st.subheader('ML models and first results')
  st.markdown(":green[E.ON Energy]")
  st.markdown('First, we replace the two possible answers in the Answer variable by 0 and 1. The now binary target Answer is closely tied to the star rating: The variable Stars_geq4_TF is Pearson correlated to Answer by 0.9957. In fact, there are only 9 cases out of 4965 where Stars_geq4_TF and Answer do not match. These cases occur when people confuse the star rating (5 is the best) with the German grading system (1 is the best), leading to comments which are contrary to the ratings. E.ON Energy gave the correct answers to the sentiment of the comment, not to the star rating. This means they either use a strong sentiment analysis model on the comments for automatized answers, or a human assigns the two standard answers manually. ')
  st.markdown('We train a **logistic regression model** on the numeric columns log_Words_Comment, log_Words_Headline, Stars_geq4_TF. The test set is 20% of the total population. The model with default hyper parameters suppresses the first two numeric columns, i.e. it reduces to a copy of Stars_geq4_TF. It is exactly the 9 cases discussed that it cannot predict correctly, neither on the training, nor on the test set. We reach an astonishing accuracy on the total data set of 1 - 9/4965= 99.82%.')
  st.markdown(":green[Octopus Energy]")
  st.markdown('We predict log_Words_Answer on the numerical variables log_Words_Comment, log_Words_Headline, Stars_min_max_scaled. A **standard scaler** is applied on the two logarithmic varables. We will check the performance of several models (default hyperparameters) on a test set of test size 20%. This includes a custom model defined as follows: On the training set, compute the averages of log_Words_Answer grouped by Stars_min_max_scaled. On the test set, the predictions are defined as the computed averages (learned from the test set) rise to Stars_min_max_scaled. The results are collected in the following table. ')
  st.image('results_regression_oct.PNG')
  # sentiment analysis
  st.subheader('Sentiment analysis')
  st.markdown(":green[E.ON Energy]")
  st.markdown('Finally, we perform sentiment analysis on the Comment variable. Using regex, we replace all non-letters by spaces, consecutively removing words of length 2 or less. Each comment is converted to lowercase. We filter for german stop words and replace the special german characters ä, ö, ü, ß, by ae, oe, ue, ss. We convert the column to numerical columns with **CountVectorizer** from sklearn.feature_extraction.text. These numerical columns derived from Comment are used to train a **GradientBoostingClassifier** with respect to a test set size of 20%. On the test set we obtain an accuracy of (188+643)/(51+111+188+643)= 85%.')
  st.markdown(":green[Octopus Energy]")
  st.markdown('Following the same procedure on the Comment column as for E.ON Energy, but with a GradientBoostingRegressor, we get a RMSE of 0.5211.')
  # Model deployment
  st.subheader('Model deployment: Answering of E.ON Energy')
  st.markdown('The answering model of E.ON Energy is deployed in this section for free use to play around. In the respective text box type in a customer comment and press enter. The supposed company answer (translated to english) appears in the last textbox. We also show the intermediary processing steps standartization (remove non-letters, stopwords filtering, lowercase, replace special characters) and vectorization (CountVectorizer). ') 
  st.markdown('The model is trained on German comments, that is, the input should be in German language. An example of a positive comment (English with German translation):')
  st.markdown("- *English:*  Very good service with little cost. I'm really happy!")
  st.markdown('- *German:*  Sehr guter Service und günstig. Ich bin sehr zufrieden!')
  st.markdown('An example of a negative comment:')
  st.markdown("- *English:*  Very bad service. I called 10 times!")
  st.markdown('- *German:*  Sehr schlechter Service! Habe 10 mal angerufen!')
  # interactive model deployment
  import pickle
  import re
  import nltk
  nltk.download('stopwords')
  from nltk.corpus import stopwords

  #vectorizer = pickle.load(open("vectorizer_EON", 'rb'))
  #clf = pickle.load(open("GradientBoostingClassifier_EON", 'rb'))

  from joblib import Parallel, delayed 
  import joblib   
  vectorizer = joblib.load('vectorizer_EON_joblib.pkl') 
  clf = joblib.load('GradientBoostingClassifier_EON_joblib.pkl') 
  

  comment = st.text_input('Customer comment:', 'Sehr schlechter Service! Habe 10 mal angerufen!')

  reply_to_negative_comment_ger = 'Lieber Trustpilot Nutzer, vielen Dank für deine offenen Worte. Es tut uns sehr leid, dass du dich im Moment über uns ärgerst. Danke, dass du dir die Zeit für eine Bewertung hier auf Trustpilot genommen hast. Du ahnst es wahrscheinlich schon: Weil wir hier auf einer externen Bewertungsplattform sind, liest du eine automatisierte Antwort. Trotzdem hilfst du uns mit deiner Nachricht sehr unseren \
      Service und unsere Produkte weiter zu verbessern. Solltest du offene Fragen haben, beantwortet unser Kundenservice diese gern. Dazu ein kleiner Hinweis: Aktuell erreichen uns sehr viele Kundenanfragen, so dass es zu längeren Wartezeiten auf all unseren Servicekanälen kommt. Dafür bitten wir um Entschuldigung und bedanken uns für deine Geduld. Herzliche Grüße deine E.ON Energie Deutschland'
  reply_to_positive_comment_ger = 'Lieber Trustpilot Nutzer, wir freuen uns sehr, dass du zufrieden mit uns bist! Herzlichen Dank für dein positives Feedback und dafür, dass du dir die Zeit für eine Bewertung hier auf Trustpilot genommen hast. Damit hilfst du uns, unseren Service und unsere Produkte weiter zu verbessern. Viele Grüße deine E.ON Energie Deutschland'

  reply_to_negative_comment = "Dear Trustpilot user, thank you so much for your honest opinion. We're really sorry that you are not happy with us at the moment. Thanks that you took the time to leave a rating on Trustpilot. You can probably already guess: Since we are on an external rating platform, you're reading an automatized answer. Still, you're helping us a lot to improve our service and our products. If you have any open questions our customer service will be happy to help. A little note: At the moment we receive a lot of customer queries such that there could be longer waiting times on all our service lines. We're really sorry and ask for your patience. Sincerely, your E.ON Energy Germany. "

  reply_to_positive_comment = "Dear Trustpilot user, we're really happy that you are content with us! Thank you so much for your positive feedback and that you took the time to leave a rating at Trustpilot. You're helping us to improve our service and our products. Sincerely, your E.ON Energy Germany."

  stop_words = stopwords.words('german')
  special_char_map = {ord('ä'):'ae', ord('ü'):'ue', ord('ö'):'oe', ord('ß'):'ss'}

  def standardize_comment(comment):
    x = re.sub('[^a-zA-ZäüöÄÜÖß]+', ' ', comment).lower()
    x = re.sub('\s[a-zäöü]{1}\s', ' ', x)
    x = re.sub('\s[a-zäöü]{2}\s', ' ', x)
    x = ' '.join([word for word in x.split() if word not in (stop_words)]).translate(special_char_map)
    return x

  def vectorize(comment):
    standardized_comment= standardize_comment(comment)
    vector = np.asarray(vectorizer.transform([standardized_comment]).todense())
    return vector

  def predict_answer(comment):
      if clf.predict(vectorize(comment)):
        return reply_to_positive_comment
      else: 
        return reply_to_negative_comment

  st.text_area('Standartization:',standardize_comment(comment))
  
  #st.write(vectorize(comment))

  st.text_area('Modeled answer of E.ON Energy:',predict_answer(comment), height=100)
  # st.write(predict_answer(comment))
    


# page 6: Outlook - Practical Application - Stefanie
if page == pages[6] : 
  st.header("Outlook", divider='green')
  st.subheader("Practical Application")
  st.markdown("Monitoring customer feedback and regularly assessing customer satisfaction are key objectives in **customer service management**. With the analysis and predictive modeling in this scraped data set of Trustpilot ratings and comments for Energy Suppliers in Germany, we have simulated **two typical use cases**:")
  st.image("CustomerServiceMgmt_usecases.png", caption="Uses Cases from Customer Service Management")

  st.markdown("The next steps for **Customer Reply** could be to collect the previous answers as a **library to design new and individualized answers**, which could be prompted to a large language model, together with expected length, customer name, focus words and company values like eco-friendly marketing messages.")
  st.markdown("Looking at **Benchmark**: In case of specific complaints accumulating, **warning systems** could be triggered to rectify what was crooked as soon as possible. Distribution of information and quantified **basis for decision making** can be achieved more easily with pre-set **dashboards** and visualizations in reports.")

  st.subheader("If we had more time...")
  st.markdown("For a follow-up project, a seamless **integration of a pre-trained Language Model** with an API could be pursued in order to automate the generation of company responses to customer inquiries.")
  st.markdown(" - Leveraging keyword analysis as prompts, the model could generate responses, drawing from a library of authentic company answers. ")
  st.markdown(" - Additional prompts such as answer length, customer name, focus words, and core company values could further refine the response generation process.")
  st.markdown(" - To ensure accuracy, a comparison sentiment analysis could be employed to evaluate both the generated and authentic responses.")
