#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate TMDb Movies data set

# # Research Question
# 1. What are the most popular genres from year to year?
# 2. What types of properties are associated with high income films?
# 3. What are the most popular films from one year to the next?

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


# Data wrangling
# 
# load the data

# In[6]:


df = pd.read_csv('tmdb-movies.csv')


# In[8]:


df.head(5)


# In[ ]:





# In[9]:


df = pd.read_csv('tmdb-movies.csv')


# In[10]:


df.duplicated().sum()


# In[11]:


df.drop_duplicates(inplace = True)


# In[12]:


df.duplicated().sum()


# In[7]:


df.info()


# In[13]:


import datetime as dt # for convering to date time object
df['release_date'] = df['release_date'].apply(lambda x: dt.datetime.strptime(x, "%m/%d/%y"))


# In[14]:


df['release_date'].head()


# In[15]:


df['release_date'].tail()


# May be there is a problem while getting the value of year for some datetime objects. let's see the year which is converted incorectly
# 
# the range of year was  of 1960-2015(both inclusive) but 2066 is showing in above code cell.
# 
# let's check after conversion

# In[16]:


years = []
for _ in df['release_date']:
    years.append(_.year)
print(list(set(years)))


# No logical reason for this failure, but the years from 1969 to 1979 are historically known as the Python years. I believe that this may be the main reason why the years before 1969 were translated into 21st century years instead of the correct years (like 68 may be interpreted by Python as 2068 instead of 1968). just this is how i can explain this error.
# 
# Fixing the year values for 2060-2068 ( both included ) by moving back 100 years:

# In[17]:


d_100 = []
years = 100
days_per_year = 365.24


# In[18]:


for d in df['release_date']:
    if d.year >=2060:
        #shifting back 100 years 
        tdelta = dt.timedelta(days = (years * days_per_year))
        # corrected date
        d-=tdelta
        d_100.append(d)
    else:
        #as it is
        d_100.append(d)


# We do not move the years directly by -100 at once since the year parameter is not available in the dt.timedelta() method.
# 
# We are now replacing the original release_date column (which was of type string) with the list d_100 (which is the corrected release_date).

# In[19]:


df['release_date'] = d_100


# Checking

# In[20]:


d = []
for _ in df['release_date']:
    d.append(_.year)
print(list(set(d)))


# # Removing undesired columns and handling missing data
# Checking important column for our analysis

# In[18]:


df.head(3)


# In[21]:


len(df['id']), len(df['imdb_id']), df.shape[0]


# As the duplicates were removed in the first part of the analysis, the columns ['id', 'imdb_id'] are not relevant for this analysis. These values are single for each row, so it is better to use the row index instead of these columns.

# In[22]:


df.drop(['id', 'imdb_id'], axis = 1, inplace = True)


# Let's investigate if the columns ['overview', 'original_title', 'homepage', 'tagline', 'production_companies', 'keywords'] can be classed as categorical data or not.

# In[23]:


test_col = ['overview', 'original_title', 'homepage', 'tagline', 'production_companies', 'keywords'] # 6 columns
new_dict = {}
for column in df:
    if column in test_col:
        new_dict[column] = df[column].nunique()
print(new_dict)


# Levels of these 6 columns are very big. If levels for a certain column would have been around 10 then it is ok to consider that column as a categorical variable and faceting could be done based on that categorical column.
# 
# But here, levels are 10571, 2896, 7997, 8804, 10847, 7445. Number of rows(10865) are around this much level. Faceting will be inconvenient.
# 
# So, removing these columns, too, will not affect the result.

# In[24]:


df.drop(['overview', 'original_title', 'homepage', 'tagline', 'production_companies', 'keywords'], axis = 1, inplace = True)


# Checking and handling missing data.

# In[25]:


df.isnull().any().sum()


# 3 columns are missing values.
# 
# Checking the incomplete columns:

# In[26]:


df.isnull().any()


# These column 'cast', 'director' and 'genres' are missing values. Let's see number of missing values for each of these columns to decide whether it is ok to drop them from dataframe.
# 
# Creating a subset of df which has only incomplete rows:

# In[27]:


null_data = df[df.isnull().any(axis = 1)]
null_data.shape[0]


# In[28]:


round(100*null_data.shape[0]/df.shape[0], 2)


# There are only 134 incomplete rows in aggregate. That's just 1.2 per cent of df. Now, let's see incomplete rows count for each column.

# In[29]:


miss_count = {}
for column in null_data:
    miss_count[column] = sum(pd.isnull(null_data[column]))


# In[30]:


miss_count


# In[31]:


76+44+23, round(100*(76+44+23)/df.shape[0], 2) # missing counts for columns: cast, director and genres


# In[32]:


df.dropna(subset=['cast', 'director', 'genres'], inplace = True)


# In[33]:


miss_count = {}
for column in df:
    miss_count[column] = sum(pd.isnull(df[column]))


# In[34]:


miss_count


# # Exploratory Data Analysis
# Research Question 1: What are the most popular genres from year to year?
# I will first sort my data by release_date column for year to year analysis.
# 
# Sorting by date:

# In[35]:


df.sort_values(by = ['release_date'], inplace = True)
df.head(3)


# There are too many columns to deal with. We want to know which genres popular in this time period 1960-2015.
# For this part of EDA(question 1), I will remove columns that I am not interested in using.

# In[36]:


df.columns


# In[37]:


df_r1 = df.drop(['popularity', 'budget', 'revenue', 'cast', 'director', 'runtime', 'vote_count', 'vote_average', 'budget_adj', 'revenue_adj'], axis = 1)
df_r1.head(3)


# You might be thinking why I removed popularity and budget-revenue columns, specifically popularity.That's because they are properties of each movie, not each genre. We cannot infer the popularity of a genre by popularity of the movie in which it is there. That's because every movie is a mix of different type of genres and there is not a correct way to distribute the popularity of a movie among its constitute genres.
# 
# Since we are analysing genres trends from year to year, we can further remove release_date column particularly for this research question because we have release_year column for that.
# 
# Removing release_date column:

# In[38]:


df_r1.drop(['release_date'], axis = 1, inplace = True)
df_r1.head(3)


# We can see that Action and Drama are repeated above.
# This means each movie contains genres separated by pipe operator "|".
# Let's look at of genres column:

# In[39]:


df_r1['genres']


# Creating a list of unique genres from 1960-2015:

# In[40]:


def unique(col):
    """
    Takes input as a dataframe column and output a list containing
    unique values from that column.
    """
    
    # pandas series to list
    entries = list(col)
    
    # handling "|" separator and removing duplicates
    collect = [] #this will contain all the unique genres
    
    for entry in entries:
        for _ in entry.split("|"):
            if _ not in collect:
                collect.append(_)
    return(collect)


# In[41]:


unique(df_r1['genres'])


# In[42]:


len(unique(df_r1['genres'])), df_r1.shape[0]


# The o/p in above text cell shows that all the entries in df_r1['genres'] has 10731 entries which are different combinations of elements from unique(df_r1['genres']). Now let's calculate occurrence of each genre for corresponding year.
# 
# Creating a list of all the years:

# In[43]:


year = df_r1['release_year'].unique()


# In[44]:


year


# I will start by creating an empty dictionary new_df.

# In[45]:


new_df = {key:[] for key in unique(df_r1['genres'])}
new_df


# In[46]:


new_df['year'] = year
new_df


# In[47]:


for y in year: # to get genre count for each year
    
    #subsetting for corresponding year
    y_df_r1 = df_r1[df_r1['release_year'] == y]
    
    # converting pandas series to column
    genres_i1 = list(y_df_r1['genres'])  
    genres_f1 = [] # this will contain all the genres that we see for a given year(with repetition and "|" separator)
    
    for genre in genres_i1: # for splitting every entry in y_df[genres] with separator as "|" 
        for i in genre.split("|"):
            genres_f1.append(i)

    new_list = Counter(genres_f1) # occurrence of each genre in a year
    for genre in unique(df_r1['genres']): #this will create occurrence of each genre in a year
        if genre not in genres_f1:
            new_df[genre].append(0)
        else:
            new_df[genre].append(new_list[genre])


# converting this dictionary to data frame:

# In[48]:


new_df = pd.DataFrame(new_df, index = new_df['year'])
new_df.head()


# In[49]:


new_df.columns


# In[50]:


del(new_df['year'])


# In[51]:


new_df.columns


# In[52]:


new_df.shape
# new_df.head()


# In[53]:


new_df.head()


# # What this dataframe shows?
# 
# Let's see first row. If there are n number of movies released in 1960, then entries corresponding to 1960 shows how many times
# each genre appeared in 1960.
# Let's see this numerically:

# In[54]:


total_occ = new_df.loc[1960].sum()
total_occ

Total occurrences off all genres(including repetition) = 78
# In[56]:


action_occ = new_df.loc[1960,'Action']
action_occ


# Action appeared 8 times in 1960.
# 
# 

# In[57]:


action_part = 100*action_occ/total_occ
action_part


# Portion of occurrences showing Action: 10.256
# Similarly calculating proportions for all genres in this new_df and creating another data frame new_df1 for analyzing year by year genre trends.

# In[59]:


new_df1 = {key:[] for key in unique(df_r1['genres'])}
new_df1


# In[60]:


new_df1['year'] = year
new_df1


# In[61]:


for index,row in new_df.iterrows():
    for genre in list(new_df.columns):
        new_df1[genre].append((100*row[genre]/sum(row)))


# Converting to data frame:

# In[62]:


new_df1 = pd.DataFrame(new_df1)
new_df1.head(5)


# Moving new_df['year'] to first place:

# In[63]:


cols = new_df1.columns.tolist()
cols = cols[-1:] + cols[:-1]
new_df1 = new_df1[cols]
new_df1.columns


# In[64]:


new_df1.head()


# Now visualising trends for each genre. I will use the same function to get a list of unique genres:

# In[141]:


for genre in unique(df['genres']):
    plt.scatter(y = new_df1[genre], x = new_df1['year']);
#     plt.legend()

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':12}

plt.title("Year vs genre", fontdict = font1)
plt.xlabel("Year", fontdict = font2)
plt.ylabel("genre", fontdict = font2)


# This graph looks very messy. Let's see the mean value of occurrence for each genre:

# In[66]:


mean = {}
for _ in new_df1.columns[1:]: # coz firt row is year column
    mean[_] = np.mean(new_df1[_])
mean


# I am dropping all the genres except the top 5 ones.

# In[67]:


top_5 = []
for _ in new_df1.columns[1:]:
    top_5.append(np.mean(new_df1[_]))
    top_5.sort(reverse = True)
top_5[:5]


# These top 5 mean values are for Drama, Comedy, Thriller, Action, Romance. Now, visualising trends only for these genres:

# In[140]:


import seaborn as sns
g = ['Drama','Comedy','Thriller','Action','Romance']
for genre in g:
    plot = sns.regplot(y = new_df1[genre], x = new_df1['year'], lowess = True);
plot.set_ylabel("");
plot.axvline(x = 1974, color = 'black', alpha = 0.7);

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':12}

plt.title("Year vs Top 5 genre", fontdict = font1)
plt.xlabel("Year", fontdict = font2)
plt.ylabel("genre", fontdict = font2)


# # Question 2: What types of properties are associated with high income films?

# In[70]:


df.head()


# Taking inflation into account. Removing budget and revenue columns:

# In[71]:


df.drop(['revenue', 'budget'], axis = 1, inplace = True)


# In[73]:


df.head()


# In[129]:


plt.hist(x = df['revenue_adj']);
plt.xlabel('revenue');
font1 = {'family':'serif','color':'blue','size':20}
# font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Revenue histogram", fontdict = font1)


# This is skewed distribution. Let's see the summary of revenue_adj column:

# In[73]:


df['revenue_adj'].describe()


# Minimum value in revenue_adj is 0. That's why our distribution is skewed. Let's see how many such rows are there.

# In[74]:


# minimum value of revenue is 0.
df[df['revenue_adj'] == min(df['revenue_adj'])].shape[0], 100*(df[df['revenue_adj'] == min(df['revenue_adj'])].shape[0])/df.shape[0]


# 5888 movies have no revenue. It is more than 50% of df.This will skew our distribution for revenue_adj column.
# In order to correctly know the properties of high revenue movies, it is important to remove movies which have no revenue.

# In[75]:


df_r2 = df.query("revenue_adj != 0")


# ##### Definition of scatter diagram graphs
# The scatter diagram graphs pairs of numerical data, with one variable on each axis, to look for a relationship between them. If the variables are correlated, the points will fall along a line or curve.
# The better the correlation, the tighter the points will hug the line.

# In[131]:


plt.scatter(x = df_r2["budget_adj"], y = df_r2['revenue_adj'],alpha = 0.4);
# plt.xlabel("budget");
# plt.ylabel("revenue");
plt.axvline(x = 100000000, color = 'Green', alpha = 0.7);

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':12}

plt.title("Budget vs revenue", fontdict = font1)
plt.xlabel("budget", fontdict = font2)
plt.ylabel("revenue", fontdict = font2)


# from the scatterplot, it is clear that revenue is not strongly dependent on budget of movie. There are few movies which have high revenue. Budget of most of the movies is less than $100m.

# In[126]:


plt.scatter(x = df_r2['vote_average'],y = df_r2['revenue_adj'], alpha = 0.1);
# plt.xlabel("avg_vote");
# plt.ylabel("revenue");
plt.axvline(x = np.mean(df['vote_average']), color = 'Green');
# plt.title("Relation between revenue and voting average")

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Revenue vs vote average", fontdict = font1)
plt.xlabel("vote_average", fontdict = font2)
plt.ylabel("revenue_adj", fontdict = font2)


# This graph shows that high revenue movies have voting above average.

# In[121]:


# y = df_r2['revenue_adj']
# plt.plot (x, y, alpha = 0.8)
# # plt.xlabel("vote_average")
# # plt.ylabel("revenue_adj")
# plt.axvline(x = np.mean(df['vote_average']), color = 'Green')
# # plt.title("Relation between revenue and voting average")

# font1 = {'family':'serif','color':'blue','size':20}
# font2 = {'family':'serif','color':'darkred','size':15}

# plt.title("Relation between revenue and vote average", fontdict = font1)
# plt.xlabel("vote_average", fontdict = font2)
# plt.ylabel("revenue_adj", fontdict = font2)


# In[127]:


plt.scatter(df_r2["vote_count"], df_r2['revenue_adj'], alpha = 0.1);
# plt.xlabel("vote_count");
# plt.ylabel("revenue");

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Vote_count vs revenue", fontdict = font1)
plt.xlabel("vote_count", fontdict = font2)
plt.ylabel("revenue", fontdict = font2)


# There is a positive correlation between revenue and vote_count but not strong enough to make any conclusion.

# In[124]:


plt.scatter(x = df_r2['release_year'], y = df_r2['revenue_adj'], alpha = 0.3);
# plt.xlabel('year');
# plt.ylabel('revenue');
plt.axhline(y = np.mean(df_r2['revenue_adj']), color = 'Orange', alpha = 0.7)
plt.title('Revenue vs Year');

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Revenue vs Year", fontdict = font1)
plt.xlabel("Year", fontdict = font2)
plt.ylabel("Revenue", fontdict = font2)


# In[ ]:





# The increasing density of blue color shows that number of high revenue movies has increased from 1960 to 2015. The red line is for mean of revenue_adj column.
# 
# 

# # Question 3: What are the most popular films from one year to the next?

# In[89]:


df['popularity'].describe()


# In[91]:


plt.scatter(x = df['runtime'],y = df['popularity'], alpha = 0.3)
plt.axvline(x = df['runtime'].mean(), color = "blue");
plt.axvline(x = np.mean(df['runtime']) + 3*np.std(df['runtime']), color = 'blue')
plt.xlabel("runtime");
plt.ylabel("popularity");
plt.title('Popularity vs Runtime');


# In[92]:


np.std(df['runtime'])


# In[93]:


df['runtime'].mean()


# Average runtime of movies is around 102 minute. All the popular movies have runtime more than 102 minute. First vertical line is for np.mean(df['runtime']) and second vertical line is for np.mean(df['runtime']) + 3*np.std(df['runtime']).

# In[94]:


plt.scatter(x = df['budget_adj'], y = df['popularity'], alpha = 0.3);
plt.xlabel('budget');
plt.ylabel('popularity');
plt.title('Popularity vs Budget');
plt.axvline(x = 100000000, color = 'blue', alpha = 0.7);
plt.axvline(x = 200000000, color = 'blue', alpha = 0.7);


# Popular movies have budget between $100m and 200m.
# 
# 

# In[96]:


plt.scatter(x = df_r2['release_year'], y = df_r2['popularity'],alpha= 0.4);
plt.xlabel('year');
plt.ylabel('popularity');
plt.title('Popularity vs Release Year');
plt.axhline(y = np.mean(df['popularity']), color = 'black', alpha = 0.7);


# Popularity of movies has increased from 1960 to 2015, 2015 showing the highest popularity movies.
# 
# 

# In[98]:


plt.scatter(x = df['vote_count'], y = df['popularity'], alpha = 0.1);
plt.xlabel('vote_count');
plt.ylabel('popularity');
plt.axvline(x = 2000, color = 'black');
plt.title('Popularity vs Vote Count');


# bove graph shows that popular movies have high vote count, wheras all the less popular movies have vote count less than 2000. The vertical line shows this trend.

# In[128]:


plt.scatter(x = df['vote_average'], y = df['popularity'], alpha = 0.1);
plt.xlabel('vote_average');
plt.ylabel('popularity');
plt.axvline(x = np.mean(df['vote_average']), color = "black");
plt.title('Popularity vs Average Vote');


# Popular movies also have above average voting.

# In[101]:


plt.scatter(x = df['revenue_adj'], y = df['popularity'], alpha = 0.1);
plt.xlabel('revenue');
plt.ylabel('popularity');
plt.title('Popularity vs Revenue');


# # Conclusions
# 

# ## Research Question 1
# 

# The popular genres from 1960 to 2015 are drama, comedy, thriller, action and romance respectively. Drama and comedy films consistently occupy the top two places, but thriller and action films changed their position around 1974. Since 1974, the popularity of action films has declined. The fifth position is occupied by romantic films.

# ## Research Question 2
# I had to drop half of the entries in the revenue_adj column to see the properties of the high grossing films. There was no relationship between revenue and budget. Most high grossing films have above average votes. As the number of average votes for a film increases, so does its revenue. If we try to fit the curve from the revenue_adj column to the average_votes column, we will get an ascending parabola or exponential.
# 
# 

# ## Research Question 3
# 
# Most favourite films have a duration equal to or greater than the average in the duration column. The vertical lines show this trend. The budget of popular films is between 100 and 200 million dollars, but I have not indicated the exact budget values because it does not equal a statistic in the budget_adj column. The popularity and interest in films has increased from time to time. In the popularity vs. year graph, we can see that 2015 saw the most popular films of all time. The graph of Popularity vs the Average Vote shows that all the popular films have an above average vote.

# ## Limitation of the study
# 

# This analysis was done on student scope to help him to fullfil the requirement of non degree program from udacity and there is a high probabity that this is study will not be put into practice.
# 

# In[104]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:





# In[ ]:




