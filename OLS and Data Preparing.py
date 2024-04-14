#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

csv_file_path = '/Users/chantalstefan/Documents/DCU/Final Year/Thesis/Research Survey_April 8, 2024_12.23.csv'

df = pd.read_csv(csv_file_path)

print(df.head())


# In[2]:


#Exploratory Data Analysis
df.describe()


# In[3]:


#dropping all insignificant rows 

df_dropped_corrected = df.iloc[:, 17:]

df_dropped_corrected.head()


# In[4]:


#Searching for missing values
df_cleaned = df_dropped_corrected.dropna()
df_cleaned = df_cleaned.drop('Q11_2', axis=1)

missing_values = df_cleaned.isnull().sum()
missing_values


# In[5]:


#cronbach alpha = measure of internal consistency,used as an accpeted estimate of the reliability of a psychometric test
#Reliability and explainability of the dimension via Cronbach's alpha demonstrated in the paper. 

#Calculating Cronbachs Alpha for Q6(Empowernment) to ensure internal consistency of the dimension
def calculate_cronbach_alpha(df):
    df = df.dropna()
    n_items = len(df.columns)
    variance_sum = df.var(axis=0, ddof=1).sum()
    total_variance = df.sum(axis=1).var(ddof=1)
    return (n_items / (n_items - 1)) * (1 - variance_sum / total_variance)

dimension_q6_columns = ['Q6_1', 'Q6_2', 'Q6_3','Q6_4']
df_cleaned[dimension_q6_columns] = df_cleaned[dimension_q6_columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['dimension_q6_mean'] = df_cleaned[dimension_q6_columns].mean(axis=1)

cronbach_alpha_q6 = calculate_cronbach_alpha(df_cleaned[dimension_q6_columns])

dimension_q6_mean_head = df_cleaned['dimension_q6_mean'].head()
dimension_q6_mean_head, cronbach_alpha_q6


# In[6]:


#calculating PCA for Reduction of Dimensionality of Q6 adn therefore calculating factor weight 

df_cleaned_no_na_q6 = df_cleaned.dropna(subset=dimension_q6_columns)

scaler = StandardScaler()
dimension_q6_scaled = scaler.fit_transform(df_cleaned_no_na_q6[dimension_q6_columns])

pca = PCA(n_components=1)
pca.fit(dimension_q6_scaled)

weights = pca.components_[0]

weights_normalized = weights / weights.sum()

df_cleaned_no_na_q6['factor_q6_weighted'] = df_cleaned_no_na_q6[dimension_q6_columns].dot(weights_normalized)

print(df_cleaned_no_na_q6[['dimension_q6_mean', 'factor_q6_weighted']])


# In[9]:


#Conducting Factor Anaylsis to calculate factor scores for Q6
#(quantitative measure of how much an individual exhibits dimensions captured by that factor )

q6_columns = ['Q6_1', 'Q6_2', 'Q6_3', 'Q6_4']

df[q6_columns] = df[q6_columns].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=q6_columns)

fa = FactorAnalyzer(n_factors=1, rotation=None)

fa.fit(df[q6_columns])

factor_loadings = fa.loadings_

df['weighted_factor_score'] = df[q6_columns].apply(lambda row: np.dot(row, factor_loadings[:, 0]), axis=1)

def scale_to_range(x, range_min, range_max):
    x_min = x.min()
    x_max = x.max()
    return range_min + (x - x_min) * (range_max - range_min) / (x_max - x_min)

df['rescaled_weighted_score'] = scale_to_range(df['weighted_factor_score'], 1, 5)

assert df['rescaled_weighted_score'].min() >= 1 and df['rescaled_weighted_score'].max() <= 5

print(df[['rescaled_weighted_score'] + q6_columns])


# In[10]:


#Calculating Cronbachs Alpha for Q7(Standing Back) to ensure internal consistency of the dimension
dimension_q7_columns = ['Q7_1', 'Q7_2']
df_cleaned[dimension_q7_columns] = df_cleaned[dimension_q7_columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['dimension_q7_mean'] = df_cleaned[dimension_q7_columns].mean(axis=1)

cronbach_alpha_q7 = calculate_cronbach_alpha(df_cleaned[dimension_q7_columns])

dimension_q7_mean_head = df_cleaned['dimension_q7_mean'].head()
dimension_q7_mean_head, cronbach_alpha_q7


# In[11]:


#calculating PCA for Reduction of Dimensionality of Q7 and therefore calculating factor weight 
dimension_q7_columns = ['Q7_1', 'Q7_2']

df_cleaned_no_na_q7 = df_cleaned.dropna(subset=dimension_q7_columns)

scaler = StandardScaler()
dimension_q7_scaled = scaler.fit_transform(df_cleaned_no_na_q7[dimension_q7_columns])


pca = PCA(n_components=1)
pca.fit(dimension_q7_scaled)

weights_q7 = pca.components_[0]

weights_normalized_q7 = weights_q7 / weights_q7.sum()

df_cleaned_no_na_q7['factor_q7_weighted'] = df_cleaned_no_na_q7[dimension_q7_columns].dot(weights_normalized_q7)

df_cleaned_no_na_q7['dimension_q7_mean'] = df_cleaned_no_na_q7[dimension_q7_columns].mean(axis=1)

print(df_cleaned_no_na_q7[['dimension_q7_mean', 'factor_q7_weighted']])


# In[12]:


#Conducting Factor Anaylsis to calculate factor scores for Q7
q7_columns = ['Q7_1', 'Q7_2']

df[q7_columns] = df[q7_columns].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=q7_columns)

fa = FactorAnalyzer(n_factors=1, rotation=None)

fa.fit(df[q7_columns])

factor_scores = fa.transform(df[q7_columns])

df['Q7_factor_score'] = factor_scores.flatten()

print(df[['Q7_factor_score'] + q7_columns])

min_factor_score = df['Q7_factor_score'].min()
max_factor_score = df['Q7_factor_score'].max()

df['rescaled_combined_q7'] = 5 - (df['Q7_factor_score'] - min_factor_score) * (4) / (max_factor_score - min_factor_score)

print(df[['rescaled_combined_q7'] + q7_columns])


# In[13]:


#neglecting Q8(Accountablity) as it just contains of one cloumn
#Calculating Cronbachs Alpha for Q9(Forgiveness) to ensure internal consistency of the dimension
dimension_q9_columns = ['Q9_1', 'Q9_2']
df_cleaned[dimension_q9_columns] = df_cleaned[dimension_q9_columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['dimension_q9_mean'] = df_cleaned[dimension_q9_columns].mean(axis=1)

cronbach_alpha_q9 = calculate_cronbach_alpha(df_cleaned[dimension_q9_columns])

dimension_q9_mean_head = df_cleaned['dimension_q9_mean'].head()

dimension_q9_mean_head, cronbach_alpha_q9


# In[14]:


#calculating PCA for Reduction of Dimensionality of Q9 adn therefore calculating factor weight 
dimension_q9_columns = ['Q9_1', 'Q9_2']

df_cleaned_no_na_q9 = df_cleaned.dropna(subset=dimension_q9_columns)

scaler = StandardScaler()
dimension_q9_scaled = scaler.fit_transform(df_cleaned_no_na_q9[dimension_q9_columns])

pca = PCA(n_components=1)
pca.fit(dimension_q9_scaled)

weights_q9 = pca.components_[0]

weights_normalized_q9 = weights_q9 / weights_q9.sum()

df_cleaned_no_na_q9['factor_q9_weighted'] = df_cleaned_no_na_q9[dimension_q9_columns].dot(weights_normalized_q9)

df_cleaned_no_na_q9['dimension_q9_mean'] = df_cleaned_no_na_q9[dimension_q9_columns].mean(axis=1)

print(df_cleaned_no_na_q9[['dimension_q9_mean', 'factor_q9_weighted']])


# In[15]:


#Conducting Factor Anaylsis to calculate factor scores for Q9
q9_columns = ['Q9_1', 'Q9_2']

df[q9_columns] = df[q9_columns].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=q9_columns)

fa = FactorAnalyzer(n_factors=1, rotation=None)

fa.fit(df[q9_columns])

factor_scores = fa.transform(df[q9_columns])

df['Q9_factor_score'] = factor_scores.flatten()

print(df[['Q9_factor_score'] + q9_columns])

min_factor_score = df['Q9_factor_score'].min()
max_factor_score = df['Q9_factor_score'].max()

df['rescaled_combined_q9'] = 1 + (df['Q9_factor_score'] - min_factor_score) * (4) / (max_factor_score - min_factor_score)

print(df[['rescaled_combined_q9'] + q9_columns])


# In[16]:


#neglecting Q10 - 12(Courage, Authenticity, Humility) as it just contains of one cloumn
#Calculating Cronbachs Alpha for Q13(Stewartship) to ensure internal consistency of the dimension
dimension_q13_columns = ['Q13_1', 'Q13_2']
df_cleaned[dimension_q13_columns] = df_cleaned[dimension_q13_columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['dimension_q13_mean'] = df_cleaned[dimension_q13_columns].mean(axis=1)

cronbach_alpha_q13 = calculate_cronbach_alpha(df_cleaned[dimension_q13_columns])

dimension_q13_mean_head = df_cleaned['dimension_q13_mean'].head()

dimension_q13_mean_head, cronbach_alpha_q13
#Neglection of Q13 as this dimension is lacking internal consistency


# In[17]:


#Calculating Cronbachs Alpha for Q14(Change Sucess) to ensure internal consistency of the dimension
dimension_q14_columns = [f'Q14_{i}' for i in range(1, 13)]
df_cleaned[dimension_q14_columns] = df_cleaned[dimension_q14_columns].apply(pd.to_numeric, errors='coerce')
df_cleaned['dimension_q14_mean'] = df_cleaned[dimension_q14_columns].mean(axis=1)

cronbach_alpha_q14 = calculate_cronbach_alpha(df_cleaned[dimension_q14_columns])

dimension_q14_mean_head = df_cleaned['dimension_q14_mean'].head()

dimension_q14_mean_head, cronbach_alpha_q14


# In[18]:


#calculating PCA for Reduction of Dimensionality of Q14 and therefore calculating factor weight 

dimension_q14_columns = [f'Q14_{i}' for i in range(1, 13)]

df_cleaned_no_na_q14 = df_cleaned.dropna(subset=dimension_q14_columns)

scaler = StandardScaler()
dimension_q14_scaled = scaler.fit_transform(df_cleaned_no_na_q14[dimension_q14_columns])

pca = PCA(n_components=1)
pca.fit(dimension_q14_scaled)

weights_q14 = pca.components_[0]

weights_normalized_q14 = weights_q14 / weights_q14.sum()

df_cleaned_no_na_q14['factor_q14_weighted'] = df_cleaned_no_na_q14[dimension_q14_columns].dot(weights_normalized_q14)

df_cleaned_no_na_q14['dimension_q14_mean'] = df_cleaned_no_na_q14[dimension_q14_columns].mean(axis=1)

print(df_cleaned_no_na_q14[['dimension_q14_mean', 'factor_q14_weighted']])


# In[19]:


#Conducting Factor Anaylsis to calculate factor scores for Q14
q14_columns = [f'Q14_{i}' for i in range(1, 13)]

df[q14_columns] = df[q14_columns].apply(pd.to_numeric, errors='coerce')

df = df.dropna(subset=q14_columns)

fa = FactorAnalyzer(n_factors=1, rotation=None)

fa.fit(df[q14_columns])

factor_scores = fa.transform(df[q14_columns])

df['Q14_factor_score'] = factor_scores.flatten()

print(df[['Q14_factor_score'] + q14_columns])

min_factor_score = df['Q14_factor_score'].min()
max_factor_score = df['Q14_factor_score'].max()

df['rescaled_combined_q14'] = 1 + (df['Q14_factor_score'] - min_factor_score) * (4) / (max_factor_score - min_factor_score)

print(df[['rescaled_combined_q14'] + q14_columns])


# In[20]:


# merging of weighted and non-weighted dimensions
non_weighted_columns = [
    'Q8_1',
    'Q10_1',
    'Q11_1',
    'Q12_1'
]

weighted_columns = [
    'factor_q6_weighted',   
    'factor_q7_weighted', 
    'factor_q9_weighted', 
    'factor_q14_weighted'
]

new_df = df_cleaned[non_weighted_columns].copy()

new_df['factor_q6_weighted'] = df_cleaned_no_na_q6['factor_q6_weighted']
new_df['factor_q7_weighted'] = df_cleaned_no_na_q7['factor_q7_weighted']
new_df['factor_q9_weighted'] = df_cleaned_no_na_q9['factor_q9_weighted']
new_df['factor_q14_weighted'] = df_cleaned_no_na_q14['factor_q14_weighted']

print(new_df.head())


# In[21]:


#rename and reorder columns for easy readability adn consistency
new_df_renamed = new_df.rename(columns={
    'factor_q6_weighted': 'q6',
    'factor_q7_weighted': 'q7',
    'Q8_1': 'q8',  
    'factor_q9_weighted': 'q9',
    'Q10_1': 'q10',
    'Q11_1': 'q11', 
    'Q12_1': 'q12', 
    'factor_q14_weighted': 'q14'
})

column_order = ['q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q14']

new_df_ordered = new_df_renamed[column_order]

print(new_df_ordered.head())


# In[22]:


#Converting to numeric values
new_df_ordered['q8'] = pd.to_numeric(new_df_ordered['q8'], errors='coerce')
new_df_ordered['q10'] = pd.to_numeric(new_df_ordered['q10'], errors='coerce')
new_df_ordered['q11'] = pd.to_numeric(new_df_ordered['q11'], errors='coerce')
new_df_ordered['q12'] = pd.to_numeric(new_df_ordered['q12'], errors='coerce')

df = new_df_ordered.dropna(subset=['q8', 'q10', 'q11', 'q12'])


# In[23]:


# regression and fit analysis - Q6 to Q13 on Q14
import pandas as pd
import statsmodels.api as sm

X = new_df_ordered[[f'q{i}' for i in range(6, 14) if i != 13]] 

y = new_df_ordered['q14']

combined = pd.concat([X, y], axis=1).dropna()
X = combined[[f'q{i}' for i in range(6, 14) if i != 13]]
y = combined['q14']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print('Regression analysis for q6 to q12 (excluding q13) influencing q14:')
print(model.summary())
print("\n")

print(f'Accuracy score (R-squared) for q6 to q12 (excluding q13) on q14: {model.rsquared:.4f}')
print("--------------------------------------------------\n")


# In[24]:


#
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

#Plots residuals against fitted values to check for non-linear patterns and homoscedasticity, with a lowess line to help identify trends
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax[0, 0], line_kws={'color': 'red', 'lw': 1})

ax[0, 0].set_title('Residuals vs Fitted')
ax[0, 0].set_xlabel('Fitted values')
ax[0, 0].set_ylabel('Residuals')

#Displays a quantile-quantile plot of the standardized residuals against the theoretical quantiles from a normal distribution to assess normality.
sm.qqplot(model.resid, line='s', ax=ax[0, 1])
ax[0, 1].set_title('Normal Q-Q')
ax[0, 1].set_xlabel('Theoretical Quantiles')
ax[0, 1].set_ylabel('Standardized Residuals')

#Plots the square root of the absolute standardized residuals against the fitted values to check for constant variance across the range of fitted values
ax[1, 0].scatter(model.fittedvalues, model.resid ** 0.5, alpha=0.5)
sns.regplot(x=model.fittedvalues, y=model.resid ** 0.5, scatter=False, ci=False, lowess=True, ax=ax[1, 0], line_kws={'color': 'red', 'lw': 1})
ax[1, 0].set_title('Scale-Location')
ax[1, 0].set_xlabel('Fitted values')
ax[1, 0].set_ylabel('Sqrt(Standardized Residuals)')

#Shows the influence of each data point, highlighting points with high leverage and large residuals that could be potential outliers or have substantial influence on the model fit
sm.graphics.influence_plot(model, ax=ax[1, 1], criterion="cooks")
ax[1, 1].set_title('Influence Plot')
ax[1, 1].set_xlabel('Leverage')
ax[1, 1].set_ylabel('Standardized Residuals')

plt.tight_layout()
plt.show()


# In[ ]:




