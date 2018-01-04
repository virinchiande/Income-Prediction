
# coding: utf-8

# In[737]:


import pandas as pd

data = pd.read_csv("C:\\Users\\virin\\Desktop\\edx\\data\\final\\public\\train_values.csv")
data1= pd.read_csv("C:\Users\\virin\\Desktop\\edx\\data\\final\\public\\train_labels.csv")


# In[738]:


data.head(5)


# In[739]:


data1.head(5)


# In[740]:


final_data = pd.merge(data, data1, on='row_id')


# In[741]:


final_data.head(5)


# In[742]:


final_data.info()


# In[743]:


final_data.describe()


# In[744]:


corr = final_data.corr()


# In[745]:


corr["income"].sort_values(ascending = False)


# In[746]:


print(final_data["admissions__sat_scores_average_overall"].isnull().sum())


# In[747]:


count_1 = pd.value_counts(final_data["admissions__act_scores_25th_percentile_math"])


# In[748]:


print(count_1)


# In[749]:


import matplotlib.pyplot as plt
count_1.plot(kind = "bar")
plt.show()


# In[750]:


from sklearn.preprocessing import Imputer
import numpy as np

imputer = Imputer(strategy = "median")
x =final_data.iloc[:, np.r_[229:298]]


final_data= final_data.fillna((x.median()), inplace=True)
final_data["admissions__act_scores_25th_percentile_math"].head(5)


# In[751]:


corr.loc["income", "cost__tuition_in_state"]


# In[752]:


import statsmodels.api as sm

X = final_data["school__degrees_awarded_predominant_recoded"]
y = final_data["income"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()





# In[753]:


some_data = final_data["income"]
input_data = final_data["school__degrees_awarded_predominant_recoded"]


# In[754]:


pred = model.predict(input_data)
print(model.predict(input_data))
print(some_data)
from sklearn.metrics import mean_squared_error
lin_mse = mean_squared_error(pred, some_data)
rmse = np.sqrt(lin_mse)
rmse


# In[793]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

some_data = (final_data["income"])
print(type(some_data))
input_data = final_data[["school__degrees_awarded_predominant_recoded","student__demographics_first_generation",
                         "student__share_firstgeneration_parents_highschool","school__faculty_salary"]]

x = input_data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
input_data = pd.DataFrame(x_scaled)

tree_reg = RandomForestRegressor(max_depth = 30, random_state=2)
tree_reg.fit(input_data,some_data)
tree_pred = tree_reg.predict(input_data)
print(pd.DataFrame(tree_pred))
print(some_data)
dec_mse = mean_squared_error(tree_pred, some_data)
rmse = np.sqrt(dec_mse)
rmse 


# In[794]:


test_data = pd.read_csv("C:\\Users\\virin\\Desktop\\edx\\data\\final\\public\\test_values.csv")

x =test_data.iloc[:, np.r_[229:298]]
test_data= test_data.fillna((x.median()), inplace=True)


# In[795]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,input_data,some_data,scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[796]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("SD:",scores.std())


# In[797]:


display_scores(rmse_scores)


# In[756]:


degree_count = pd.value_counts(final_data["school__degrees_awarded_highest"])
print(degree_count)


# In[757]:


def degree_to_numeric(x):
    if x=="Graduate degree":
        return 4
    if x=="Bachelor's degree":
        return 3
    if x=="Associate degree":
        return 2
    if x=="Certificate degree":
        return 1
    if x=="Non-degree-granting":
        return 0


# In[758]:


final_data['highest_degree_num'] = final_data['school__degrees_awarded_highest'].apply(degree_to_numeric)
final_data['highest_degree_num']


# In[759]:


degree_count1 = pd.value_counts(final_data["school__degrees_awarded_predominant"])
print(degree_count1)


# In[760]:


def degree_to_numeric(x):
    if x=="Entirely graduate-degree granting":
        return 4
    if x=="Predominantly bachelor's-degree granting":
        return 3
    if x=="Predominantly associate's-degree granting":
        return 2
    if x=="Predominantly certificate-degree granting":
        return 1
    if x=="Not classified":
        return 0


# In[761]:


final_data['highest_degree_num1'] = final_data['school__degrees_awarded_predominant'].apply(degree_to_numeric)
final_data['highest_degree_num1']


# In[762]:



degree_count1 = pd.value_counts(final_data["school__main_campus"])
print(degree_count1)


# In[763]:


def campus_to_numeric(x):
    if x=="Main campus":
        return 1
    if x=="Not main campus":
        return 0


# In[764]:


final_data['school__main_campus'] = final_data['school__main_campus'].apply(campus_to_numeric)
final_data['school__main_campus']



# In[765]:


def degree_to_numeric_t(x):
    if x=="Entirely graduate-degree granting":
        return 4
    if x=="Predominantly bachelor's-degree granting":
        return 3
    if x=="Predominantly associate's-degree granting":
        return 2
    if x=="Predominantly certificate-degree granting":
        return 1
    if x=="Not classified":
        return 0


# In[766]:


test_data['highest_degree_num'] = test_data['school__degrees_awarded_predominant'].apply(degree_to_numeric_t)
test_data['highest_degree_num']


# In[767]:


def campus_to_numeric(x):
    if x=="Main campus":
        return 1
    if x=="Not main campus":
        return 0


# In[768]:


test_data['school__main_campus'] = test_data['school__main_campus'].apply(campus_to_numeric)
test_data['school__main_campus']


# In[798]:


test_input_data = test_data[["school__degrees_awarded_predominant_recoded","student__demographics_first_generation",
                         "student__share_firstgeneration_parents_highschool","school__faculty_salary"]]


# In[799]:


tree_pred = pd.DataFrame(tree_reg.predict(test_input_data), columns = ["income"])
tree_pred.head(5)


# In[786]:


test_data_id = pd.DataFrame(test_data["row_id"])
test_data_id.head(5)


# In[787]:


output_data=pd.concat([test_data_id, tree_pred], axis = 1)


# In[788]:


output_data.to_csv('output_data2.csv', index = False)

