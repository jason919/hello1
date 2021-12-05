import pandas as pd

data = pd.read_csv('/Users/jason/Downloads/archive/Admission_Predict_Ver1.1.csv')
data.head()

continuous_features = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA']].values / 100
print(len(continuous_features))
# categorical_research_features = data[['Research']].values
