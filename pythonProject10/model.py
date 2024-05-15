from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
df = pd.read_csv('loan_approval_dataset-2.csv')

cart_model = DecisionTreeClassifier()

joblib.dump(model, 'datahunters.joblib')