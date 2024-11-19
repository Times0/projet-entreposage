import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob

# Scikit-learn models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


models = {
    'KNN': KNeighborsClassifier(n_neighbors=10),
    'CART': DecisionTreeClassifier(random_state=15),
    'Random Forest': RandomForestClassifier(random_state=15, n_estimators=100),
    'XGBoost': XGBClassifier(random_state=15),
    'CatBoost': CatBoostClassifier(random_state=15, verbose=False),
    'MLP': MLPClassifier(random_state=15, max_iter=1000),
    "SVM": SVC(random_state=15, kernel="rbf"),
    "Naive Bayes": GaussianNB()
}





@st.cache_data
def run_cross_validation(X, y, model_name, n_splits=5):
    """
    Perform cross-validation and return performance metrics
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=15)
    
    metrics = {metric: [] for metric in ['accuracy', 'precision', 'recall', 'f1']}
    
    model = models[model_name]

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(report['1']['precision'])
        metrics['recall'].append(report['1']['recall'])
        metrics['f1'].append(report['1']['f1-score'])
    
    # Calculate mean and standard deviation for each metric
    result = {metric: {
        'mean': np.mean(metrics[metric]),
        'std': np.std(metrics[metric])
    } for metric in metrics}
    
    return result, X

def main():
    st.title('Machine Learning Model Comparison Dashboard')
    
    # Load the dataset (replace this with your actual data loading method)
    @st.cache_data
    def load_data():
        df1 = pd.read_csv("dataset/Physical dataset/phy_att_1.csv", sep="\t", encoding="utf-16")
        df2 = pd.read_csv("dataset/Physical dataset/phy_att_2.csv", sep="\t", encoding="utf-16")
        df3 = pd.read_csv("dataset/Physical dataset/phy_att_3.csv", sep="\t", encoding="utf-16")

        df1.drop("Label_n",inplace=True,axis=1)
        df2.drop("Lable_n",inplace=True,axis=1)
        df3.drop("Label_n",inplace=True,axis=1)

        # merge all datasets vertically
        df = pd.concat([df1, df2, df3], axis=0)
        df.drop("Time", inplace=True, axis=1)
        df["Label_n"] = df["Label"].apply(lambda x: 1 if x != "normal" else 0)

        df = df.sample(frac=1).reset_index(drop=True)

        return df
    
    df = load_data()
        
    # Prepare data
    X = df.drop(["Label", "Label_n"], axis=1)
    y = df["Label_n"]
                
    # Sidebar for model selection and visualization options
    st.sidebar.header('Dashboard Controls')
        
    # Model selection
    selected_model = st.sidebar.selectbox(
        'Select a Model', 
        list(models.keys())
        )
        
    # Run cross-validation for the selected model and cache results
    @st.cache_data
    def get_model_results(X, y, model_name):
            return run_cross_validation(X, y, model_name)
        
    # Get results for the selected model
    model_results, _ = get_model_results(X, y, selected_model)
        
    # Display selected model results
    st.header(f'Performance Metrics for {selected_model}')
    col1, col2 = st.columns(2)
    metrics_list = list(model_results.items())
    
    for i, (metric, value) in enumerate(metrics_list):
        with col1 if i < 2 else col2:
            st.metric(label=metric.capitalize(),
                     value=f"{value['mean']:.4f}",
                     delta=f"± {value['std']:.4f}")
    
    

if __name__ == '__main__':
    main()