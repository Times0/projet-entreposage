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
import time
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob
import plotly.graph_objects as go
import plotly.express as px

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
    execution_times = []
    
    model = models[model_name]

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        end_time = time.time()
        execution_times.append(end_time - start_time)
        
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
    
    # Add execution time metrics
    result['execution_time'] = {
        'mean': np.mean(execution_times),
        'std': np.std(execution_times)
    }
    
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

    # Create sections in sidebar
    section = st.sidebar.selectbox(
        "Choose a Section",
        ["Exploratory Analysis", "Physical Data Analysis", "Network Data Analysis"],
        key="section_selector",
        help="Select a section to view different analyses"
    )

    if section == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        st.write("This section will contain exploratory analysis of the data")
        
        # show histogram of "Label" column using plotly
        fig = px.histogram(df, x="Label", title="Label Distribution")
        st.plotly_chart(fig)

        # show histogram of "Label_n" column using plotly with custom labels
        fig = px.histogram(df, x="Label_n", title="Attack vs Normal Distribution",
                          labels={'Label_n': 'Type', 'count': 'Count'},
                          category_orders={'Label_n': [0, 1]})
        fig.update_xaxes(ticktext=['Normal (0)', 'Attack (1)'], tickvals=[0, 1])
        st.plotly_chart(fig)

    elif section == "Physical Data Analysis":
        # Main content area
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)
            
        # Run cross-validation for the selected model and cache results
        @st.cache_data
        def get_model_results(X, y, model_name):
                return run_cross_validation(X, y, model_name)
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Current Model Performance", "Literature Comparison"])

        with tab1:
            st.header('Model Performance Analysis')
            st.markdown('The models are trying to predict if a given line is normal or an attack, it does not try to predict the type of attack')
            
            # Model selection with description
            st.subheader('Model Selection')
            selected_model = st.selectbox(
                'Choose a Machine Learning Model',
                list(models.keys()),
                help='Select a model to view its performance metrics'
            )
            
            st.header(f'Performance Metrics for {selected_model}')
            st.markdown('The metrics are the average of the 5 folds')
            
            # Get results for the selected model
            model_results, X_data = get_model_results(X, y, selected_model)
            
            # Display performance metrics in a grid
            col1, col2, col3, col4 = st.columns(4)
            metrics_list = [(k,v) for k,v in model_results.items() if k != 'execution_time']
            
            metrics_columns = [col1, col2, col3, col4]
            for (metric, value), col in zip(metrics_list, metrics_columns):
                with col:
                    st.metric(
                        label=metric.capitalize(),
                        value=f"{value['mean']:.4f}",
                        delta=f"± {value['std']:.4f}",
                        help=f'Mean and standard deviation for {metric}'
                    )
            
            # Display execution time with improved styling
            st.subheader("⏱️ Execution Time")
            st.metric(
                label="Average Processing Time (seconds)",
                value=f"{model_results['execution_time']['mean']:.4f}",
                delta=f"± {model_results['execution_time']['std']:.4f}",
                help='Average time taken to train and evaluate the model'
            )

            # Add confusion matrix visualization
            st.subheader("Confusion Matrix")
            
            @st.cache_data
            def get_confusion_matrix(X, y, model_name):
                model = models[model_name]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                return cm
                
            cm = get_confusion_matrix(X, y, selected_model)
            
            # Create a heatmap using plotly
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Attack'],
                y=['Normal', 'Attack'],
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                width=600,
                height=400
            )
            
            st.plotly_chart(fig)

        with tab2:
            st.header("📚 Reference Performance Metrics")
            st.markdown("""
            This section shows performance metrics from published research papers,
            demonstrating how different algorithms perform on physical and network datasets.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Literature Results")
                # Research metrics with improved formatting
                research_metrics = {
                    "Algorithm": ["KNN", "Random Forest", "SVM", "Naive Bayes"],
                    "Accuracy": [0.98, 0.99, 0.93, 0.93],
                    "Recall": [0.95, 0.98, 0.92, 0.92], 
                    "Precision": [0.95, 0.95, 0.64, 0.66],
                    "F1 Score": [0.95, 0.97, 0.75, 0.77],
                }
                
                # Convert to DataFrame for better display
                df_research = pd.DataFrame(research_metrics).set_index("Algorithm")
                
                # Style the dataframe
                styled_df = df_research.style\
                    .format({"Accuracy": "{}", "Recall": "{}", 
                            "Precision": "{}", "F1 Score": "{}"})\
                    .background_gradient(cmap='Blues', subset=['Accuracy', 'Recall', 'Precision', 'F1 Score'])\
                    .set_properties(**{'text-align': 'center'})
                    
                st.dataframe(styled_df, use_container_width=True)

            with col2:
                st.subheader("Our Results")
                # Get results for all models
                our_metrics = {
                    "Algorithm": [],
                    "Accuracy": [],
                    "Recall": [],
                    "Precision": [],
                    "F1 Score": []
                }
                
                for model_name in models.keys():
                    results, _ = get_model_results(X, y, model_name)
                    our_metrics["Algorithm"].append(model_name)
                    our_metrics["Accuracy"].append(f"{results['accuracy']['mean']:.2f}")
                    our_metrics["Recall"].append(f"{results['recall']['mean']:.2f}")
                    our_metrics["Precision"].append(f"{results['precision']['mean']:.2f}")
                    our_metrics["F1 Score"].append(f"{results['f1']['mean']:.2f}")
                
                df_our = pd.DataFrame(our_metrics).set_index("Algorithm")
                
                styled_df_our = df_our.style\
                    .format({"Accuracy": "{}", "Recall": "{}", 
                            "Precision": "{}", "F1 Score": "{}"})\
                    .background_gradient(cmap='Blues', subset=['Accuracy', 'Recall', 'Precision', 'F1 Score'])\
                    .set_properties(**{'text-align': 'center'})\
                    
                    
                st.dataframe(styled_df_our, use_container_width=True)

    else:  # Network Data Analysis
        st.header("Network Data Analysis")
        st.write("This section will contain network data analysis")
        # Add network data analysis content here

if __name__ == '__main__':
    main()