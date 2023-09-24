import streamlit as st
import numpy as np
import pandas as pd
import sklearn

st.title('Risk Performance Dashboard')
st.markdown("Predict Risk Performance")

tab1, tab2, tab3 = st.tabs(["Data Overview", "Global Performance", "Local Performance"])

dataset = pd.read_csv("heloc_dataset_v1.csv")
dataset['RiskPerformance'] = dataset['RiskPerformance'].map({'Bad': 0, 'Good': 1})
data_encoded = pd.get_dummies(dataset, columns=['MaxDelq2PublicRecLast12M', 'MaxDelqEver'])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_encoded[['ExternalRiskEstimate', 'MSinceOldestTradeOpen']] = scaler.fit_transform(data_encoded[['ExternalRiskEstimate', 'MSinceOldestTradeOpen']])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder


import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sns
import shap
import lime
import lime.lime_tabular

# Split data into training and testing sets
X = data_encoded.drop(columns=['RiskPerformance'])
y = data_encoded['RiskPerformance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3959656)

rmm = RandomForestClassifier()
rmm.fit(X_train, y_train)

y_rmm = rmm.predict(X_test)

accuracy = accuracy_score(y_test, y_rmm)
precision = precision_score(y_test, y_rmm)
recall = recall_score(y_test, y_rmm)
f1 = f1_score(y_test, y_rmm)

# Calculate feature importances
importances = rmm.feature_importances_
feature_names = X_train.columns

sorted_indices = (-importances).argsort()[:10] 

with tab1:
    st.subheader("Total Statistics")
    
    # Create a two-column layout
    col1, col2 = st.columns(2)

    # Total Satisfactory Trades
    with col1:
        total_sat = dataset['NumSatisfactoryTrades'].sum()
        st.write("Total Satisfactory Trades:", total_sat)

    # Total Total Trades
    with col2:
        total_trades = dataset['NumTotalTrades'].sum()
        st.write("Total Total Trades:", total_trades)

    # Total Trades in Last 12 Months
    with col1:
        total_trades_12 = dataset['NumTradesOpeninLast12M'].sum()
        st.write("Total Trades in Last 12 Months:", total_trades_12)

    # Total Inquiries in Last 6 Months
    with col2:
        total_inq_6 = dataset['NumInqLast6M'].sum()
        st.write("Total Inquiries in Last 6 Months:", total_inq_6)

    # Create a two-column layout for the following plots
    col1, col2 = st.columns(2)

    with col1:
    # Check if 'RiskPerformance' column exists in the dataset
        if 'RiskPerformance' in dataset.columns:
            # Count the occurrences of 0 and 1 in 'RiskPerformance'
            risk_counts = dataset['RiskPerformance'].value_counts()

            # Create a Streamlit app
            st.subheader('Bar Chart for Risk Performance')
            
            # Create a bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(risk_counts.index, risk_counts.values)
            ax.set_xlabel('Risk Performance')
            ax.set_ylabel('Count')
            ax.set_xticks(risk_counts.index)
            ax.set_xticklabels(['0', '1'])

            # Display the plot in Streamlit
            st.pyplot(fig)
        else:
            st.write("The 'RiskPerformance' column does not exist in the dataset.")
    
    # Boxplot of External Risk Estimate by Risk Performance
    with col2:
        st.subheader('External Risk Estimate by Risk Performance')
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='RiskPerformance', y='ExternalRiskEstimate', data=dataset)
        plt.xlabel('Risk Performance')
        plt.ylabel('External Risk Estimate')
        st.pyplot(fig2)

    # Distribution of External Risk Estimate
    with col1:
        st.subheader('Distribution of External Risk Estimate')
        plt.figure(figsize=(8, 5))
        sns.histplot(dataset['ExternalRiskEstimate'], bins=20, kde=True)
        plt.xlabel('External Risk Estimate')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # Box Plot of Net Fraction Revolving Burden by Risk Performance
    with col2:
        st.subheader('Net Fraction Revolving Burden by Risk Performance')
        # Create a box plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='RiskPerformance', y='NetFractionRevolvingBurden', data=dataset)
        ax.set_xlabel('Risk Performance')
        ax.set_ylabel('Net Fraction Revolving Burden')
        # Display the plot in Streamlit
        st.pyplot(fig)

with tab2:
    # Split the layout into two columns
    col1, col2 = st.columns(2)

    # Display confusion matrix in the first column
    with col1:
        st.subheader("Confusion Matrix")
        conf_mat_fig = plt.figure(figsize=(4, 4))
        ax1 = conf_mat_fig.add_subplot(111)
        skplt.metrics.plot_confusion_matrix(y_test, y_rmm, ax=ax1, normalize=True)
        st.pyplot(conf_mat_fig, use_container_width=True)

    # Display feature importance plot in the second column
    with col2:
        st.subheader("Feature Importance Plot")
        
        # Create a Matplotlib figure and axis with appropriate size
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh([feature_names[i] for i in sorted_indices], importances[sorted_indices])
        st.pyplot(fig, use_container_width=True)

    # Rest of your Streamlit code for KPIs can remain as is
    with st.expander("Key Performance Indicators (KPIs)"):
        st.write(f'Accuracy: {accuracy:.2f}')
        st.write(f'Precision: {precision:.2f}')
        st.write(f'Recall: {recall:.2f}')
        st.write(f'F1-Score: {f1:.2f}')

with tab3:
    sliders = []
    col1, col2 = st.columns(2)
    with col1:
        for ingredient in X:
            ing_slider = st.slider(label=ingredient, min_value=float(data_encoded[ingredient].min()), max_value=float(data_encoded[ingredient].max()))
            sliders.append(ing_slider)

    with col2:
        col1, col2 = st.columns(2, gap="medium")
        
        prediction = rmm.predict([sliders])
        with col1:
            st.markdown("### Model Prediction : <strong style='color:tomato;'>{}</strong>".format(y[prediction[0]]), unsafe_allow_html=True)

        probs = rmm.predict_proba([sliders])
        probability = probs[0][prediction[0]]

        with col2:
            st.metric(label="Model Confidence", value="{:.2f} %".format(probability*100), delta="{:.2f} %".format((probability-0.5)*100))

        #shap_explainer = shap.TreeExplainer(rmm)
        #instance_to_explain = X_test.iloc[0]
        #shap_values = shap_explainer.shap_values(instance_to_explain)
        #shap.summary_plot(shap_values, instance_to_explain, feature_names=X.columns.tolist())
