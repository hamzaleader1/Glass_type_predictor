# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix as cm, plot_roc_curve, plot_precision_recall_curve, ConfusionMatrixDisplay as CMD
from sklearn.metrics import precision_score, recall_score

# ML classifier Python modules
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR

# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data()

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
feature=["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"]

@st.cache()
def prediction(model,feature):
    pred=model.predict([feature])

    if pred[0]==1:
        return "building windows float processed"
    elif pred[0]==2:
        return "building windows non float processed"
    elif pred[0]==3:
        return "vehicle windows float processed"
    elif pred[0]==4:
        return "vehicle windows non float processed"
    elif pred[0]==5:
        return "containers"
    elif pred[0]==6:
        return "tableware"
    else:
        return "headlamps"

st.sidebar.title("Exploratory Data Analysis")
st.title("Glass Type Predictor")

if st.sidebar.checkbox("Show raw data"):
    st.dataframe(glass_df)

st.sidebar.subheader("Scatter Plot")
selected=st.sidebar.multiselect("Select the x-axis values for scatterplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

st.set_option('deprecation.showPyplotGlobalUse', False)

for i in selected:
    st.subheader(f"This is a scatter plot between Glass Type and {i}")
    plt.figure(figsize=(25,10))
    sns.scatterplot(x=glass_df[i],y=glass_df['GlassType'])
    st.pyplot()

st.sidebar.subheader("Visualisation Selector")

plot_selection=st.sidebar.multiselect("Select the plot or chart required",('Histogram', 'Box Plot', 'Count Plot', 'Pie Chart', 'Correlation Heatmap', 'Pair Plot'))

if 'Histogram' in plot_selection:
    st.sidebar.subheader("Histogram")
    selected2=st.sidebar.multiselect("Select the x-axis values for histogram",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

    for i in selected2:
        st.subheader(f"This is a Histogram for {i}")
        plt.figure(figsize=(25,10))
        plt.hist(glass_df[i],bins='sturges',edgecolor='red')
        st.pyplot()

if 'Box Plot' in plot_selection:
    st.sidebar.subheader("Boxplot")
    selected3=st.sidebar.multiselect("Select the x-axis values for boxplot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))

    for i in selected3:
        st.subheader(f"This is a boxplot for {i}")
        plt.figure(figsize=(25,10))
        sns.boxplot(x=glass_df[i])
        st.pyplot()

if 'Count Plot' in plot_selection:
    st.subheader("Count plot")
    plt.figure(figsize=(25,10))
    sns.countplot(x=glass_df['GlassType'])
    st.pyplot()

if "Pie Chart" in plot_selection:
    st.subheader("Piechart")
    plt.figure(figsize=(25,10))
    plt.pie(glass_df['GlassType'].value_counts(),labels=glass_df['GlassType'].value_counts().index,autopct="%1.2f%%")
    st.pyplot()

if "Correlation Heatmap" in plot_selection:
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(25,10))
    sns.heatmap(glass_df.corr(),annot=True)
    st.pyplot()

if "Pair Plot" in plot_selection:
    st.subheader("Pair plot")
    plt.figure(figsize=(25,10))
    sns.pairplot(glass_df)
    st.pyplot()

l=[]
for i in ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]:
    l.append(st.sidebar.slider(i,float(glass_df[i].min()),float(glass_df[i].max())))

st.sidebar.subheader("Choose Classifier")
classifier=st.sidebar.selectbox("Classifier",('Support Vector Machine', 'Random Forest Classifier', 'Logistic Regression'))

if classifier == "Support Vector Machine":
    st.sidebar.subheader("Model Hyperparameters")
    c=st.sidebar.number_input("C (Error Rate)",1,10,1)
    k=st.sidebar.radio("Kernel", ('linear', 'rbf','poly'))
    g=st.sidebar.number_input("Gamma",1,10,1)

    if st.sidebar.button("Classify"):
        model=SVC(kernel=k,gamma=g,C=c).fit(X_train,y_train)
        predicted=prediction(model,l)
        y_test_pred=model.predict(X_test)
        st.write(f"The glass predicted by the given features is {predicted}")
        st.write("The confusion matrix for the test set is:")
        CMD.from_predictions(y_test,y_test_pred)
        st.pyplot()
        st.write("The accuracy score is:")
        st.write(round(model.score(X_train,y_train),2))

if classifier == "Random Forest Classifier":
    st.sidebar.subheader("Model Hyperparameters")
    trees=st.sidebar.number_input("Number of trees in forest",1,100,1)
    depth=st.sidebar.number_input("Depth of trees",1,100,1)

    if st.sidebar.button("Classify"):
        model=RFC(n_estimators=trees,max_depth=depth,n_jobs=-1).fit(X_train,y_train)
        predicted=prediction(model,l)
        y_test_pred=model.predict(X_test)
        st.write(f"The glass predicted by the given features is {predicted}")
        st.write("The confusion matrix for the test set is:")
        CMD.from_predictions(y_test,y_test_pred)
        st.pyplot()
        st.write("The accuracy score is:")
        st.write(round(model.score(X_train,y_train),2))

if classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    c=st.sidebar.number_input("c value",1,100,1)
    max_it=st.sidebar.number_input("maximum iterations",10,1000,10)

    if st.sidebar.button("Classify"):
        model=LR(C=c,max_iter=max_it).fit(X_train,y_train)
        predicted=prediction(model,l)
        y_test_pred=model.predict(X_test)
        st.write(f"The glass predicted by the given features is {predicted}")
        st.write("The confusion matrix for the test set is:")
        CMD.from_predictions(y_test,y_test_pred)
        st.pyplot()
        st.write("The accuracy score is:")
        st.write(round(model.score(X_train,y_train),2))