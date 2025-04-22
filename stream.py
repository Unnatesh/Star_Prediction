
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
st.set_page_config(
    page_title="🌌 Star Classifier",
    page_icon="✨",
    layout="wide")
st.sidebar.title("🪐 Navigation")
page = st.sidebar.radio("Go to", [
    "🌠 Intro & Dataset", 
    "🧹 Preprocessing", 
    "⚙️ Model Selection & 📉 Results", 
])
data = pd.read_csv(r"star_classification.csv")

if page == "🌠 Intro & Dataset":
    # App Title
    st.title("🌌 Predicting Stars, Galaxy and Quasar with ML")

    st.markdown("""
        <div style="text-align: center;">
            <img src="https://chandra.harvard.edu/photo/2019/gsn069/gsn069_525.gif" alt="star gif" width="400">
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## 🌠 About the Problem & Our Dataset

    ### So what exactly are stars, galaxies, and quasars?

    #### 🌀 GALAXY
    <p align="justify">
    A GALAXY is a gravitationally bound system of stars, stellar remnants, interstellar gas, dust, and dark matter. Galaxies are categorised according to their visual morphology as elliptical, spiral, or irregular. Many galaxies are thought to have supermassive black holes at their active centers.
    </p>

    <center>
        <img src="https://media.hswstatic.com/eyJidWNrZXQiOiJjb250ZW50Lmhzd3N0YXRpYy5jb20iLCJrZXkiOiJnaWZcL2dldHR5aW1hZ2VzLTU0MTM4NTg1Ni5qcGciLCJlZGl0cyI6eyJyZXNpemUiOnsid2lkdGgiOiIxMjAwIn19fQ==" width="70%">
    </center>

    #### ☀️ STAR
    <p align="justify">
    A STAR is a type of astronomical object consisting of a luminous spheroid of plasma held together by its own gravity. The nearest star to Earth is the Sun.
    </p>

    <center>
        <img src="https://cdn.eso.org/images/original/science-stars-formation-banner1.jpg" width="70%">
    </center>

    #### ✨ QUASAR
    <p align="justify">
    A QUASAR, also known as a quasi-stellar object, is an extremely luminous active galactic nucleus (AGN). The power radiated by quasars is enormous. A typical quasar is 27 trillion times brighter than our sun! If you were to place a quasar at the distance of Pluto, it would vaporise all of Earth’s oceans to steam in a fifth of a second.
    </p>

    <center>
        <img src="https://earthsky.org/upl/2020/06/quasar-black-hole-original.png" width="70%">
    </center>

    ---


    <p align="justify">
    The data consists of 10,000 observations of space taken by the SDSS. Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar. 30% of it is used in testing the model performance and 70% in training of the model.
    </p>
    """, unsafe_allow_html=True)


    # Load data

    st.subheader("🔭 Raw Data")
    st.write(data.head())

    column_info = {
        "Column Name": [
            "obj_ID", "alpha", "delta", "u", "g", "r", "i", "z", "run_ID", "rerun_ID",
            "cam_col", "field_ID", "spec_obj_ID", "class", "redshift", "plate", "MJD", "fiber_ID"
        ],
        "Description": [
            "Unique object identifier",
            "Right Ascension (RA) — celestial longitude",
            "Declination (DEC) — celestial latitude",
            "Ultraviolet filter magnitude",
            "Green filter magnitude",
            "Red filter magnitude",
            "Near-infrared filter magnitude",
            "Infrared filter magnitude",
            "Observation run ID",
            "Data processing version ID",
            "Camera column",
            "Field number",
            "Spectroscopic observation ID",
            "Target class (GALAXY, STAR, or QSO)",
            "Redshift value",
            "Spectroscopic plate number",
            "Modified Julian Date of observation",
            "Fiber number in spectrograph"
        ]
    }
    st.subheader("📑 Raw Data")
    st.write(data.head())
    st.subheader("📑 Column Info")
    st.dataframe(pd.DataFrame(column_info))
    st.subheader("🔍 Data Description")
    st.write(data.describe())
 
    # Get null counts per column
    null_counts = data.isnull().sum()

    # Filter only columns with at least 1 null value
    null_counts = null_counts[null_counts > 0]

    st.subheader("Null Values Summary")

    if null_counts.empty:
        st.success("No missing values found in the dataset! ✅")
    else:
        # Convert to DataFrame for prettier display
        null_df = pd.DataFrame({'Column': null_counts.index, 'Missing Values': null_counts.values})
        st.dataframe(null_df, use_container_width=True)
    # Define a custom palette for the classes
    palette = {
        'GALAXY': 'blue',
        'STAR': 'red',
        'QSO': 'yellow'
    }
    st.subheader("📈 Class Distribution")
    counts = data['class'].value_counts().reset_index()
    counts.columns = ['class', 'count']

    st.altair_chart(alt.Chart(counts).mark_bar().encode(
        x='class', y='count',
        color=alt.Color('class', scale=alt.Scale(domain=['GALAXY', 'STAR', 'QSO'],
                                                range=['blue', 'red', 'green']))
    ), use_container_width=True)
    sns.set(style="ticks", context="paper")
    plt.style.use('dark_background')
    st.subheader("📊 Boxplot of Redshift by Class")


    # Create the boxplot with customized dots for each class
    sns.boxplot(data=data, x="class", y="redshift", palette=palette)

    # To change the dot colors specifically, we can access the individual plot components
    for i, class_label in enumerate(data['class'].unique()):
        # Add a scatter plot over the box plot for better dot control
        subset = data[data['class'] == class_label]
        plt.scatter([i] * len(subset), subset['redshift'], color=palette[class_label], s=50, label=class_label)

    # Display the plot
    st.pyplot(plt)

elif page == "🧹 Preprocessing":
    data["class"] = [0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in data["class"]]
    st.subheader("🔄 Data Preprocessing")
    st.subheader("🔥Heat Map")
    f,ax = plt.subplots(figsize=(12,8))
    sns.heatmap(data.corr(), cmap="PuBu", annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
    st.pyplot(f)
    data = data.drop(['obj_ID','alpha','delta','run_ID','rerun_ID','cam_col','field_ID','fiber_ID','spec_obj_ID','MJD','plate'], axis=1)

    # Data Split and SMOTE
    from collections import Counter
    from imblearn.over_sampling import SMOTE
    X = data.drop(['class'], axis = 1)
    y_ = data.loc[:,'class'].values
    st.subheader("SMOTE: Synthetic Minority Oversampling Technique")
    sm = SMOTE(random_state=42)
    print('Original dataset shape %s' % Counter(y_))
    X, y = sm.fit_resample(X, y_)
    print('Resampled dataset shape %s' % Counter(y))
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.success("Applied SMOTE to handle class imbalance")
    # Visualizing class distribution before SMOTE
    st.subheader("Class Distribution Before SMOTE")
    fig, ax = plt.subplots()
    sns.countplot(x=y_, ax=ax, palette='Set1')
    ax.set_title('Class Distribution Before SMOTE')
    st.pyplot(fig)

    # Visualizing class distribution after SMOTE
    st.subheader("Class Distribution After SMOTE")
    fig, ax = plt.subplots()
    sns.countplot(x=y_train, ax=ax, palette='Set1')
    ax.set_title('Class Distribution After SMOTE')
    st.pyplot(fig)
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test



elif page == "⚙️ Model Selection & 📉 Results":
    if "X_train" in st.session_state:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
    # proceed with model selection and training
    else:
        st.warning("⚠️ Please run the '🧹 Preprocessing' step first.")
   # Model Training
    st.subheader("🤖 Model Training and Evaluation")
    model_choice = st.selectbox("Choose Model", [
        "Random Forest", 
        "SVM", 
        "KNN", 
        "Logistic Regression", 
        "XGBoost", 
        "Naive Bayes", 
        "Gradient Boosting"
    ])

    # Initialize the model based on choice
    if model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "SVM":
        model = SVC(probability=True)
    elif model_choice == "KNN":
        model = KNeighborsClassifier()
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_choice == "XGBoost":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_choice == "Naive Bayes":
        model = GaussianNB()
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier()

    # Fit and predict
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    score = model.score(X_test, y_test)
    score_ = np.mean(score)
    # Display accuracy as a progress bar
    st.progress(score_)
    st.success(f"Model Accuracy: {score_:.2f}")

    st.subheader("📈 Evaluation Metrics")
    y_pred = predicted

    # ROC Curve Visualization
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)


    # Plotting ROC Curve using Plotly
    roc_fig = px.area(
        x=fpr, y=tpr,
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        title=f'ROC Curve (AUC = {roc_auc:.2f})'
    )
    roc_fig.update_traces(fill='toself', line=dict(color='blue', width=2))
    st.plotly_chart(roc_fig)

    # Evaluation
    metrics_df = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_pred)],
        "Precision": [precision_score(y_test, y_pred, average='macro')],
        "Recall": [recall_score(y_test, y_pred, average='macro')],
        "F1-Score": [f1_score(y_test, y_pred, average='macro')]
    })
    st.dataframe(metrics_df)

    accuracy_percentage = score_ * 100
    st.subheader("Model Accuracy")
    progress_bar = st.progress(0)
    for percent in range(int(accuracy_percentage) + 1):
        progress_bar.progress(percent)

    st.text(f"Accuracy : {accuracy_percentage:.2f}%")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig_cm)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp

    st.subheader("Confusion Matrix Metrics")

    for i, class_label in enumerate(['GALAXY', 'STARS', 'QUASAR']):
        st.subheader(f"{class_label}:")
        st.write(f"{tp[i]} {class_label} were correctly predicted (True Positive (TP))")
        st.write(f"And {fp[i]} {class_label} were incorrectly predicted (False Positive (FP))")


st.markdown("---")
st.markdown("Made with 💫 by Unnatesh Hetampuria")
st.markdown("Inspired by the beauty of the universe and the mysteries of the cosmos.")
