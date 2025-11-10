# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Student Performance Predictor",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for attractive design
# st.markdown("""
#     <style>
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
#     .stApp {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#     }
#     .metric-card {
#         background: white;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 10px 0;
#     }
#     .title-text {
#         color: #1e3a8a;
#         font-size: 7rem;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#     .subtitle-text {
#         color: #475569;
#         font-size: 1.2rem;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     div[data-testid="stMetricValue"] {
#         font-size: 2rem;
#         color: #1e3a8a;
#     }
#     .stButton>button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 10px 30px;
#         border-radius: 25px;
#         font-weight: bold;
#         transition: all 0.3s;
#     }
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(0,0,0,0.2);
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title
# st.markdown('<h1 class="title-text">🎓 Students Performance Prediction System</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle-text">Advanced Machine Learning Analytics for Gexton and PITP Excellence</p>', unsafe_allow_html=True)

# # Sidebar
# with st.sidebar: 
#     st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
#     st.title("📊 Navigation")
#     page = st.radio("Select Page:", ["🏠 Dashboard", "📈 Model Training", "🔮 Prediction", "📊 Data Analysis"])
    
#     st.markdown("---")
#     st.info("*About:* This system uses Random Forest Classifier to predict student performance based on various academic and demographic factors.")

# # Dataset - Manually provided
# @st.cache_data
# def load_data():
#     data = {
#         'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
#                    'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
#         'Study_Hours': [7, 5, 8, 3, 6, 9, 4, 7, 5, 8, 6, 4, 9, 3, 7, 8, 5, 6, 4, 9],
#         'Attendance': [85, 70, 92, 55, 78, 95, 65, 88, 72, 90, 80, 68, 94, 60, 86, 91, 75, 82, 67, 96],
#         'Previous_Grade': [78, 62, 85, 45, 70, 92, 58, 80, 65, 87, 75, 60, 90, 50, 82, 88, 68, 76, 55, 93],
#         'Parent_Education': ['Graduate', 'High School', 'Postgraduate', 'High School', 'Graduate', 
#                            'Postgraduate', 'High School', 'Graduate', 'High School', 'Postgraduate',
#                            'Graduate', 'High School', 'Postgraduate', 'High School', 'Graduate',
#                            'Postgraduate', 'Graduate', 'Graduate', 'High School', 'Postgraduate'],
#         'Internet_Access': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes',
#                            'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes'],
#         'Performance': ['Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Fail', 'Pass', 'Fail', 'Pass',
#                        'Pass', 'Fail', 'Pass', 'Fail', 'Pass', 'Pass', 'Pass', 'Pass', 'Fail', 'Pass']
#     }
#     return pd.DataFrame(data)

# df = load_data()

# # Encode categorical variables
# @st.cache_data
# def encode_data(data):
#     le_gender = LabelEncoder()
#     le_parent = LabelEncoder()
#     le_internet = LabelEncoder()
#     le_performance = LabelEncoder()
    
#     data_encoded = data.copy()
#     data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
#     data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
#     data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
#     data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
    
#     return data_encoded, le_gender, le_parent, le_internet, le_performance

# df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)

# # Train model
# @st.cache_resource
# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     return model, accuracy, cm, X_test, y_test, y_pred

# X = df_encoded.drop('Performance', axis=1)
# y = df_encoded['Performance']
# model, accuracy, cm, X_test, y_test, y_pred = train_model(X, y)

# # Dashboard Page
# if page == "🏠 Dashboard":
#     st.header("📊 Performance Overview")
    
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Total Students", len(df), delta="Active")
#     with col2:
#         pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
#         st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
#     with col3:
#         st.metric("Model Accuracy", f"{accuracy*100:.1f}%", delta="High")
#     with col4:
#         avg_study = df['Study_Hours'].mean()
#         st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
    
#     st.markdown("---")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📊 Performance Distribution")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         performance_counts = df['Performance'].value_counts()
#         colors = ['#667eea', '#f87171']
#         ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', 
#                colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
#         ax.set_title('Student Performance Distribution', fontsize=14, weight='bold', pad=20)
#         st.pyplot(fig)
    
#     with col2:
#         st.subheader("📈 Study Hours vs Performance")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.boxplot(data=df, x='Performance', y='Study_Hours', palette=['#667eea', '#f87171'], ax=ax)
#         ax.set_title('Study Hours Impact on Performance', fontsize=14, weight='bold', pad=20)
#         ax.set_xlabel('Performance', fontsize=12, weight='bold')
#         ax.set_ylabel('Study Hours', fontsize=12, weight='bold')
#         st.pyplot(fig)
    
#     st.markdown("---")
    
#     st.subheader("📋 Student Data Overview")
#     st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

# # Model Training Page
# elif page == "📈 Model Training":
#     st.header("🤖 Machine Learning Model Training")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🎯 Model Performance Metrics")
#         st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
#         st.metric("Training Samples", len(X) - len(X_test))
#         st.metric("Testing Samples", len(X_test))
        
#         st.markdown("---")
#         st.subheader("📊 Classification Report")
#         report = classification_report(y_test, y_pred, target_names=['Fail', 'Pass'], output_dict=True)
#         report_df = pd.DataFrame(report).transpose()
#         st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
#     with col2:
#         st.subheader("🔥 Confusion Matrix")
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
#         xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], ax=ax)
#         ax.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
#         ax.set_xlabel('Predicted', fontsize=12, weight='bold')
#         ax.set_ylabel('Actual', fontsize=12, weight='bold')
#         st.pyplot(fig)
    
#     st.markdown("---")
    
#     st.subheader("🌟 Feature Importance")
#     feature_importance = pd.DataFrame({
#         'Feature': X.columns,
#         'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
#     ax.set_title('Feature Importance Analysis', fontsize=14, weight='bold', pad=20)
#     ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
#     ax.set_ylabel('Features', fontsize=12, weight='bold')
#     st.pyplot(fig)

# # Prediction Page
# elif page == "🔮 Prediction":
#     st.header("🔮 Student Performance Prediction")
    
#     st.markdown("### Enter Student Information")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         gender = st.selectbox("Gender", ['Male', 'Female'])
#         study_hours = st.slider("Study Hours per Day", 1, 12, 6)
    
#     with col2:
#         attendance = st.slider("Attendance %", 0, 100, 75)
#         previous_grade = st.slider("Previous Grade", 0, 100, 70)
    
#     with col3:
#         parent_education = st.selectbox("Parent Education", ['High School', 'Graduate', 'Postgraduate'])
#         internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
    
#     if st.button("🎯 Predict Performance", use_container_width=True):
#         # Encode inputs
#         gender_encoded = le_gender.transform([gender])[0]
#         parent_encoded = le_parent.transform([parent_education])[0]
#         internet_encoded = le_internet.transform([internet_access])[0]
        
#         # Make prediction
#         input_data = np.array([[gender_encoded, study_hours, attendance, previous_grade,parent_encoded, internet_encoded]])
#         prediction = model.predict(input_data)[0]
#         prediction_proba = model.predict_proba(input_data)[0]
        
#         result = le_performance.inverse_transform([prediction])[0]
        
#         st.markdown("---")
#         st.markdown("### 🎊 Prediction Results")
        
#         col1, col2, col3 = st.columns([1, 2, 1])
        
#         with col2:
#             if result == 'Pass':
#                 st.success("### ✅ Predicted Performance: PASS")
#                 st.balloons()
#             else:
#                 st.error("### ❌ Predicted Performance: FAIL")
            
#             st.markdown(f"*Confidence:* {max(prediction_proba)*100:.2f}%")
            
#             # Probability visualization
#             fig, ax = plt.subplots(figsize=(8, 4))
#             classes = ['Fail', 'Pass']
#             colors = ['#f87171', '#667eea']
#             ax.barh(classes, prediction_proba, color=colors)
#             ax.set_xlabel('Probability', fontsize=12, weight='bold')
#             ax.set_title('Prediction Confidence', fontsize=14, weight='bold', pad=20)
#             ax.set_xlim([0, 1])
#             for i, v in enumerate(prediction_proba):
#                 ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')
#             st.pyplot(fig)

# # Data Analysis Page
# elif page == "📊 Data Analysis":
#     st.header("📊 Advanced Data Analysis")
    
#     tab1, tab2, tab3 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights"])
    
#     with tab1:
#         st.subheader("Correlation Heatmap")
#         fig, ax = plt.subplots(figsize=(10, 8))
#         correlation = df_encoded.corr()
#         sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
#                    center=0, square=True, ax=ax)
#         ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
#         st.pyplot(fig)
    
#     with tab2:
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Attendance Distribution")
#             fig, ax = plt.subplots(figsize=(8, 6))
#             sns.histplot(data=df, x='Attendance', hue='Performance', 
#                         palette=['#f87171', '#667eea'], kde=True, ax=ax)
#             ax.set_title('Attendance Impact', fontsize=14, weight='bold', pad=20)
#             st.pyplot(fig)
        
#         with col2:
#             st.subheader("Previous Grade Distribution")
#             fig, ax = plt.subplots(figsize=(8, 6))
#             sns.histplot(data=df, x='Previous_Grade', hue='Performance', 
#                         palette=['#f87171', '#667eea'], kde=True, ax=ax)
#             ax.set_title('Previous Grade Impact', fontsize=14, weight='bold', pad=20)
#             st.pyplot(fig)
    
#     with tab3:
#         st.subheader("📌 Key Insights")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.info(f"""
#             *Academic Performance Insights:*
            
#             The average study hours for passing students is {df[df['Performance']=='Pass']['Study_Hours'].mean():.1f} hours, 
#             compared to {df[df['Performance']=='Fail']['Study_Hours'].mean():.1f} hours for failing students. 
#             This demonstrates a clear correlation between dedicated study time and academic success.
            
#             Attendance rates show a similar pattern, with passing students maintaining an average of 
#             {df[df['Performance']=='Pass']['Attendance'].mean():.1f}% attendance versus 
#             {df[df['Performance']=='Fail']['Attendance'].mean():.1f}% for those who fail.
#             """)
        
#         with col2:
#             st.success(f"""
#             *Model Performance Summary:*
            
#             The Random Forest Classifier achieves {accuracy*100:.1f}% accuracy in predicting student outcomes. 
#             The most influential factors in determining performance are study hours, attendance, and previous grades.
            
#             Students with consistent attendance above 80% and regular study habits of 6+ hours daily 
#             demonstrate significantly higher success rates in academic performance.
#             """)
        
#         st.markdown("---")
#         st.subheader("Gender-wise Performance")
#         gender_performance = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
#         fig, ax = plt.subplots(figsize=(10, 6))
#         gender_performance.plot(kind='bar', ax=ax, color=['#f87171', '#667eea'])
#         ax.set_title('Performance by Gender', fontsize=14, weight='bold', pad=20)
#         ax.set_xlabel('Gender', fontsize=12, weight='bold')
#         ax.set_ylabel('Percentage', fontsize=12, weight='bold')
#         ax.legend(title='Performance', title_fontsize=12)
#         plt.xticks(rotation=0)
#         st.pyplot(fig)

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #64748b; padding: 20px;'>
#         <p><strong>Some of my Friends: Arslan, Ahmer, Wasey, Furqan, Haseeb, Jawad, Akbar and Jawad Baloch</strong></p>
#         <p>Created By Rashid Ali Soomro | Thanks to Sir Arham and Ma'am Moona Shah</p>
#     </div>
#     """, unsafe_allow_html=True)



























# # 2nd






# import streamlit as st
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Student Performance Predictor",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for attractive design
# st.markdown("""
#     <style>
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
#     .stApp {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#     }
#     .metric-card {
#         background: white;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 10px 0;
#     }
#     .title-text {
#         color: #1e3a8a;
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#     .subtitle-text {
#         color: #475569;
#         font-size: 1.2rem;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     div[data-testid="stMetricValue"] {
#         font-size: 2rem;
#         color: #1e3a8a;
#     }
#     .stButton>button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 10px 30px;
#         border-radius: 25px;
#         font-weight: bold;
#         transition: all 0.3s;
#     }
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(0,0,0,0.2);
#     }
#     .upload-section {
#         background: white;
#         padding: 30px;
#         border-radius: 15px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 20px 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title
# st.markdown('<p class="title-text">🎓 Student Performance Prediction System</p>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle-text">Advanced Machine Learning Analytics for Educational Excellence</p>', unsafe_allow_html=True)

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None

# # Sidebar
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
#     st.title("📊 Navigation")
    
#     if st.session_state.data_loaded:
#         page = st.radio("Select Page:", ["🏠 Dashboard", "📈 Model Training", "🔮 Prediction", "📊 Data Analysis"])
#     else:
#         page = "📁 Data Upload"
    
#     st.markdown("---")
#     st.info("*About:* This system uses Random Forest Classifier to predict student performance based on various academic and demographic factors.")
    
#     if st.session_state.data_loaded:
#         st.success(f"✅ Data Loaded: {len(st.session_state.df)} records")
#         if st.button("🔄 Upload New Dataset"):
#             st.session_state.data_loaded = False
#             st.session_state.df = None
#             st.rerun()

# # Data Upload Page
# if not st.session_state.data_loaded:
#     st.header("📁 Upload Student Dataset")
    
#     st.markdown("""
#         <div class='upload-section'>
#             <h3>📋 Dataset Requirements</h3>
#             <p>Please ensure your CSV file contains the following columns:</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         *Required Columns:*
#         - Gender (Male/Female)
#         - Study_Hours (numeric)
#         - Attendance (numeric, 0-100)
#         - Previous_Grade (numeric, 0-100)
#         - Parent_Education (High School/Graduate/Postgraduate)
#         - Internet_Access (Yes/No)
#         - Performance (Pass/Fail)
#         """)
    
#     with col2:
#         st.markdown("""
#         *Example Data Format:*
        
#         Gender,Study_Hours,Attendance,Previous_Grade,Parent_Education,Internet_Access,Performance
#         Male,7,85,78,Graduate,Yes,Pass
#         Female,5,70,62,High School,No,Fail
#         Male,8,92,85,Postgraduate,Yes,Pass
        
#         """)
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             required_columns = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                               'Parent_Education', 'Internet_Access', 'Performance']
            
#             missing_columns = [col for col in required_columns if col not in df.columns]
            
#             if missing_columns:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
#             else:
#                 st.success("✅ Dataset uploaded successfully!")
                
#                 st.subheader("📊 Dataset Preview")
#                 st.dataframe(df.head(10), use_container_width=True)
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Records", len(df))
#                 with col2:
#                     st.metric("Features", len(df.columns)-1)
#                 with col3:
#                     pass_count = (df['Performance'] == 'Pass').sum()
#                     st.metric("Pass Count", pass_count)
#                 with col4:
#                     fail_count = (df['Performance'] == 'Fail').sum()
#                     st.metric("Fail Count", fail_count)
                
#                 if st.button("✅ Proceed with Analysis", use_container_width=True):
#                     st.session_state.df = df
#                     st.session_state.data_loaded = True
#                     st.rerun()
        
#         except Exception as e:
#             st.error(f"❌ Error loading file: {str(e)}")
    
#     st.markdown("---")
#     st.info("💡 *Tip:* You can download a sample dataset template to understand the required format.")

# else:
#     df = st.session_state.df.copy()
    
#     # -------------------------------
#     # 🧹 IMPROVED DATA CLEANING
#     # -------------------------------
#     st.write("---")
#     st.header("🧹 Data Cleaning")
    
#     # First, standardize the Performance column to only Pass/Fail
#     if 'Performance' in df.columns:
#         st.write("### 📊 Original Performance Values:")
#         st.write(df['Performance'].value_counts())
        
#         # Convert various performance indicators to Pass/Fail
#         df['Performance'] = df['Performance'].astype(str).str.strip().str.lower()
        
#         # Map common variations to Pass/Fail
#         pass_values = ['pass', 'passed', 'success', 'good', 'excellent', 'yes', '1', 'true']
#         fail_values = ['fail', 'failed', 'failure', 'poor', 'bad', 'no', '0', 'false']
        
#         df['Performance'] = df['Performance'].apply(lambda x: 
#             'Pass' if any(val in str(x).lower() for val in pass_values)
#             else 'Fail' if any(val in str(x).lower() for val in fail_values)
#             else x
#         )
        
#         # If still numeric, convert based on threshold
#         try:
#             numeric_perf = pd.to_numeric(df['Performance'], errors='coerce')
#             if numeric_perf.notna().any():
#                 threshold = numeric_perf.median()
#                 df.loc[numeric_perf.notna(), 'Performance'] = numeric_perf[numeric_perf.notna()].apply(
#                     lambda x: 'Pass' if x >= threshold else 'Fail'
#                 )
#                 st.info(f"📊 Converted numeric performance values using threshold: {threshold:.2f}")
#         except:
#             pass
        
#         # Remove any remaining invalid values
#         valid_performance = df['Performance'].isin(['Pass', 'Fail'])
#         if not valid_performance.all():
#             st.warning(f"⚠ Removing {(~valid_performance).sum()} rows with invalid Performance values")
#             df = df[valid_performance]
        
#         st.write("### ✅ Standardized Performance Values:")
#         st.write(df['Performance'].value_counts())

#     # Define which columns should remain categorical
#     categorical_columns = ['Gender', 'Parent_Education', 'Internet_Access', 'Performance']
    
#     # Clean numeric columns
#     for col in df.columns:
#         if col not in categorical_columns:
#             # Convert to string first to handle all types
#             df[col] = df[col].astype(str)
            
#             # Try multiple date formats
#             date_formats = ['%d-%b', '%d-%B', '%b-%d', '%B-%d', '%d/%m', '%m/%d']
#             date_converted = False
            
#             for date_format in date_formats:
#                 try:
#                     parsed_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
#                     if parsed_dates.notna().sum() > 0:
#                         df[col] = parsed_dates.dt.day
#                         date_converted = True
#                         st.info(f"📅 Converted date format in column '{col}' to day numbers")
#                         break
#                 except:
#                     continue
            
#             # If not a date, convert to numeric
#             if not date_converted:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Fill missing values with mean for numeric columns
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 mean_val = df[col].mean()
#                 missing_count = df[col].isna().sum()
#                 if missing_count > 0:
#                     df[col] = df[col].fillna(mean_val)
#                     st.warning(f"⚠ Filled {missing_count} missing values in '{col}' with mean: {mean_val:.2f}")
    
#     # Handle categorical columns - remove rows with missing values
#     for col in categorical_columns:
#         if col in df.columns:
#             missing_before = df[col].isna().sum()
#             if missing_before > 0:
#                 df = df.dropna(subset=[col])
#                 st.warning(f"⚠ Removed {missing_before} rows with missing '{col}' values")
    
#     # Final check - ensure all numeric columns are properly converted
#     numeric_columns = [col for col in df.columns if col not in categorical_columns]
#     for col in numeric_columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
#     st.success("✅ Data cleaned successfully!")
#     st.write("### 🔍 Cleaned Data Preview:")
#     st.dataframe(df.head())
    
#     # Show data types
#     with st.expander("📋 View Data Types"):
#         st.write(df.dtypes)
    
#     # Encode categorical variables
#     @st.cache_data
#     def encode_data(data):
#         le_gender = LabelEncoder()
#         le_parent = LabelEncoder()
#         le_internet = LabelEncoder()
#         le_performance = LabelEncoder()
        
#         data_encoded = data.copy()
#         data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
#         data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
#         data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
#         data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
        
#         return data_encoded, le_gender, le_parent, le_internet, le_performance

#     try:
#         df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)
        
#         # Train model
#         @st.cache_resource
#         def train_model(X, y):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)
#             return model, accuracy, cm, X_test, y_test, y_pred

#         X = df_encoded.drop('Performance', axis=1)
#         y = df_encoded['Performance']
        
#         # Ensure X contains only numeric data
#         X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
#         model, accuracy, cm, X_test, y_test, y_pred = train_model(X, y)

#         # Dashboard Page
#         if page == "🏠 Dashboard":
#             st.header("📊 Performance Overview")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Total Students", len(df), delta="Active")
#             with col2:
#                 pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
#                 st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
#             with col3:
#                 st.metric("Model Accuracy", f"{accuracy*100:.1f}%", delta="High")
#             with col4:
#                 avg_study = df['Study_Hours'].mean()
#                 st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
            
#             st.markdown("---")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("📊 Performance Distribution")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 performance_counts = df['Performance'].value_counts()
#                 colors = ['#667eea', '#f87171']
#                 ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', 
#                        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
#                 ax.set_title('Student Performance Distribution', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with col2:
#                 st.subheader("📈 Study Hours vs Performance")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.boxplot(data=df, x='Performance', y='Study_Hours', palette=['#667eea', '#f87171'], ax=ax)
#                 ax.set_title('Study Hours Impact on Performance', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Performance', fontsize=12, weight='bold')
#                 ax.set_ylabel('Study Hours', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("📋 Student Data Overview")
#             st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

#         # Model Training Page
#         elif page == "📈 Model Training":
#             st.header("🤖 Machine Learning Model Training")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("🎯 Model Performance Metrics")
#                 st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
#                 st.metric("Training Samples", len(X) - len(X_test))
#                 st.metric("Testing Samples", len(X_test))
                
#                 st.markdown("---")
#                 st.subheader("📊 Classification Report")
#                 report = classification_report(y_test, y_pred, target_names=['Fail', 'Pass'], output_dict=True)
#                 report_df = pd.DataFrame(report).transpose()
#                 st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
#             with col2:
#                 st.subheader("🔥 Confusion Matrix")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
#                            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'], ax=ax)
#                 ax.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Predicted', fontsize=12, weight='bold')
#                 ax.set_ylabel('Actual', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("🌟 Feature Importance")
#             feature_importance = pd.DataFrame({
#                 'Feature': X.columns,
#                 'Importance': model.feature_importances_
#             }).sort_values('Importance', ascending=False)
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
#             ax.set_title('Feature Importance Analysis', fontsize=14, weight='bold', pad=20)
#             ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
#             ax.set_ylabel('Features', fontsize=12, weight='bold')
#             st.pyplot(fig)

#         # Prediction Page
#         elif page == "🔮 Prediction":
#             st.header("🔮 Student Performance Prediction")
            
#             st.markdown("### Enter Student Information")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 gender = st.selectbox("Gender", df['Gender'].unique())
#                 study_hours = st.slider("Study Hours per Day", 
#                                        int(df['Study_Hours'].min()), 
#                                        int(df['Study_Hours'].max()), 
#                                        int(df['Study_Hours'].mean()))
            
#             with col2:
#                 attendance = st.slider("Attendance %", 
#                                       int(df['Attendance'].min()), 
#                                       int(df['Attendance'].max()), 
#                                       int(df['Attendance'].mean()))
#                 previous_grade = st.slider("Previous Grade", 
#                                           int(df['Previous_Grade'].min()), 
#                                           int(df['Previous_Grade'].max()), 
#                                           int(df['Previous_Grade'].mean()))
            
#             with col3:
#                 parent_education = st.selectbox("Parent Education", df['Parent_Education'].unique())
#                 internet_access = st.selectbox("Internet Access", df['Internet_Access'].unique())
            
#             if st.button("🎯 Predict Performance", use_container_width=True):
#                 gender_encoded = le_gender.transform([gender])[0]
#                 parent_encoded = le_parent.transform([parent_education])[0]
#                 internet_encoded = le_internet.transform([internet_access])[0]
                
#                 input_data = np.array([[gender_encoded, study_hours, attendance, previous_grade, 
#                                        parent_encoded, internet_encoded]])
#                 prediction = model.predict(input_data)[0]
#                 prediction_proba = model.predict_proba(input_data)[0]
                
#                 result = le_performance.inverse_transform([prediction])[0]
                
#                 st.markdown("---")
#                 st.markdown("### 🎊 Prediction Results")
                
#                 col1, col2, col3 = st.columns([1, 2, 1])
                
#                 with col2:
#                     if result == 'Pass':
#                         st.success("### ✅ Predicted Performance: PASS")
#                         st.balloons()
#                     else:
#                         st.error("### ❌ Predicted Performance: FAIL")
                    
#                     st.markdown(f"*Confidence:* {max(prediction_proba)*100:.2f}%")
                    
#                     fig, ax = plt.subplots(figsize=(8, 4))
#                     classes = ['Fail', 'Pass']
#                     colors = ['#f87171', '#667eea']
#                     ax.barh(classes, prediction_proba, color=colors)
#                     ax.set_xlabel('Probability', fontsize=12, weight='bold')
#                     ax.set_title('Prediction Confidence', fontsize=14, weight='bold', pad=20)
#                     ax.set_xlim([0, 1])
#                     for i, v in enumerate(prediction_proba):
#                         ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')
#                     st.pyplot(fig)

#         # Data Analysis Page
#         elif page == "📊 Data Analysis":
#             st.header("📊 Advanced Data Analysis")
            
#             tab1, tab2, tab3 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights"])
            
#             with tab1:
#                 st.subheader("Correlation Heatmap")
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 correlation = df_encoded.corr()
#                 sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
#                            center=0, square=True, ax=ax)
#                 ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with tab2:
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Attendance Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Attendance', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Attendance Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
                
#                 with col2:
#                     st.subheader("Previous Grade Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Previous_Grade', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Previous Grade Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
            
#             with tab3:
#                 st.subheader("📌 Key Insights")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     avg_study_pass = df[df['Performance']=='Pass']['Study_Hours'].mean()
#                     avg_study_fail = df[df['Performance']=='Fail']['Study_Hours'].mean()
#                     avg_att_pass = df[df['Performance']=='Pass']['Attendance'].mean()
#                     avg_att_fail = df[df['Performance']=='Fail']['Attendance'].mean()
                    
#                     st.info(f"""
#                     *Academic Performance Insights:*
                    
#                     The average study hours for passing students is {avg_study_pass:.1f} hours, 
#                     compared to {avg_study_fail:.1f} hours for failing students. 
#                     This demonstrates a clear correlation between dedicated study time and academic success.
                    
#                     Attendance rates show a similar pattern, with passing students maintaining an average of 
#                     {avg_att_pass:.1f}% attendance versus {avg_att_fail:.1f}% for those who fail.
#                     """)
                
#                 with col2:
#                     st.success(f"""
#                     *Model Performance Summary:*
                    
#                     The Random Forest Classifier achieves {accuracy*100:.1f}% accuracy in predicting student outcomes. 
#                     The most influential factors in determining performance are study hours, attendance, and previous grades.
                    
#                     Students with consistent attendance above 80% and regular study habits demonstrate 
#                     significantly higher success rates in academic performance.
#                     """)
                
#                 st.markdown("---")
#                 st.subheader("Gender-wise Performance")
#                 gender_performance = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 gender_performance.plot(kind='bar', ax=ax, color=['#f87171', '#667eea'])
#                 ax.set_title('Performance by Gender', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Gender', fontsize=12, weight='bold')
#                 ax.set_ylabel('Percentage', fontsize=12, weight='bold')
#                 ax.legend(title='Performance', title_fontsize=12)
#                 plt.xticks(rotation=0)
#                 st.pyplot(fig)
    
#     except Exception as e:
#         st.error(f"❌ Error during processing: {str(e)}")
#         st.write("Please check your data and try again. Make sure:")
#         st.write("- All numeric columns contain valid numbers")
#         st.write("- Categorical columns contain expected values")
#         st.write("- There are no completely empty columns")

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #64748b; padding: 20px;'>
#         <p><strong>Student Performance Prediction System</strong></p>
#         <p>Powered by Machine Learning | Built with Streamlit</p>
#     </div>
#     """, unsafe_allow_html=True)







# 3rd



# import streamlit as st
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Student Performance Predictor",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for attractive design
# st.markdown("""
#     <style>
#     .main {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     }
#     .stApp {
#         background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
#     }
#     .metric-card {
#         background: white;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 10px 0;
#     }
#     .title-text {
#         color: #1e3a8a;
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         margin-bottom: 10px;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
#     .subtitle-text {
#         color: #475569;
#         font-size: 1.2rem;
#         text-align: center;
#         margin-bottom: 30px;
#     }
#     div[data-testid="stMetricValue"] {
#         font-size: 2rem;
#         color: #1e3a8a;
#     }
#     .stButton>button {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 10px 30px;
#         border-radius: 25px;
#         font-weight: bold;
#         transition: all 0.3s;
#     }
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(0,0,0,0.2);
#     }
#     .upload-section {
#         background: white;
#         padding: 30px;
#         border-radius: 15px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 20px 0;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title
# st.markdown('<h1 class="title-text">🎓 Student Performance Prediction System</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle-text">Advanced Machine Learning Analytics for Educational Excellence</p>', unsafe_allow_html=True)

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None

# # Sidebar
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
#     st.title("📊 Navigation")
    
#     if st.session_state.data_loaded:
#         page = st.radio("Select Page:", ["🏠 Dashboard", "📈 Model Training", "🔮 Prediction", "📊 Data Analysis"])
#     else:
#         page = "📁 Data Upload"
    
#     st.markdown("---")
#     st.info("*About:* This system uses Random Forest Classifier to predict student performance based on various academic and demographic factors.")
    
#     if st.session_state.data_loaded:
#         st.success(f"✅ Data Loaded: {len(st.session_state.df)} records")
#         if st.button("🔄 Upload New Dataset"):
#             st.session_state.data_loaded = False
#             st.session_state.df = None
#             st.rerun()

# # Data Upload Page
# if not st.session_state.data_loaded:
#     st.header("📁 Upload Student Dataset")
    
#     st.markdown("""
#         <div class='upload-section'>
#             <h3>📋 Dataset Requirements</h3>
#             <p>Please ensure your CSV file contains the following columns:</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         *Required Columns:*
#         - Gender (Male/Female)
#         - Study_Hours (numeric)
#         - Attendance (numeric, 0-100)
#         - Previous_Grade (numeric, 0-100)
#         - Parent_Education (High School/Graduate/Postgraduate)
#         - Internet_Access (Yes/No)
#         - Performance (Pass/Fail)
#         """)
    
#     with col2:
#         st.markdown("""
#         *Example Data Format:*
        
#         Gender,Study_Hours,Attendance,Previous_Grade,Parent_Education,Internet_Access,Performance
#         Male,7,85,78,Graduate,Yes,Pass
#         Female,5,70,62,High School,No,Fail
#         Male,8,92,85,Postgraduate,Yes,Pass
        
#         """)
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             required_columns = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                               'Parent_Education', 'Internet_Access', 'Performance']
            
#             missing_columns = [col for col in required_columns if col not in df.columns]
            
#             if missing_columns:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
#             else:
#                 st.success("✅ Dataset uploaded successfully!")
                
#                 st.subheader("📊 Dataset Preview")
#                 st.dataframe(df.head(10), use_container_width=True)
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Records", len(df))
#                 with col2:
#                     st.metric("Features", len(df.columns)-1)
#                 with col3:
#                     pass_count = (df['Performance'] == 'Pass').sum()
#                     st.metric("Pass Count", pass_count)
#                 with col4:
#                     fail_count = (df['Performance'] == 'Fail').sum()
#                     st.metric("Fail Count", fail_count)
                
#                 if st.button("✅ Proceed with Analysis", use_container_width=True):
#                     st.session_state.df = df
#                     st.session_state.data_loaded = True
#                     st.rerun()
        
#         except Exception as e:
#             st.error(f"❌ Error loading file: {str(e)}")
    
#     st.markdown("---")
#     st.info("💡 *Tip:* You can download a sample dataset template to understand the required format.")

# else:
#     df = st.session_state.df.copy()
    
#     # -------------------------------
#     # 🧹 IMPROVED DATA CLEANING
#     # -------------------------------
#     st.write("---")
#     st.header("🧹 Data Cleaning")
    
#     # First, standardize the Performance column to only Pass/Fail
#     if 'Performance' in df.columns:
#         st.write("### 📊 Original Performance Values:")
#         st.write(df['Performance'].value_counts())
        
#         # Convert various performance indicators to Pass/Fail
#         df['Performance'] = df['Performance'].astype(str).str.strip().str.lower()
        
#         # Map common variations to Pass/Fail
#         pass_values = ['pass', 'passed', 'success', 'good', 'excellent', 'yes', '1', 'true']
#         fail_values = ['fail', 'failed', 'failure', 'poor', 'bad', 'no', '0', 'false']
        
#         df['Performance'] = df['Performance'].apply(lambda x: 
#             'Pass' if any(val in str(x).lower() for val in pass_values)
#             else 'Fail' if any(val in str(x).lower() for val in fail_values)
#             else x
#         )
        
#         # If still numeric, convert based on threshold
#         try:
#             numeric_perf = pd.to_numeric(df['Performance'], errors='coerce')
#             if numeric_perf.notna().any():
#                 threshold = numeric_perf.median()
#                 df.loc[numeric_perf.notna(), 'Performance'] = numeric_perf[numeric_perf.notna()].apply(
#                     lambda x: 'Pass' if x >= threshold else 'Fail'
#                 )
#                 st.info(f"📊 Converted numeric performance values using threshold: {threshold:.2f}")
#         except:
#             pass
        
#         # Remove any remaining invalid values
#         valid_performance = df['Performance'].isin(['Pass', 'Fail'])
#         if not valid_performance.all():
#             st.warning(f"⚠ Removing {(~valid_performance).sum()} rows with invalid Performance values")
#             df = df[valid_performance]
        
#         st.write("### ✅ Standardized Performance Values:")
#         st.write(df['Performance'].value_counts())

#     # Define which columns should remain categorical
#     categorical_columns = ['Gender', 'Parent_Education', 'Internet_Access', 'Performance']
    
#     # Clean numeric columns
#     for col in df.columns:
#         if col not in categorical_columns:
#             # Convert to string first to handle all types
#             df[col] = df[col].astype(str)
            
#             # Try multiple date formats
#             date_formats = ['%d-%b', '%d-%B', '%b-%d', '%B-%d', '%d/%m', '%m/%d']
#             date_converted = False
            
#             for date_format in date_formats:
#                 try:
#                     parsed_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
#                     if parsed_dates.notna().sum() > 0:
#                         df[col] = parsed_dates.dt.day
#                         date_converted = True
#                         st.info(f"📅 Converted date format in column '{col}' to day numbers")
#                         break
#                 except:
#                     continue
            
#             # If not a date, convert to numeric
#             if not date_converted:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Fill missing values with mean for numeric columns
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 mean_val = df[col].mean()
#                 missing_count = df[col].isna().sum()
#                 if missing_count > 0:
#                     df[col] = df[col].fillna(mean_val)
#                     st.warning(f"⚠ Filled {missing_count} missing values in '{col}' with mean: {mean_val:.2f}")
    
#     # Handle categorical columns - remove rows with missing values
#     for col in categorical_columns:
#         if col in df.columns:
#             missing_before = df[col].isna().sum()
#             if missing_before > 0:
#                 df = df.dropna(subset=[col])
#                 st.warning(f"⚠ Removed {missing_before} rows with missing '{col}' values")
    
#     # Final check - ensure all numeric columns are properly converted
#     numeric_columns = [col for col in df.columns if col not in categorical_columns]
#     for col in numeric_columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
#     st.success("✅ Data cleaned successfully!")
#     st.write("### 🔍 Cleaned Data Preview:")
#     st.dataframe(df.head())
    
#     # Show data types
#     with st.expander("📋 View Data Types"):
#         st.write(df.dtypes)
    
#     # Encode categorical variables
#     @st.cache_data
#     def encode_data(data):
#         le_gender = LabelEncoder()
#         le_parent = LabelEncoder()
#         le_internet = LabelEncoder()
#         le_performance = LabelEncoder()
        
#         data_encoded = data.copy()
#         data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
#         data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
#         data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
#         data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
        
#         return data_encoded, le_gender, le_parent, le_internet, le_performance

#     try:
#         df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)
        
#         # Train model
#         @st.cache_resource
#         def train_model(X, y):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)
#             return model, accuracy, cm, X_test, y_test, y_pred

#         X = df_encoded.drop('Performance', axis=1)
#         y = df_encoded['Performance']
        
#         # Ensure X contains only numeric data
#         X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
#         model, accuracy, cm, X_test, y_test, y_pred = train_model(X, y)

#         # Dashboard Page
#         if page == "🏠 Dashboard":
#             st.header("📊 Performance Overview")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Total Students", len(df), delta="Active")
#             with col2:
#                 pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
#                 st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
#             with col3:
#                 st.metric("Model Accuracy", f"{accuracy*100:.1f}%", delta="High")
#             with col4:
#                 avg_study = df['Study_Hours'].mean()
#                 st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
            
#             st.markdown("---")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("📊 Performance Distribution")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 performance_counts = df['Performance'].value_counts()
#                 colors = ['#667eea', '#f87171']
#                 ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', 
#                        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
#                 ax.set_title('Student Performance Distribution', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with col2:
#                 st.subheader("📈 Study Hours vs Performance")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.boxplot(data=df, x='Performance', y='Study_Hours', palette=['#667eea', '#f87171'], ax=ax)
#                 ax.set_title('Study Hours Impact on Performance', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Performance', fontsize=12, weight='bold')
#                 ax.set_ylabel('Study Hours', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("📋 Student Data Overview")
#             st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

#         # Model Training Page
#         elif page == "📈 Model Training":
#             st.header("🤖 Machine Learning Model Training")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("🎯 Model Performance Metrics")
#                 st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
#                 st.metric("Training Samples", len(X) - len(X_test))
#                 st.metric("Testing Samples", len(X_test))
                
#                 st.markdown("---")
#                 st.subheader("📊 Classification Report")
#                 try:
#                     # Get unique classes in predictions
#                     unique_classes = sorted(y_test.unique())
#                     class_names = ['Fail', 'Pass'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
                    
#                     report = classification_report(y_test, y_pred, 
#                                                   labels=unique_classes,
#                                                   target_names=class_names, 
#                                                   output_dict=True,
#                                                   zero_division=0)
#                     report_df = pd.DataFrame(report).transpose()
#                     st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
#                 except Exception as e:
#                     st.error(f"Could not generate classification report: {str(e)}")
#                     st.write("Prediction distribution:")
#                     st.write(pd.Series(y_pred).value_counts())
            
#             with col2:
#                 st.subheader("🔥 Confusion Matrix")
#                 fig, ax = plt.subplots(figsize=(8, 6))
                
#                 # Get unique classes for labels
#                 unique_classes = sorted(y_test.unique())
#                 class_labels = ['Fail', 'Pass'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
                
#                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
#                            xticklabels=class_labels, yticklabels=class_labels, ax=ax)
#                 ax.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Predicted', fontsize=12, weight='bold')
#                 ax.set_ylabel('Actual', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("🌟 Feature Importance")
#             feature_importance = pd.DataFrame({
#                 'Feature': X.columns,
#                 'Importance': model.feature_importances_
#             }).sort_values('Importance', ascending=False)
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
#             ax.set_title('Feature Importance Analysis', fontsize=14, weight='bold', pad=20)
#             ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
#             ax.set_ylabel('Features', fontsize=12, weight='bold')
#             st.pyplot(fig)

#         # Prediction Page
#         elif page == "🔮 Prediction":
#             st.header("🔮 Student Performance Prediction")
            
#             st.markdown("### Enter Student Information")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 gender = st.selectbox("Gender", df['Gender'].unique())
                
#                 # Handle Study_Hours with same min/max
#                 study_min = int(df['Study_Hours'].min())
#                 study_max = int(df['Study_Hours'].max())
#                 study_mean = int(df['Study_Hours'].mean())
                
#                 if study_min == study_max:
#                     st.info(f"Study Hours: {study_min} (constant in dataset)")
#                     study_hours = study_min
#                 else:
#                     study_hours = st.slider("Study Hours per Day", 
#                                            study_min, 
#                                            study_max, 
#                                            study_mean)
            
#             with col2:
#                 # Handle Attendance with same min/max
#                 att_min = int(df['Attendance'].min())
#                 att_max = int(df['Attendance'].max())
#                 att_mean = int(df['Attendance'].mean())
                
#                 if att_min == att_max:
#                     st.info(f"Attendance: {att_min}% (constant in dataset)")
#                     attendance = att_min
#                 else:
#                     attendance = st.slider("Attendance %", 
#                                           att_min, 
#                                           att_max, 
#                                           att_mean)
                
#                 # Handle Previous_Grade with same min/max
#                 grade_min = int(df['Previous_Grade'].min())
#                 grade_max = int(df['Previous_Grade'].max())
#                 grade_mean = int(df['Previous_Grade'].mean())
                
#                 if grade_min == grade_max:
#                     st.info(f"Previous Grade: {grade_min} (constant in dataset)")
#                     previous_grade = grade_min
#                 else:
#                     previous_grade = st.slider("Previous Grade", 
#                                               grade_min, 
#                                               grade_max, 
#                                               grade_mean)
            
#             with col3:
#                 parent_education = st.selectbox("Parent Education", df['Parent_Education'].unique())
#                 internet_access = st.selectbox("Internet Access", df['Internet_Access'].unique())
            
#             if st.button("🎯 Predict Performance", use_container_width=True):
#                 gender_encoded = le_gender.transform([gender])[0]
#                 parent_encoded = le_parent.transform([parent_education])[0]
#                 internet_encoded = le_internet.transform([internet_access])[0]
                
#                 input_data = np.array([[gender_encoded, study_hours, attendance, previous_grade, 
#                                        parent_encoded, internet_encoded]])
#                 prediction = model.predict(input_data)[0]
#                 prediction_proba = model.predict_proba(input_data)[0]
                
#                 result = le_performance.inverse_transform([prediction])[0]
                
#                 st.markdown("---")
#                 st.markdown("### 🎊 Prediction Results")
                
#                 col1, col2, col3 = st.columns([1, 2, 1])
                
#                 with col2:
#                     if result == 'Pass':
#                         st.success("### ✅ Predicted Performance: PASS")
#                         st.balloons()
#                     else:
#                         st.error("### ❌ Predicted Performance: FAIL")
                    
#                     st.markdown(f"*Confidence:* {max(prediction_proba)*100:.2f}%")
                    
#                     fig, ax = plt.subplots(figsize=(8, 4))
#                     classes = ['Fail', 'Pass']
#                     colors = ['#f87171', '#667eea']
#                     ax.barh(classes, prediction_proba, color=colors)
#                     ax.set_xlabel('Probability', fontsize=12, weight='bold')
#                     ax.set_title('Prediction Confidence', fontsize=14, weight='bold', pad=20)
#                     ax.set_xlim([0, 1])
#                     for i, v in enumerate(prediction_proba):
#                         ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')
#                     st.pyplot(fig)

#         # Data Analysis Page
#         elif page == "📊 Data Analysis":
#             st.header("📊 Advanced Data Analysis")
            
#             tab1, tab2, tab3 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights"])
            
#             with tab1:
#                 st.subheader("Correlation Heatmap")
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 correlation = df_encoded.corr()
#                 sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
#                            center=0, square=True, ax=ax)
#                 ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with tab2:
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Attendance Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Attendance', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Attendance Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
                
#                 with col2:
#                     st.subheader("Previous Grade Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Previous_Grade', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Previous Grade Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
            
#             with tab3:
#                 st.subheader("📌 Key Insights")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     avg_study_pass = df[df['Performance']=='Pass']['Study_Hours'].mean()
#                     avg_study_fail = df[df['Performance']=='Fail']['Study_Hours'].mean()
#                     avg_att_pass = df[df['Performance']=='Pass']['Attendance'].mean()
#                     avg_att_fail = df[df['Performance']=='Fail']['Attendance'].mean()
                    
#                     st.info(f"""
#                     *Academic Performance Insights:*
                    
#                     The average study hours for passing students is {avg_study_pass:.1f} hours, 
#                     compared to {avg_study_fail:.1f} hours for failing students. 
#                     This demonstrates a clear correlation between dedicated study time and academic success.
                    
#                     Attendance rates show a similar pattern, with passing students maintaining an average of 
#                     {avg_att_pass:.1f}% attendance versus {avg_att_fail:.1f}% for those who fail.
#                     """)
                
#                 with col2:
#                     st.success(f"""
#                     *Model Performance Summary:*
                    
#                     The Random Forest Classifier achieves {accuracy*100:.1f}% accuracy in predicting student outcomes. 
#                     The most influential factors in determining performance are study hours, attendance, and previous grades.
                    
#                     Students with consistent attendance above 80% and regular study habits demonstrate 
#                     significantly higher success rates in academic performance.
#                     """)
                
#                 st.markdown("---")
#                 st.subheader("Gender-wise Performance")
#                 gender_performance = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 gender_performance.plot(kind='bar', ax=ax, color=['#f87171', '#667eea'])
#                 ax.set_title('Performance by Gender', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Gender', fontsize=12, weight='bold')
#                 ax.set_ylabel('Percentage', fontsize=12, weight='bold')
#                 ax.legend(title='Performance', title_fontsize=12)
#                 plt.xticks(rotation=0)
#                 st.pyplot(fig)
    
#     except Exception as e:
#         st.error(f"❌ Error during processing: {str(e)}")
#         st.write("Please check your data and try again. Make sure:")
#         st.write("- All numeric columns contain valid numbers")
#         st.write("- Categorical columns contain expected values")
#         st.write("- There are no completely empty columns")

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #64748b; padding: 20px;'>
#         <p><strong>Student Performance Prediction System</strong></p>
#         <p>Powered by Machine Learning | Built with Streamlit</p>
#     </div>
#     """, unsafe_allow_html=True)









# # # Claude final Version.....!



# import streamlit as st
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Student Performance Predictor",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )


# # Custom CSS for an ultra-modern, visually stunning design
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Montserrat:wght@700;900&display=swap');

#     /* Main Layout Background */
#     .main {
#         background: linear-gradient(135deg, #74ABE2 0%, #5563DE 50%, #A683E3 100%);
#         font-family: 'Poppins', sans-serif;
#         color: #1e293b;
#     }

#     /* Streamlit App Background */
#     .stApp {
#         background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
#         font-family: 'Poppins', sans-serif;
#     }

#     /* Metric Cards */
#     .metric-card {
#         background: linear-gradient(145deg, #ffffff 0%, #f3f4f6 100%);
#         padding: 24px;
#         border-radius: 16px;
#         box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
#         margin: 15px 0;
#         transition: transform 0.3s ease, box-shadow 0.3s ease;
#         hover: transform 0.3s ease;
#         hover: box-shadow 0.3s ease;
#     }
#     .metric-card:hover {
#         transform: translateY(-4px);
#         box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
#         animation: gradientShift 5s ease infinite;
#     }

#     # /* Headings */
#     # .title-text {
#     #     font-family: 'Montserrat', sans-serif;
#     #     color: #2b2d42;
#     #     font-size: 3.8rem;
#     #     font-weight: 900;
#     #     text-align: center;
#     #     margin-bottom: 10px;
#     #     text-shadow: 2px 2px 12px rgba(67, 56, 202, 0.25);
#     #     letter-spacing: -0.5px;
#     #     animation: gradientShift 5s ease infinite;
#     # }    
        
# /* Title with Gradient Text */
#     .title-text {
#         font-family: 'Montserrat', sans-serif;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         font-size: 4rem;
#         font-weight: 900;
#         text-align: center;
#         margin-bottom: 10px;
#         letter-spacing: -2px;
#         text-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
#         animation: titlePulse 3s ease-in-out infinite;
#     }
    
#     @keyframes titlePulse {
#         0%, 100% { transform: scale(1); }
#         50% { transform: scale(1.02); }
#     }
    
    

#     .subtitle-text {
#         font-family: 'Poppins', sans-serif;
#         color: #475569;
#         font-size: 1.25rem;
#         font-weight: 400;
#         text-align: center;
#         margin-bottom: 35px;
#         letter-spacing: 0.6px;
#         animation: titlePulse 3s ease-in-out infinite;
#     }
    
#     @keyframes subtitlePulse {
#         0%, 100% { transform: scale(1); }
#         50% { transform: scale(1.02); }
#     }

#     /* Metrics */
#     div[data-testid="stMetricValue"] {
#         font-family: 'Montserrat', sans-serif;
#         font-size: 2.4rem;
#         font-weight: 800;
#         color: #4338ca;
#         text-shadow: 1px 1px 8px rgba(99, 102, 241, 0.25);
#     }
#     div[data-testid="stMetricLabel"] {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1rem;
#         font-weight: 600;
#         color: #334155;
#     }

#     /* Buttons */
#     .stButton>button {
#         font-family: 'Poppins', sans-serif;
#         background: linear-gradient(90deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
#         color: white;
#         border: none;
#         padding: 14px 38px;
#         border-radius: 30px;
#         font-weight: 700;
#         font-size: 1.1rem;
#         letter-spacing: 0.6px;
#         transition: all 0.3s ease;
#         background-size: 200% 200%;
#         animation: gradientShift 5s ease infinite;
#     }
#     @keyframes gradientShift {
#         0% { background-position: 0% 50%; }
#         50% { background-position: 100% 50%; }
#         100% { background-position: 0% 50%; }
#     }
#     .stButton>button:hover {
#         transform: translateY(-3px) scale(1.03);
#         box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
#     }

#     /* Upload Section */
#     .upload-section {
#         background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
#         padding: 32px;
#         border-radius: 18px;
#         box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
#         margin: 25px 0;
#         border-left: 6px solid #6366f1;
#         transition: box-shadow 0.3s ease;
#         animation: gradientShift 5s ease infinite;
        
#     }
#     .upload-section:hover {
#         transform: translateY(-3px) scale(1.03);
#         box-shadow: 0 12px 30px rgba(99, 102, 241, 0.2);
        
#     }
    
    

#     /* Tabs Styling */
#     .stTabs [data-baseweb="tab-list"] button {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1.1rem;
#         font-weight: 600;
#         color: #374151;
#         background: linear-gradient(90deg, #f9fafb 0%, #e2e8f0 100%);
#         border-radius: 12px;
#         margin-right: 6px;
#         transition: all 0.3s ease;
#         animation: gradientShift 5s ease infinite;
#     }
#     .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
#         background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
#         color: white;
#         box-shadow: 0 6px 12px rgba(99, 102, 241, 0.25);
#     }

#     /* Headers */
#     h1, h2, h3 {
#         font-family: 'Montserrat', sans-serif;
#         font-weight: 700;
#         color: #1e293b;
#         animation: gradientShift 5s ease infinite;
#     }

#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 10px;
#         animation: gradientShift 5s ease infinite;
#     }
#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(180deg, #818cf8, #a78bfa);
#         border-radius: 5px;
#     }
#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(180deg, #6366f1, #8b5cf6);
#     }

#     </style>
# """, unsafe_allow_html=True)



# # Title
# st.markdown('<h1 class="title-text">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle-text">Advanced Machine Learning Analytics for PITP and Gexton Educational Excellence</p>', unsafe_allow_html=True)

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'feature_columns' not in st.session_state:
#     st.session_state.feature_columns = None

# # Sidebar
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
#     st.title("📊 Navigation")
    
#     if st.session_state.data_loaded:
#         page = st.radio("Select Page:", ["🏠 Dashboard", "📈 Model Training", "🔮 Prediction", "📊 Data Analysis"])
#     else:
#         page = "📁 Data Upload"
    
#     st.markdown("---")
#     st.info("**About:** This system uses Random Forest Classifier to predict student performance based on various academic and demographic factors.")
    
#     if st.session_state.data_loaded:
#         st.success(f"✅ Data Loaded: {len(st.session_state.df)} records")
#         if st.button("🔄 Upload New Dataset"):
#             st.session_state.data_loaded = False
#             st.session_state.df = None
#             st.session_state.feature_columns = None
#             st.rerun()

# # Data Upload Page
# if not st.session_state.data_loaded:
#     st.header("📁 Upload Student Dataset")
    
#     st.markdown("""
#         <div class='upload-section'>
#             <h3>📋 Dataset Requirements</h3>
#             <p>Please ensure your CSV file contains the following columns:</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Required Columns:**
#         - Gender (Male/Female)
#         - Study_Hours (numeric)
#         - Attendance (numeric, 0-100)
#         - Previous_Grade (numeric, 0-100)
#         - Parent_Education (High School/Graduate/Postgraduate)
#         - Internet_Access (Yes/No)
#         - Performance (Pass/Fail)
#         """)
    
#     with col2:
#         st.markdown("""
#         **Example Data Format:**
        
#         ```
#         Gender,Study_Hours,Attendance,Previous_Grade,Parent_Education,Internet_Access,Performance
#         Male,7,85,78,Graduate,Yes,Pass
#         Female,5,70,62,High School,No,Fail
#         Male,8,92,85,Postgraduate,Yes,Pass
#         ```
#         """)
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             required_columns = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                               'Parent_Education', 'Internet_Access', 'Performance']
            
#             missing_columns = [col for col in required_columns if col not in df.columns]
            
#             if missing_columns:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
#             else:
#                 st.success("✅ Dataset uploaded successfully!")
                
#                 st.subheader("📊 Dataset Preview")
#                 st.dataframe(df.head(10), use_container_width=True)
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Records", len(df))
#                 with col2:
#                     st.metric("Features", len(df.columns)-1)
#                 with col3:
#                     pass_count = (df['Performance'] == 'Pass').sum()
#                     st.metric("Pass Count", pass_count)
#                 with col4:
#                     fail_count = (df['Performance'] == 'Fail').sum()
#                     st.metric("Fail Count", fail_count)
                
#                 if st.button("✅ Proceed with Analysis", use_container_width=True):
#                     st.session_state.df = df
#                     st.session_state.data_loaded = True
#                     st.rerun()
        
#         except Exception as e:
#             st.error(f"❌ Error loading file: {str(e)}")
    
#     st.markdown("---")
#     st.info("💡 **Tip:** You can download a sample dataset template to understand the required format.")

# else:
#     df = st.session_state.df.copy()
    
#     # Define required feature columns
#     REQUIRED_FEATURES = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                         'Parent_Education', 'Internet_Access']
    
#     # -------------------------------
#     # 🧹 IMPROVED DATA CLEANING
#     # -------------------------------
#     st.write("---")
#     st.header("🧹 Data Cleaning")
    
#     # First, keep only required columns + Performance
#     available_features = [col for col in REQUIRED_FEATURES if col in df.columns]
#     df = df[available_features + ['Performance']]
    
#     # Standardize the Performance column to only Pass/Fail
#     if 'Performance' in df.columns:
#         st.write("### 📊 Original Performance Values:")
#         st.write(df['Performance'].value_counts())
        
#         # Convert various performance indicators to Pass/Fail
#         df['Performance'] = df['Performance'].astype(str).str.strip().str.lower()
        
#         # Map common variations to Pass/Fail
#         pass_values = ['pass', 'passed', 'success', 'good', 'excellent', 'yes', '1', 'true']
#         fail_values = ['fail', 'failed', 'failure', 'poor', 'bad', 'no', '0', 'false']
        
#         df['Performance'] = df['Performance'].apply(lambda x: 
#             'Pass' if any(val in str(x).lower() for val in pass_values)
#             else 'Fail' if any(val in str(x).lower() for val in fail_values)
#             else x
#         )
        
#         # If still numeric, convert based on threshold
#         try:
#             numeric_perf = pd.to_numeric(df['Performance'], errors='coerce')
#             if numeric_perf.notna().any():
#                 threshold = numeric_perf.median()
#                 df.loc[numeric_perf.notna(), 'Performance'] = numeric_perf[numeric_perf.notna()].apply(
#                     lambda x: 'Pass' if x >= threshold else 'Fail'
#                 )
#                 st.info(f"📊 Converted numeric performance values using threshold: {threshold:.2f}")
#         except:
#             pass
        
#         # Remove any remaining invalid values
#         valid_performance = df['Performance'].isin(['Pass', 'Fail'])
#         if not valid_performance.all():
#             st.warning(f"⚠ Removing {(~valid_performance).sum()} rows with invalid Performance values")
#             df = df[valid_performance]
        
#         st.write("### ✅ Standardized Performance Values:")
#         st.write(df['Performance'].value_counts())

#     # Define which columns should remain categorical
#     categorical_columns = ['Gender', 'Parent_Education', 'Internet_Access', 'Performance']
    
#     # Clean numeric columns
#     numeric_columns = ['Study_Hours', 'Attendance', 'Previous_Grade']
#     for col in numeric_columns:
#         if col in df.columns:
#             # Convert to string first to handle all types
#             df[col] = df[col].astype(str)
            
#             # Try multiple date formats
#             date_formats = ['%d-%b', '%d-%B', '%b-%d', '%B-%d', '%d/%m', '%m/%d']
#             date_converted = False
            
#             for date_format in date_formats:
#                 try:
#                     parsed_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
#                     if parsed_dates.notna().sum() > 0:
#                         df[col] = parsed_dates.dt.day
#                         date_converted = True
#                         st.info(f"📅 Converted date format in column '{col}' to day numbers")
#                         break
#                 except:
#                     continue
            
#             # If not a date, convert to numeric
#             if not date_converted:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Fill missing values with mean for numeric columns
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 mean_val = df[col].mean()
#                 missing_count = df[col].isna().sum()
#                 if missing_count > 0:
#                     df[col] = df[col].fillna(mean_val)
#                     st.warning(f"⚠ Filled {missing_count} missing values in '{col}' with mean: {mean_val:.2f}")
    
#     # Handle categorical columns - remove rows with missing values
#     for col in categorical_columns:
#         if col in df.columns:
#             missing_before = df[col].isna().sum()
#             if missing_before > 0:
#                 df = df.dropna(subset=[col])
#                 st.warning(f"⚠ Removed {missing_before} rows with missing '{col}' values")
    
#     # Final check - ensure all numeric columns are properly converted
#     for col in numeric_columns:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
#     st.success("✅ Data cleaned successfully!")
#     st.write("### 🔍 Cleaned Data Preview:")
#     st.dataframe(df.head())
    
#     # Show data types
#     with st.expander("📋 View Data Types"):
#         st.write(df.dtypes)
    
#     # Encode categorical variables
#     @st.cache_data
#     def encode_data(data):
#         le_gender = LabelEncoder()
#         le_parent = LabelEncoder()
#         le_internet = LabelEncoder()
#         le_performance = LabelEncoder()
        
#         data_encoded = data.copy()
#         data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
#         data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
#         data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
#         data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
        
#         return data_encoded, le_gender, le_parent, le_internet, le_performance

#     try:
#         df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)
        
#         # Store feature columns in session state
#         feature_columns = [col for col in df_encoded.columns if col != 'Performance']
#         st.session_state.feature_columns = feature_columns
        
#         # Train model
#         @st.cache_resource
#         def train_model(X, y, feature_cols):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)
#             return model, accuracy, cm, X_test, y_test, y_pred, feature_cols

#         # Prepare features in correct order
#         X = df_encoded[feature_columns]
#         y = df_encoded['Performance']
        
#         # Ensure X contains only numeric data
#         X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
#         model, accuracy, cm, X_test, y_test, y_pred, trained_feature_cols = train_model(X, y, feature_columns)

#         # Dashboard Page
#         if page == "🏠 Dashboard":
#             st.header("📊 Performance Overview")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Total Students", len(df), delta="Active")
#             with col2:
#                 pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
#                 st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
#             with col3:
#                 st.metric("Model Accuracy", f"{accuracy*100:.1f}%", delta="High")
#             with col4:
#                 avg_study = df['Study_Hours'].mean()
#                 st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
            
#             st.markdown("---")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("📊 Performance Distribution")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 performance_counts = df['Performance'].value_counts()
#                 colors = ['#667eea', '#f87171']
#                 ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', 
#                        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
#                 ax.set_title('Student Performance Distribution', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with col2:
#                 st.subheader("📈 Study Hours vs Performance")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.boxplot(data=df, x='Performance', y='Study_Hours', palette=['#667eea', '#f87171'], ax=ax)
#                 ax.set_title('Study Hours Impact on Performance', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Performance', fontsize=12, weight='bold')
#                 ax.set_ylabel('Study Hours', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("📋 Student Data Overview")
#             st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

#         # Model Training Page
#         elif page == "📈 Model Training":
#             st.header("🤖 Machine Learning Model Training")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("🎯 Model Performance Metrics")
#                 st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
#                 st.metric("Training Samples", len(X) - len(X_test))
#                 st.metric("Testing Samples", len(X_test))
                
#                 st.markdown("---")
#                 st.subheader("📊 Classification Report")
#                 try:
#                     # Get unique classes in predictions
#                     unique_classes = sorted(y_test.unique())
#                     class_names = ['Fail', 'Pass'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
                    
#                     report = classification_report(y_test, y_pred, 
#                                                   labels=unique_classes,
#                                                   target_names=class_names, 
#                                                   output_dict=True,
#                                                   zero_division=0)
#                     report_df = pd.DataFrame(report).transpose()
#                     st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
#                 except Exception as e:
#                     st.error(f"Could not generate classification report: {str(e)}")
#                     st.write("Prediction distribution:")
#                     st.write(pd.Series(y_pred).value_counts())
            
#             with col2:
#                 st.subheader("🔥 Confusion Matrix")
#                 fig, ax = plt.subplots(figsize=(8, 6))
                
#                 # Get unique classes for labels
#                 unique_classes = sorted(y_test.unique())
#                 class_labels = ['Fail', 'Pass'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
                
#                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
#                            xticklabels=class_labels, yticklabels=class_labels, ax=ax)
#                 ax.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Predicted', fontsize=12, weight='bold')
#                 ax.set_ylabel('Actual', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("🌟 Feature Importance")
#             feature_importance = pd.DataFrame({
#                 'Feature': trained_feature_cols,
#                 'Importance': model.feature_importances_
#             }).sort_values('Importance', ascending=False)
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
#             ax.set_title('Feature Importance Analysis', fontsize=14, weight='bold', pad=20)
#             ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
#             ax.set_ylabel('Features', fontsize=12, weight='bold')
#             st.pyplot(fig)

#         # Prediction Page
#         elif page == "🔮 Prediction":
#             st.header("🔮 Student Performance Prediction")
            
#             st.markdown("### Enter Student Information")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 gender = st.selectbox("Gender", df['Gender'].unique())
                
#                 # Handle Study_Hours with same min/max
#                 study_min = int(df['Study_Hours'].min())
#                 study_max = int(df['Study_Hours'].max())
#                 study_mean = int(df['Study_Hours'].mean())
                
#                 if study_min == study_max:
#                     st.info(f"Study Hours: {study_min} (constant in dataset)")
#                     study_hours = study_min
#                 else:
#                     study_hours = st.slider("Study Hours per Day", 
#                                            study_min, 
#                                            study_max, 
#                                            study_mean)
            
#             with col2:
#                 # Handle Attendance with same min/max
#                 att_min = int(df['Attendance'].min())
#                 att_max = int(df['Attendance'].max())
#                 att_mean = int(df['Attendance'].mean())
                
#                 if att_min == att_max:
#                     st.info(f"Attendance: {att_min}% (constant in dataset)")
#                     attendance = att_min
#                 else:
#                     attendance = st.slider("Attendance %", 
#                                           att_min, 
#                                           att_max, 
#                                           att_mean)
                
#                 # Handle Previous_Grade with same min/max
#                 grade_min = int(df['Previous_Grade'].min())
#                 grade_max = int(df['Previous_Grade'].max())
#                 grade_mean = int(df['Previous_Grade'].mean())
                
#                 if grade_min == grade_max:
#                     st.info(f"Previous Grade: {grade_min} (constant in dataset)")
#                     previous_grade = grade_min
#                 else:
#                     previous_grade = st.slider("Previous Grade", 
#                                               grade_min, 
#                                               grade_max, 
#                                               grade_mean)
            
#             with col3:
#                 parent_education = st.selectbox("Parent Education", df['Parent_Education'].unique())
#                 internet_access = st.selectbox("Internet Access", df['Internet_Access'].unique())
            
#             if st.button("🎯 Predict Performance", use_container_width=True):
#                 # Encode input data
#                 gender_encoded = le_gender.transform([gender])[0]
#                 parent_encoded = le_parent.transform([parent_education])[0]
#                 internet_encoded = le_internet.transform([internet_access])[0]
                
#                 # Create input array in the exact same order as training features
#                 input_dict = {
#                     'Gender': gender_encoded,
#                     'Study_Hours': study_hours,
#                     'Attendance': attendance,
#                     'Previous_Grade': previous_grade,
#                     'Parent_Education': parent_encoded,
#                     'Internet_Access': internet_encoded
#                 }
                
#                 # Create input array matching trained feature columns
#                 input_data = np.array([[input_dict[col] for col in trained_feature_cols]])
                
#                 prediction = model.predict(input_data)[0]
#                 prediction_proba = model.predict_proba(input_data)[0]
                
#                 result = le_performance.inverse_transform([prediction])[0]
                
#                 st.markdown("---")
#                 st.markdown("### 🎊 Prediction Results")
                
#                 col1, col2, col3 = st.columns([1, 2, 1])
                
#                 with col2:
#                     if result == 'Pass':
#                         st.success("### ✅ Predicted Performance: PASS")
#                         st.balloons()
#                     else:
#                         st.error("### ❌ Predicted Performance: FAIL")
                    
#                     st.markdown(f"**Confidence:** {max(prediction_proba)*100:.2f}%")
                    
#                     fig, ax = plt.subplots(figsize=(8, 4))
#                     classes = ['Fail', 'Pass']
#                     colors = ['#f87171', '#667eea']
#                     ax.barh(classes, prediction_proba, color=colors)
#                     ax.set_xlabel('Probability', fontsize=12, weight='bold')
#                     ax.set_title('Prediction Confidence', fontsize=14, weight='bold', pad=20)
#                     ax.set_xlim([0, 1])
#                     for i, v in enumerate(prediction_proba):
#                         ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')
#                     st.pyplot(fig)

#         # Data Analysis Page
#         elif page == "📊 Data Analysis":
#             st.header("📊 Advanced Data Analysis")
            
#             tab1, tab2, tab3 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights"])
            
#             with tab1:
#                 st.subheader("Correlation Heatmap")
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 correlation = df_encoded.corr()
#                 sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
#                            center=0, square=True, ax=ax)
#                 ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with tab2:
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Attendance Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Attendance', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Attendance Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
                
#                 with col2:
#                     st.subheader("Previous Grade Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Previous_Grade', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Previous Grade Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
            
#             with tab3:
#                 st.subheader("📌 Key Insights")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     avg_study_pass = df[df['Performance']=='Pass']['Study_Hours'].mean()
#                     avg_study_fail = df[df['Performance']=='Fail']['Study_Hours'].mean()
#                     avg_att_pass = df[df['Performance']=='Pass']['Attendance'].mean()
#                     avg_att_fail = df[df['Performance']=='Fail']['Attendance'].mean()
                    
#                     st.info(f"""
#                     **Academic Performance Insights:**
                    
#                     The average study hours for passing students is {avg_study_pass:.1f} hours, 
#                     compared to {avg_study_fail:.1f} hours for failing students. 
#                     This demonstrates a clear correlation between dedicated study time and academic success.
                    
#                     Attendance rates show a similar pattern, with passing students maintaining an average of 
#                     {avg_att_pass:.1f}% attendance versus {avg_att_fail:.1f}% for those who fail.
#                     """)
                
#                 with col2:
#                     st.success(f"""
#                     **Model Performance Summary:**
                    
#                     The Random Forest Classifier achieves {accuracy*100:.1f}% accuracy in predicting student outcomes. 
#                     The most influential factors in determining performance are study hours, attendance, and previous grades.
                    
#                     Students with consistent attendance above 80% and regular study habits demonstrate 
#                     significantly higher success rates in academic performance.
#                     """)
                
#                 st.markdown("---")
#                 st.subheader("Gender-wise Performance")
#                 gender_performance = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 gender_performance.plot(kind='bar', ax=ax, color=['#f87171', '#667eea'])
#                 ax.set_title('Performance by Gender', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Gender', fontsize=12, weight='bold')
#                 ax.set_ylabel('Percentage', fontsize=12, weight='bold')
#                 ax.legend(title='Performance', title_fontsize=12)
#                 plt.xticks(rotation=0)
#                 st.pyplot(fig)
    
#     except Exception as e:
#         st.error(f"❌ Error during processing: {str(e)}")
#         st.write("Please check your data and try again. Make sure:")
#         st.write("- All numeric columns contain valid numbers")
#         st.write("- Categorical columns contain expected values")
#         st.write("- There are no completely empty columns")
        
#         # Show debug information
#         with st.expander("🔍 Debug Information"):
#             st.write("**Available columns:**", list(df.columns))
#             st.write("**Data types:**")
#             st.write(df.dtypes)
#             st.write("**First few rows:**")
#             st.write(df.head())

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #64748b; padding: 20px; font-family: "Poppins", sans-serif;'>
#         <p style='font-size: 1.1rem; font-weight: 600;'><strong>Student Performance Prediction System</strong></p>
#         <p style='font-size: 0.95rem;'>Powered by Machine Learning | By Rashid Ali </p>
        
#     </div>
#     """, unsafe_allow_html=True)













# # neon

# import streamlit as st
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Student Performance Predictor",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for stunning NEON attractive design
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800;900&family=Montserrat:wght@700;900&family=Orbitron:wght@400;700;900&display=swap');
    
#     /* Main App Background with Animated Neon Gradient */
#     .stApp {
#         background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
#         background-size: 400% 400%;
#         animation: gradientShift 15s ease infinite;
#         font-family: 'Poppins', sans-serif;
#         position: relative;
#     }
    
#     .stApp::before {
#         content: '';
#         position: fixed;
#         top: 0;
#         left: 0;
#         right: 0;
#         bottom: 0;
#         background: 
#             radial-gradient(circle at 20% 50%, rgba(0, 255, 255, 0.1) 0%, transparent 50%),
#             radial-gradient(circle at 80% 80%, rgba(255, 0, 255, 0.1) 0%, transparent 50%),
#             radial-gradient(circle at 40% 20%, rgba(0, 255, 136, 0.1) 0%, transparent 50%);
#         pointer-events: none;
#         animation: neonPulse 8s ease-in-out infinite;
#     }
    
#     @keyframes gradientShift {
#         0% { background-position: 0% 50%; }
#         50% { background-position: 100% 50%; }
#         100% { background-position: 0% 50%; }
#     }
    
#     @keyframes neonPulse {
#         0%, 100% { opacity: 0.3; }
#         50% { opacity: 0.6; }
#     }
    
#     /* Main Content Area with Glassmorphism */
#     .main .block-container {
#         background: rgba(255, 255, 255, 0.95);
#         backdrop-filter: blur(20px);
#         border-radius: 30px;
#         padding: 3rem 2rem;
#         box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
#         border: 1px solid rgba(255, 255, 255, 0.3);
#     }
    
#     /* Sidebar Styling */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
#         border-right: 3px solid rgba(255, 255, 255, 0.2);
#     }
    
#     [data-testid="stSidebar"] * {
#         color: white !important;
#     }
    
#     /* Title with Gradient Text */
#     .title-text {
#         font-family: 'Montserrat', sans-serif;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         font-size: 4rem;
#         font-weight: 900;
#         text-align: center;
#         margin-bottom: 10px;
#         letter-spacing: -2px;
#         text-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
#         animation: titlePulse 3s ease-in-out infinite;
#     }
    
#     @keyframes titlePulse {
#         0%, 100% { transform: scale(1); }
#         50% { transform: scale(1.02); }
#     }
    
#     .subtitle-text {
#         font-family: 'Poppins', sans-serif;
#         color: #475569;
#         font-size: 1.4rem;
#         font-weight: 500;
#         text-align: center;
#         margin-bottom: 40px;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#         opacity: 0.9;
#     }
    
#     /* Enhanced Metrics with Gradient Cards */
#     div[data-testid="stMetric"] {
#         background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
#         padding: 25px;
#         border-radius: 20px;
#         box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
#         border: 2px solid transparent;
#         background-clip: padding-box;
#         transition: all 0.3s ease;
#         position: relative;
#         overflow: hidden;
#     }
    
#     div[data-testid="stMetric"]::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: 0;
#         right: 0;
#         bottom: 0;
#         border-radius: 20px;
#         padding: 2px;
#         background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
#         -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
#         -webkit-mask-composite: xor;
#         mask-composite: exclude;
#         opacity: 0;
#         transition: opacity 0.3s ease;
#     }
    
#     div[data-testid="stMetric"]:hover::before {
#         opacity: 1;
#     }
    
#     div[data-testid="stMetric"]:hover {
#         transform: translateY(-5px) scale(1.02);
#         box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
#     }
    
#     div[data-testid="stMetricValue"] {
#         font-family: 'Montserrat', sans-serif;
#         font-size: 2.5rem;
#         font-weight: 900;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     div[data-testid="stMetricLabel"] {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1.1rem;
#         font-weight: 600;
#         color: #64748b !important;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     /* Premium Button Styling */
#     .stButton>button {
#         font-family: 'Poppins', sans-serif;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
#         background-size: 200% auto;
#         color: white;
#         border: none;
#         padding: 15px 45px;
#         border-radius: 50px;
#         font-weight: 700;
#         font-size: 1.1rem;
#         letter-spacing: 1px;
#         text-transform: uppercase;
#         transition: all 0.4s ease;
#         box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
#         position: relative;
#         overflow: hidden;
#     }
    
#     .stButton>button::before {
#         content: '';
#         position: absolute;
#         top: 0;
#         left: -100%;
#         width: 100%;
#         height: 100%;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
#         transition: left 0.5s;
#     }
    
#     .stButton>button:hover::before {
#         left: 100%;
#     }
    
#     .stButton>button:hover {
#         background-position: right center;
#         transform: translateY(-3px) scale(1.05);
#         box-shadow: 0 12px 35px rgba(102, 126, 234, 0.5);
#     }
    
#     .stButton>button:active {
#         transform: translateY(-1px) scale(1.02);
#     }
    
#     /* Upload Section with Glassmorphism */
#     .upload-section {
#         background: rgba(255, 255, 255, 0.9);
#         backdrop-filter: blur(10px);
#         padding: 35px;
#         border-radius: 25px;
#         box-shadow: 0 15px 50px rgba(102, 126, 234, 0.2);
#         margin: 25px 0;
#         border: 2px solid rgba(102, 126, 234, 0.2);
#         font-family: 'Poppins', sans-serif;
#         transition: all 0.3s ease;
#     }
    
#     .upload-section:hover {
#         box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
#         transform: translateY(-2px);
#     }
    
#     /* Headers with Gradient Underline */
#     h1, h2, h3 {
#         font-family: 'Montserrat', sans-serif;
#         font-weight: 800;
#         color: #1e3a8a;
#         position: relative;
#         padding-bottom: 15px;
#     }
    
#     h1::after, h2::after {
#         content: '';
#         position: absolute;
#         bottom: 0;
#         left: 0;
#         width: 80px;
#         height: 4px;
#         background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
#         border-radius: 2px;
#     }
    
#     /* Enhanced Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 15px;
#         background: rgba(102, 126, 234, 0.05);
#         padding: 10px;
#         border-radius: 15px;
#     }
    
#     .stTabs [data-baseweb="tab-list"] button {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1.1rem;
#         font-weight: 700;
#         border-radius: 12px;
#         padding: 12px 25px;
#         color: #64748b;
#         transition: all 0.3s ease;
#     }
    
#     .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white !important;
#         box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
#     }
    
#     .stTabs [data-baseweb="tab-list"] button:hover {
#         background: rgba(102, 126, 234, 0.1);
#         transform: translateY(-2px);
#     }
    
#     /* DataFrames */
#     .stDataFrame {
#         border-radius: 15px;
#         overflow: hidden;
#         box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
#     }
    
#     /* Select boxes and inputs */
#     .stSelectbox, .stSlider {
#         font-family: 'Poppins', sans-serif;
#     }
    
#     .stSelectbox > div > div {
#         background: white;
#         border: 2px solid rgba(102, 126, 234, 0.3);
#         border-radius: 12px;
#         transition: all 0.3s ease;
#     }
    
#     .stSelectbox > div > div:hover {
#         border-color: #667eea;
#         box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
#     }
    
#     /* File Uploader */
#     [data-testid="stFileUploader"] {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
#         border: 3px dashed rgba(102, 126, 234, 0.4);
#         border-radius: 20px;
#         padding: 30px;
#         transition: all 0.3s ease;
#     }
    
#     [data-testid="stFileUploader"]:hover {
#         border-color: #667eea;
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
#         transform: scale(1.01);
#     }
    
#     /* Info/Warning/Error boxes with gradient borders */
#     .stAlert {
#         border-radius: 15px;
#         border-left: 5px solid;
#         box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
#         font-family: 'Poppins', sans-serif;
#     }
    
#     /* Expander */
#     .streamlit-expanderHeader {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
#         border-radius: 12px;
#         font-family: 'Poppins', sans-serif;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
    
#     .streamlit-expanderHeader:hover {
#         background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
#         transform: translateX(5px);
#     }
    
#     /* Scrollbar */
#     ::-webkit-scrollbar {
#         width: 12px;
#         height: 12px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: rgba(102, 126, 234, 0.05);
#         border-radius: 10px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#         border: 2px solid rgba(255, 255, 255, 0.3);
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
#     }
    
#     /* Animations */
#     @keyframes fadeInUp {
#         from {
#             opacity: 0;
#             transform: translateY(30px);
#         }
#         to {
#             opacity: 1;
#             transform: translateY(0);
#         }
#     }
    
#     .main .block-container > div {
#         animation: fadeInUp 0.6s ease-out;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Title
# st.markdown('<h1 class="title-text">🎓 Student Performance Predictor</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle-text">Advanced Machine Learning Analytics for Educational Excellence</p>', unsafe_allow_html=True)

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'feature_columns' not in st.session_state:
#     st.session_state.feature_columns = None

# # Sidebar
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
#     st.title("📊 Navigation")
    
#     if st.session_state.data_loaded:
#         page = st.radio("Select Page:", ["🏠 Dashboard", "📈 Model Training", "🔮 Prediction", "📊 Data Analysis"])
#     else:
#         page = "📁 Data Upload"
    
#     st.markdown("---")
#     st.info("**About:** This system uses Random Forest Classifier to predict student performance based on various academic and demographic factors.")
    
#     if st.session_state.data_loaded:
#         st.success(f"✅ Data Loaded: {len(st.session_state.df)} records")
#         if st.button("🔄 Upload New Dataset"):
#             st.session_state.data_loaded = False
#             st.session_state.df = None
#             st.session_state.feature_columns = None
#             st.rerun()

# # Data Upload Page
# if not st.session_state.data_loaded:
#     st.header("📁 Upload Student Dataset")
    
#     st.markdown("""
#         <div class='upload-section'>
#             <h3>📋 Dataset Requirements</h3>
#             <p>Please ensure your CSV file contains the following columns:</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         **Required Columns:**
#         - Gender (Male/Female)
#         - Study_Hours (numeric)
#         - Attendance (numeric, 0-100)
#         - Previous_Grade (numeric, 0-100)
#         - Parent_Education (High School/Graduate/Postgraduate)
#         - Internet_Access (Yes/No)
#         - Performance (Pass/Fail)
#         """)
    
#     with col2:
#         st.markdown("""
#         **Example Data Format:**
        
#         ```
#         Gender,Study_Hours,Attendance,Previous_Grade,Parent_Education,Internet_Access,Performance
#         Male,7,85,78,Graduate,Yes,Pass
#         Female,5,70,62,High School,No,Fail
#         Male,8,92,85,Postgraduate,Yes,Pass
#         ```
#         """)
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             required_columns = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                               'Parent_Education', 'Internet_Access', 'Performance']
            
#             missing_columns = [col for col in required_columns if col not in df.columns]
            
#             if missing_columns:
#                 st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
#             else:
#                 st.success("✅ Dataset uploaded successfully!")
                
#                 st.subheader("📊 Dataset Preview")
#                 st.dataframe(df.head(10), use_container_width=True)
                
#                 col1, col2, col3, col4 = st.columns(4)
#                 with col1:
#                     st.metric("Total Records", len(df))
#                 with col2:
#                     st.metric("Features", len(df.columns)-1)
#                 with col3:
#                     pass_count = (df['Performance'] == 'Pass').sum()
#                     st.metric("Pass Count", pass_count)
#                 with col4:
#                     fail_count = (df['Performance'] == 'Fail').sum()
#                     st.metric("Fail Count", fail_count)
                
#                 if st.button("✅ Proceed with Analysis", use_container_width=True):
#                     st.session_state.df = df
#                     st.session_state.data_loaded = True
#                     st.rerun()
        
#         except Exception as e:
#             st.error(f"❌ Error loading file: {str(e)}")
    
#     st.markdown("---")
#     st.info("💡 **Tip:** You can download a sample dataset template to understand the required format.")

# else:
#     df = st.session_state.df.copy()
    
#     # Define required feature columns
#     REQUIRED_FEATURES = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                         'Parent_Education', 'Internet_Access']
    
#     # -------------------------------
#     # 🧹 IMPROVED DATA CLEANING
#     # -------------------------------
#     st.write("---")
#     st.header("🧹 Data Cleaning")
    
#     # First, keep only required columns + Performance
#     available_features = [col for col in REQUIRED_FEATURES if col in df.columns]
#     df = df[available_features + ['Performance']]
    
#     # Standardize the Performance column to only Pass/Fail
#     if 'Performance' in df.columns:
#         st.write("### 📊 Original Performance Values:")
#         st.write(df['Performance'].value_counts())
        
#         # Convert various performance indicators to Pass/Fail
#         df['Performance'] = df['Performance'].astype(str).str.strip().str.lower()
        
#         # Map common variations to Pass/Fail
#         pass_values = ['pass', 'passed', 'success', 'good', 'excellent', 'yes', '1', 'true']
#         fail_values = ['fail', 'failed', 'failure', 'poor', 'bad', 'no', '0', 'false']
        
#         df['Performance'] = df['Performance'].apply(lambda x: 
#             'Pass' if any(val in str(x).lower() for val in pass_values)
#             else 'Fail' if any(val in str(x).lower() for val in fail_values)
#             else x
#         )
        
#         # If still numeric, convert based on threshold
#         try:
#             numeric_perf = pd.to_numeric(df['Performance'], errors='coerce')
#             if numeric_perf.notna().any():
#                 threshold = numeric_perf.median()
#                 df.loc[numeric_perf.notna(), 'Performance'] = numeric_perf[numeric_perf.notna()].apply(
#                     lambda x: 'Pass' if x >= threshold else 'Fail'
#                 )
#                 st.info(f"📊 Converted numeric performance values using threshold: {threshold:.2f}")
#         except:
#             pass
        
#         # Remove any remaining invalid values
#         valid_performance = df['Performance'].isin(['Pass', 'Fail'])
#         if not valid_performance.all():
#             st.warning(f"⚠ Removing {(~valid_performance).sum()} rows with invalid Performance values")
#             df = df[valid_performance]
        
#         st.write("### ✅ Standardized Performance Values:")
#         st.write(df['Performance'].value_counts())

#     # Define which columns should remain categorical
#     categorical_columns = ['Gender', 'Parent_Education', 'Internet_Access', 'Performance']
    
#     # Clean numeric columns
#     numeric_columns = ['Study_Hours', 'Attendance', 'Previous_Grade']
#     for col in numeric_columns:
#         if col in df.columns:
#             # Convert to string first to handle all types
#             df[col] = df[col].astype(str)
            
#             # Try multiple date formats
#             date_formats = ['%d-%b', '%d-%B', '%b-%d', '%B-%d', '%d/%m', '%m/%d']
#             date_converted = False
            
#             for date_format in date_formats:
#                 try:
#                     parsed_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
#                     if parsed_dates.notna().sum() > 0:
#                         df[col] = parsed_dates.dt.day
#                         date_converted = True
#                         st.info(f"📅 Converted date format in column '{col}' to day numbers")
#                         break
#                 except:
#                     continue
            
#             # If not a date, convert to numeric
#             if not date_converted:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Fill missing values with mean for numeric columns
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 mean_val = df[col].mean()
#                 missing_count = df[col].isna().sum()
#                 if missing_count > 0:
#                     df[col] = df[col].fillna(mean_val)
#                     st.warning(f"⚠ Filled {missing_count} missing values in '{col}' with mean: {mean_val:.2f}")
    
#     # Handle categorical columns - remove rows with missing values
#     for col in categorical_columns:
#         if col in df.columns:
#             missing_before = df[col].isna().sum()
#             if missing_before > 0:
#                 df = df.dropna(subset=[col])
#                 st.warning(f"⚠ Removed {missing_before} rows with missing '{col}' values")
    
#     # Final check - ensure all numeric columns are properly converted
#     for col in numeric_columns:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
#     st.success("✅ Data cleaned successfully!")
#     st.write("### 🔍 Cleaned Data Preview:")
#     st.dataframe(df.head())
    
#     # Show data types
#     with st.expander("📋 View Data Types"):
#         st.write(df.dtypes)
    
#     # Encode categorical variables
#     @st.cache_data
#     def encode_data(data):
#         le_gender = LabelEncoder()
#         le_parent = LabelEncoder()
#         le_internet = LabelEncoder()
#         le_performance = LabelEncoder()
        
#         data_encoded = data.copy()
#         data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
#         data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
#         data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
#         data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
        
#         return data_encoded, le_gender, le_parent, le_internet, le_performance

#     try:
#         df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)
        
#         # Store feature columns in session state
#         feature_columns = [col for col in df_encoded.columns if col != 'Performance']
#         st.session_state.feature_columns = feature_columns
        
#         # Train model
#         @st.cache_resource
#         def train_model(X, y, feature_cols):
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             accuracy = accuracy_score(y_test, y_pred)
#             cm = confusion_matrix(y_test, y_pred)
#             return model, accuracy, cm, X_test, y_test, y_pred, feature_cols

#         # Prepare features in correct order
#         X = df_encoded[feature_columns]
#         y = df_encoded['Performance']
        
#         # Ensure X contains only numeric data
#         X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
#         model, accuracy, cm, X_test, y_test, y_pred, trained_feature_cols = train_model(X, y, feature_columns)

#         # Dashboard Page
#         if page == "🏠 Dashboard":
#             st.header("📊 Performance Overview")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Total Students", len(df), delta="Active")
#             with col2:
#                 pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
#                 st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
#             with col3:
#                 st.metric("Model Accuracy", f"{accuracy*100:.1f}%", delta="High")
#             with col4:
#                 avg_study = df['Study_Hours'].mean()
#                 st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
            
#             st.markdown("---")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("📊 Performance Distribution")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 performance_counts = df['Performance'].value_counts()
#                 colors = ['#667eea', '#f87171']
#                 ax.pie(performance_counts, labels=performance_counts.index, autopct='%1.1f%%', 
#                        colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
#                 ax.set_title('Student Performance Distribution', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with col2:
#                 st.subheader("📈 Study Hours vs Performance")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.boxplot(data=df, x='Performance', y='Study_Hours', palette=['#667eea', '#f87171'], ax=ax)
#                 ax.set_title('Study Hours Impact on Performance', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Performance', fontsize=12, weight='bold')
#                 ax.set_ylabel('Study Hours', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("📋 Student Data Overview")
#             st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

#         # Model Training Page
#         elif page == "📈 Model Training":
#             st.header("🤖 Machine Learning Model Training")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("🎯 Model Performance Metrics")
#                 st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
#                 st.metric("Training Samples", len(X) - len(X_test))
#                 st.metric("Testing Samples", len(X_test))
                
#                 st.markdown("---")
#                 st.subheader("📊 Classification Report")
#                 try:
#                     # Get unique classes in predictions
#                     unique_classes = sorted(y_test.unique())
#                     class_names = ['Fail', 'Pass'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
                    
#                     report = classification_report(y_test, y_pred, 
#                                                   labels=unique_classes,
#                                                   target_names=class_names, 
#                                                   output_dict=True,
#                                                   zero_division=0)
#                     report_df = pd.DataFrame(report).transpose()
#                     st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
#                 except Exception as e:
#                     st.error(f"Could not generate classification report: {str(e)}")
#                     st.write("Prediction distribution:")
#                     st.write(pd.Series(y_pred).value_counts())
            
#             with col2:
#                 st.subheader("🔥 Confusion Matrix")
#                 fig, ax = plt.subplots(figsize=(8, 6))
                
#                 # Get unique classes for labels
#                 unique_classes = sorted(y_test.unique())
#                 class_labels = ['Fail', 'Pass'] if len(unique_classes) == 2 else [f'Class {i}' for i in unique_classes]
                
#                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
#                            xticklabels=class_labels, yticklabels=class_labels, ax=ax)
#                 ax.set_title('Confusion Matrix', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Predicted', fontsize=12, weight='bold')
#                 ax.set_ylabel('Actual', fontsize=12, weight='bold')
#                 st.pyplot(fig)
            
#             st.markdown("---")
            
#             st.subheader("🌟 Feature Importance")
#             feature_importance = pd.DataFrame({
#                 'Feature': trained_feature_cols,
#                 'Importance': model.feature_importances_
#             }).sort_values('Importance', ascending=False)
            
#             fig, ax = plt.subplots(figsize=(10, 6))
#             sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis', ax=ax)
#             ax.set_title('Feature Importance Analysis', fontsize=14, weight='bold', pad=20)
#             ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
#             ax.set_ylabel('Features', fontsize=12, weight='bold')
#             st.pyplot(fig)

#         # Prediction Page
#         elif page == "🔮 Prediction":
#             st.header("🔮 Student Performance Prediction")
            
#             st.markdown("### Enter Student Information")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 gender = st.selectbox("Gender", df['Gender'].unique())
                
#                 # Handle Study_Hours with same min/max
#                 study_min = int(df['Study_Hours'].min())
#                 study_max = int(df['Study_Hours'].max())
#                 study_mean = int(df['Study_Hours'].mean())
                
#                 if study_min == study_max:
#                     st.info(f"Study Hours: {study_min} (constant in dataset)")
#                     study_hours = study_min
#                 else:
#                     study_hours = st.slider("Study Hours per Day", 
#                                            study_min, 
#                                            study_max, 
#                                            study_mean)
            
#             with col2:
#                 # Handle Attendance with same min/max
#                 att_min = int(df['Attendance'].min())
#                 att_max = int(df['Attendance'].max())
#                 att_mean = int(df['Attendance'].mean())
                
#                 if att_min == att_max:
#                     st.info(f"Attendance: {att_min}% (constant in dataset)")
#                     attendance = att_min
#                 else:
#                     attendance = st.slider("Attendance %", 
#                                           att_min, 
#                                           att_max, 
#                                           att_mean)
                
#                 # Handle Previous_Grade with same min/max
#                 grade_min = int(df['Previous_Grade'].min())
#                 grade_max = int(df['Previous_Grade'].max())
#                 grade_mean = int(df['Previous_Grade'].mean())
                
#                 if grade_min == grade_max:
#                     st.info(f"Previous Grade: {grade_min} (constant in dataset)")
#                     previous_grade = grade_min
#                 else:
#                     previous_grade = st.slider("Previous Grade", 
#                                               grade_min, 
#                                               grade_max, 
#                                               grade_mean)
            
#             with col3:
#                 parent_education = st.selectbox("Parent Education", df['Parent_Education'].unique())
#                 internet_access = st.selectbox("Internet Access", df['Internet_Access'].unique())
            
#             if st.button("🎯 Predict Performance", use_container_width=True):
#                 # Encode input data
#                 gender_encoded = le_gender.transform([gender])[0]
#                 parent_encoded = le_parent.transform([parent_education])[0]
#                 internet_encoded = le_internet.transform([internet_access])[0]
                
#                 # Create input array in the exact same order as training features
#                 input_dict = {
#                     'Gender': gender_encoded,
#                     'Study_Hours': study_hours,
#                     'Attendance': attendance,
#                     'Previous_Grade': previous_grade,
#                     'Parent_Education': parent_encoded,
#                     'Internet_Access': internet_encoded
#                 }
                
#                 # Create input array matching trained feature columns
#                 input_data = np.array([[input_dict[col] for col in trained_feature_cols]])
                
#                 prediction = model.predict(input_data)[0]
#                 prediction_proba = model.predict_proba(input_data)[0]
                
#                 result = le_performance.inverse_transform([prediction])[0]
                
#                 st.markdown("---")
#                 st.markdown("### 🎊 Prediction Results")
                
#                 col1, col2, col3 = st.columns([1, 2, 1])
                
#                 with col2:
#                     if result == 'Pass':
#                         st.success("### ✅ Predicted Performance: PASS")
#                         st.balloons()
#                     else:
#                         st.error("### ❌ Predicted Performance: FAIL")
                    
#                     st.markdown(f"**Confidence:** {max(prediction_proba)*100:.2f}%")
                    
#                     fig, ax = plt.subplots(figsize=(8, 4))
#                     classes = ['Fail', 'Pass']
#                     colors = ['#f87171', '#667eea']
#                     ax.barh(classes, prediction_proba, color=colors)
#                     ax.set_xlabel('Probability', fontsize=12, weight='bold')
#                     ax.set_title('Prediction Confidence', fontsize=14, weight='bold', pad=20)
#                     ax.set_xlim([0, 1])
#                     for i, v in enumerate(prediction_proba):
#                         ax.text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')
#                     st.pyplot(fig)

#         # Data Analysis Page
#         elif page == "📊 Data Analysis":
#             st.header("📊 Advanced Data Analysis")
            
#             tab1, tab2, tab3 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights"])
            
#             with tab1:
#                 st.subheader("Correlation Heatmap")
#                 fig, ax = plt.subplots(figsize=(10, 8))
#                 correlation = df_encoded.corr()
#                 sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
#                            center=0, square=True, ax=ax)
#                 ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold', pad=20)
#                 st.pyplot(fig)
            
#             with tab2:
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Attendance Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Attendance', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Attendance Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
                
#                 with col2:
#                     st.subheader("Previous Grade Distribution")
#                     fig, ax = plt.subplots(figsize=(8, 6))
#                     sns.histplot(data=df, x='Previous_Grade', hue='Performance', 
#                                 palette=['#f87171', '#667eea'], kde=True, ax=ax)
#                     ax.set_title('Previous Grade Impact', fontsize=14, weight='bold', pad=20)
#                     st.pyplot(fig)
            
#             with tab3:
#                 st.subheader("📌 Key Insights")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     avg_study_pass = df[df['Performance']=='Pass']['Study_Hours'].mean()
#                     avg_study_fail = df[df['Performance']=='Fail']['Study_Hours'].mean()
#                     avg_att_pass = df[df['Performance']=='Pass']['Attendance'].mean()
#                     avg_att_fail = df[df['Performance']=='Fail']['Attendance'].mean()
                    
#                     st.info(f"""
#                     **Academic Performance Insights:**
                    
#                     The average study hours for passing students is {avg_study_pass:.1f} hours, 
#                     compared to {avg_study_fail:.1f} hours for failing students. 
#                     This demonstrates a clear correlation between dedicated study time and academic success.
                    
#                     Attendance rates show a similar pattern, with passing students maintaining an average of 
#                     {avg_att_pass:.1f}% attendance versus {avg_att_fail:.1f}% for those who fail.
#                     """)
                
#                 with col2:
#                     st.success(f"""
#                     **Model Performance Summary:**
                    
#                     The Random Forest Classifier achieves {accuracy*100:.1f}% accuracy in predicting student outcomes. 
#                     The most influential factors in determining performance are study hours, attendance, and previous grades.
                    
#                     Students with consistent attendance above 80% and regular study habits demonstrate 
#                     significantly higher success rates in academic performance.
#                     """)
                
#                 st.markdown("---")
#                 st.subheader("Gender-wise Performance")
#                 gender_performance = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
#                 fig, ax = plt.subplots(figsize=(10, 6))
#                 gender_performance.plot(kind='bar', ax=ax, color=['#f87171', '#667eea'])
#                 ax.set_title('Performance by Gender', fontsize=14, weight='bold', pad=20)
#                 ax.set_xlabel('Gender', fontsize=12, weight='bold')
#                 ax.set_ylabel('Percentage', fontsize=12, weight='bold')
#                 ax.legend(title='Performance', title_fontsize=12)
#                 plt.xticks(rotation=0)
#                 st.pyplot(fig)
    
#     except Exception as e:
#         st.error(f"❌ Error during processing: {str(e)}")
#         st.write("Please check your data and try again. Make sure:")
#         st.write("- All numeric columns contain valid numbers")
#         st.write("- Categorical columns contain expected values")
#         st.write("- There are no completely empty columns")
        
#         # Show debug information
#         with st.expander("🔍 Debug Information"):
#             st.write("**Available columns:**", list(df.columns))
#             st.write("**Data types:**")
#             st.write(df.dtypes)
#             st.write("**First few rows:**")
#             st.write(df.head())

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #64748b; padding: 20px; font-family: "Poppins", sans-serif;'>
#         <p style='font-size: 1.1rem; font-weight: 600;'><strong>Student Performance Prediction System</strong></p>
#         <p style='font-size: 0.95rem;'>Powered by Machine Learning | Built with Streamlit</p>
#         <p style='font-size: 0.85rem; color: #94a3b8; margin-top: 10px;'>© 2024 Educational Analytics Platform</p>
#     </div>
#     """, unsafe_allow_html=True)

















# # # Most refined one and most correct with improvements incorporated above.

# import streamlit as st
# from datetime import datetime
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
# from sklearn.feature_selection import SelectKBest, f_classif
# from sklearn.cluster import KMeans
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import joblib
# import io
# import gc
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Advanced Student Performance Predictor",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS with dark mode support
# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Montserrat:wght@700;900&display=swap');

#     .main {
#         background: linear-gradient(135deg, #74ABE2 0%, #5563DE 50%, #A683E3 100%);
#         font-family: 'Poppins', sans-serif;
#         color: #1e293b;
#     }

#     .stApp {
#         background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
#         font-family: 'Poppins', sans-serif;
#     }

#     .metric-card {
#         background: linear-gradient(145deg, #ffffff 0%, #f3f4f6 100%);
#         padding: 24px;
#         border-radius: 16px;
#         box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
#         margin: 15px 0;
#         transition: all 0.4s ease-in-out;
#         border-image: linear-gradient(90deg, #3B82F6, #9333EA) 1;
#     }
    
#     .metric-card:hover {
#         transform: translateY(-4px) scale(1.02);
#         box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
#     }

#     .title-text {
#         font-family: 'Montserrat', sans-serif;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         font-size: 4rem;
#         font-weight: 900;
#         text-align: center;
#         margin-bottom: 10px;
#         letter-spacing: -2px;
#         text-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
#         animation: titlePulse 3s ease-in-out infinite;
#     }
    
#     @keyframes titlePulse {
#         0%, 100% { transform: scale(1); }
#         50% { transform: scale(1.02); }
#     }

#     .subtitle-text {
#         font-family: 'Poppins', sans-serif;
#         color: #475569;
#         font-size: 1.25rem;
#         font-weight: 400;
#         text-align: center;
#         margin-bottom: 35px;
#         letter-spacing: 0.6px;
#     }

#     div[data-testid="stMetricValue"] {
#         font-family: 'Montserrat', sans-serif;
#         font-size: 2.4rem;
#         font-weight: 800;
#         color: #4338ca;
#         text-shadow: 1px 1px 8px rgba(99, 102, 241, 0.25);
#     }
    
#     div[data-testid="stMetricLabel"] {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1rem;
#         font-weight: 600;
#         color: #334155;
#     }

#     .stButton>button {
#         font-family: 'Poppins', sans-serif;
#         background: linear-gradient(90deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
#         color: white;
#         border: none;
#         padding: 14px 38px;
#         border-radius: 30px;
#         font-weight: 700;
#         font-size: 1.1rem;
#         letter-spacing: 0.6px;
#         transition: all 0.3s ease;
#         background-size: 200% 200%;
#         animation: gradientShift 5s ease infinite;
#     }
    
#     @keyframes gradientShift {
#         0% { background-position: 0% 50%; }
#         50% { background-position: 100% 50%; }
#         100% { background-position: 0% 50%; }
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-3px) scale(1.03);
#         box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
#     }

#     .upload-section {
#         background: linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%);
#         padding: 32px;
#         border-radius: 18px;
#         box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
#         margin: 25px 0;
#         border-left: 6px solid #6366f1;
#         transition: all 0.4s ease-in-out;
#     }
    
#     .upload-section:hover {
#         transform: translateY(-3px) scale(1.01);
#         box-shadow: 0 12px 30px rgba(99, 102, 241, 0.2);
#     }

#     .stTabs [data-baseweb="tab-list"] button {
#         font-family: 'Poppins', sans-serif;
#         font-size: 1.1rem;
#         font-weight: 600;
#         color: #374151;
#         background: linear-gradient(90deg, #f9fafb 0%, #e2e8f0 100%);
#         border-radius: 12px;
#         margin-right: 6px;
#         transition: all 0.3s ease;
#     }
    
#     .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
#         background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
#         color: white;
#         box-shadow: 0 6px 12px rgba(99, 102, 241, 0.25);
#     }

#     h1, h2, h3 {
#         font-family: 'Montserrat', sans-serif;
#         font-weight: 700;
#         color: #1e293b;
#     }

#     ::-webkit-scrollbar {
#         width: 10px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(180deg, #818cf8, #a78bfa);
#         border-radius: 5px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(180deg, #6366f1, #8b5cf6);
#     }

#     .linkedin-badge {
#         position: fixed;
#         bottom: 20px;
#         right: 20px;
#         background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%);
#         padding: 12px 20px;
#         border-radius: 50px;
#         box-shadow: 0 4px 15px rgba(0, 119, 181, 0.3);
#         transition: all 0.3s ease;
#         z-index: 1000;
#     }
    
#     .linkedin-badge:hover {
#         transform: translateY(-3px) scale(1.05);
#         box-shadow: 0 6px 20px rgba(0, 119, 181, 0.5);
#     }
    
#     .linkedin-badge a {
#         color: white;
#         text-decoration: none;
#         font-weight: 600;
#         font-size: 0.9rem;
#         display: flex;
#         align-items: center;
#         gap: 8px;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # LinkedIn Badge
# st.markdown("""
#     <div class="linkedin-badge">
#         <a href="https://www.linkedin.com/in/rashid-ali-619671357/" target="_blank">
#             <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
#                 <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
#             </svg>
#             Connect on LinkedIn
#         </a>
#     </div>
# """, unsafe_allow_html=True)

# # Title
# st.markdown('<h1 class="title-text">🎓 Advanced Student Performance Predictor</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle-text">AI-Powered Analytics with Multiple ML Models, Hyperparameter Tuning & Explainability</p>', unsafe_allow_html=True)

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None
# if 'feature_columns' not in st.session_state:
#     st.session_state.feature_columns = None
# if 'dark_mode' not in st.session_state:
#     st.session_state.dark_mode = False
# if 'trained_models' not in st.session_state:
#     st.session_state.trained_models = {}

# # Sidebar
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
#     st.title("📊 Navigation")
    
#     # Dark Mode Toggle
#     dark_mode = st.toggle("🌗 Dark Mode", value=st.session_state.dark_mode)
#     if dark_mode != st.session_state.dark_mode:
#         st.session_state.dark_mode = dark_mode
#         st.rerun()
    
#     if st.session_state.data_loaded:
#         page = st.radio("Select Page:", [
#             "🏠 Dashboard", 
#             "📈 Model Training", 
#             "🔮 Prediction", 
#             "📊 Data Analysis",
#             "🧠 AI Explainability",
#             "👥 Student Clustering",
#             "⚙️ Advanced Settings"
#         ])
#     else:
#         page = "📁 Data Upload"
    
#     st.markdown("---")
#     st.info("**About:** Advanced ML system with multiple algorithms, hyperparameter tuning, and SHAP explainability for predicting student performance.")
    
#     if st.session_state.data_loaded:
#         st.success(f"✅ Data Loaded: {len(st.session_state.df)} records")
        
#         # Model Management
#         if st.session_state.trained_models:
#             st.markdown("### 🤖 Trained Models")
#             for model_name in st.session_state.trained_models.keys():
#                 st.write(f"✓ {model_name}")
        
#         if st.button("🔄 Upload New Dataset"):
#             st.session_state.data_loaded = False
#             st.session_state.df = None
#             st.session_state.feature_columns = None
#             st.session_state.trained_models = {}
#             gc.collect()
#             st.rerun()
    
#     st.markdown("---")
#     st.markdown("**Created by:** [Rashid Ali](https://www.linkedin.com/in/rashid-ali-619671357/)")

# # Sample dataset generator
# def generate_sample_dataset():
#     np.random.seed(42)
#     n_samples = 100
    
#     data = {
#         'Gender': np.random.choice(['Male', 'Female'], n_samples),
#         'Study_Hours': np.random.randint(1, 12, n_samples),
#         'Attendance': np.random.randint(50, 100, n_samples),
#         'Previous_Grade': np.random.randint(40, 100, n_samples),
#         'Parent_Education': np.random.choice(['High School', 'Graduate', 'Postgraduate'], n_samples),
#         'Internet_Access': np.random.choice(['Yes', 'No'], n_samples),
#     }
    
#     df = pd.DataFrame(data)
#     df['Performance'] = df.apply(
#         lambda row: 'Pass' if (row['Study_Hours'] * 0.3 + row['Attendance'] * 0.4 + row['Previous_Grade'] * 0.3) > 60 else 'Fail',
#         axis=1
#     )
    
#     return df

# # Data Upload Page
# if not st.session_state.data_loaded:
#     st.header("📁 Upload Student Dataset")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("""
#             <div class='upload-section'>
#                 <h3>📋 Dataset Requirements</h3>
#                 <p>Please ensure your CSV file contains the following columns:</p>
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("""
#         **Required Columns:**
#         - Gender (Male/Female)
#         - Study_Hours (numeric)
#         - Attendance (numeric, 0-100)
#         - Previous_Grade (numeric, 0-100)
#         - Parent_Education (High School/Graduate/Postgraduate)
#         - Internet_Access (Yes/No)
#         - Performance (Pass/Fail)
#         """)
    
#     with col2:
#         st.markdown("### 📥 Download Sample Dataset")
#         sample_df = generate_sample_dataset()
#         csv = sample_df.to_csv(index=False).encode('utf-8')
#         st.download_button(
#             label="⬇️ Download Sample CSV",
#             data=csv,
#             file_name="sample_student_data.csv",
#             mime="text/csv",
#             use_container_width=True
#         )
    
#     uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
#     if uploaded_file is not None:
#         try:
#             # File size validation
#             file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
#             if file_size > 50:
#                 st.error("❌ File size exceeds 50MB limit. Please upload a smaller file.")
#             else:
#                 df = pd.read_csv(uploaded_file)
                
#                 required_columns = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                                   'Parent_Education', 'Internet_Access', 'Performance']
                
#                 missing_columns = [col for col in required_columns if col not in df.columns]
                
#                 if missing_columns:
#                     st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    
#                     # Auto column mapping suggestion
#                     st.markdown("### 🔄 Column Mapping Assistant")
#                     st.info("Try mapping your columns to the required format:")
                    
#                     mapping = {}
#                     for req_col in missing_columns:
#                         available_cols = [col for col in df.columns if col not in mapping.values()]
#                         mapping[req_col] = st.selectbox(f"Map '{req_col}' to:", [''] + available_cols, key=req_col)
                    
#                     if st.button("Apply Mapping"):
#                         for req_col, user_col in mapping.items():
#                             if user_col:
#                                 df[req_col] = df[user_col]
#                         st.success("✅ Columns mapped successfully!")
#                         st.rerun()
#                 else:
#                     st.success("✅ Dataset uploaded successfully!")
#                     st.toast("🎉 Dataset validated successfully!")
                    
#                     st.subheader("📊 Dataset Preview")
#                     st.dataframe(df.head(10), use_container_width=True)
                    
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric("Total Records", len(df))
#                     with col2:
#                         st.metric("Features", len(df.columns)-1)
#                     with col3:
#                         pass_count = (df['Performance'] == 'Pass').sum()
#                         st.metric("Pass Count", pass_count)
#                     with col4:
#                         fail_count = (df['Performance'] == 'Fail').sum()
#                         st.metric("Fail Count", fail_count)
                    
#                     if st.button("✅ Proceed with Analysis", use_container_width=True):
#                         st.session_state.df = df
#                         st.session_state.data_loaded = True
#                         st.rerun()
        
#         except Exception as e:
#             st.error(f"❌ Error loading file: {str(e)}")

# else:
#     df = st.session_state.df.copy()
    
#     REQUIRED_FEATURES = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
#                         'Parent_Education', 'Internet_Access']
    
#     # Data Cleaning with progress bar
#     if 'data_cleaned' not in st.session_state:
#         st.write("---")
#         st.header("🧹 Data Cleaning in Progress...")
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Step 1: Column selection
#         status_text.text("Step 1/5: Selecting required columns...")
#         progress_bar.progress(20)
#         available_features = [col for col in REQUIRED_FEATURES if col in df.columns]
#         df = df[available_features + ['Performance']]
        
#         # Step 2: Performance standardization
#         status_text.text("Step 2/5: Standardizing performance values...")
#         progress_bar.progress(40)
#         if 'Performance' in df.columns:
#             df['Performance'] = df['Performance'].astype(str).str.strip().str.lower()
#             pass_values = ['pass', 'passed', 'success', 'good', 'excellent', 'yes', '1', 'true']
#             fail_values = ['fail', 'failed', 'failure', 'poor', 'bad', 'no', '0', 'false']
            
#             df['Performance'] = df['Performance'].apply(lambda x: 
#                 'Pass' if any(val in str(x).lower() for val in pass_values)
#                 else 'Fail' if any(val in str(x).lower() for val in fail_values)
#                 else x
#             )
            
#             try:
#                 numeric_perf = pd.to_numeric(df['Performance'], errors='coerce')
#                 if numeric_perf.notna().any():
#                     threshold = numeric_perf.median()
#                     df.loc[numeric_perf.notna(), 'Performance'] = numeric_perf[numeric_perf.notna()].apply(
#                         lambda x: 'Pass' if x >= threshold else 'Fail'
#                     )
#             except:
#                 pass
            
#             valid_performance = df['Performance'].isin(['Pass', 'Fail'])
#             if not valid_performance.all():
#                 df = df[valid_performance]
        
#         # Step 3: Numeric columns cleaning
#         status_text.text("Step 3/5: Cleaning numeric columns...")
#         progress_bar.progress(60)
#         categorical_columns = ['Gender', 'Parent_Education', 'Internet_Access', 'Performance']
#         numeric_columns = ['Study_Hours', 'Attendance', 'Previous_Grade']
        
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = df[col].astype(str)
#                 date_formats = ['%d-%b', '%d-%B', '%b-%d', '%B-%d', '%d/%m', '%m/%d']
#                 date_converted = False
                
#                 for date_format in date_formats:
#                     try:
#                         parsed_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
#                         if parsed_dates.notna().sum() > 0:
#                             df[col] = parsed_dates.dt.day
#                             date_converted = True
#                             break
#                     except:
#                         continue
                
#                 if not date_converted:
#                     df[col] = pd.to_numeric(df[col], errors='coerce')
                
#                 if pd.api.types.is_numeric_dtype(df[col]):
#                     mean_val = df[col].mean()
#                     df[col] = df[col].fillna(mean_val)
        
#         # Step 4: Categorical columns handling
#         status_text.text("Step 4/5: Processing categorical columns...")
#         progress_bar.progress(80)
#         for col in categorical_columns:
#             if col in df.columns:
#                 df = df.dropna(subset=[col])
        
#         # Step 5: Final validation
#         status_text.text("Step 5/5: Final validation...")
#         progress_bar.progress(100)
#         for col in numeric_columns:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
        
#         st.session_state.data_cleaned = True
#         st.success("✅ Data cleaned successfully!")
#         st.toast("✅ Data cleaning completed!")
#         progress_bar.empty()
#         status_text.empty()
    
#     # Encode categorical variables
#     @st.cache_data
#     def encode_data(data):
#         le_gender = LabelEncoder()
#         le_parent = LabelEncoder()
#         le_internet = LabelEncoder()
#         le_performance = LabelEncoder()
        
#         data_encoded = data.copy()
#         data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
#         data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
#         data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
#         data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
        
#         return data_encoded, le_gender, le_parent, le_internet, le_performance

#     try:
#         df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)
        
#         feature_columns = [col for col in df_encoded.columns if col != 'Performance']
#         st.session_state.feature_columns = feature_columns
        
#         X = df_encoded[feature_columns]
#         y = df_encoded['Performance']
#         X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

#         # Dashboard Page
#         if page == "🏠 Dashboard":
#             st.header("📊 Performance Overview Dashboard")
            
#             col1, col2, col3, col4 = st.columns(4)
            
#             with col1:
#                 st.metric("Total Students", len(df), delta="Active")
#             with col2:
#                 pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
#                 st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
#             with col3:
#                 if st.session_state.trained_models:
#                     best_model = max(st.session_state.trained_models.items(), 
#                                    key=lambda x: x[1]['accuracy'])
#                     st.metric("Best Model Accuracy", f"{best_model[1]['accuracy']*100:.1f}%", 
#                             delta=best_model[0])
#                 else:
#                     st.metric("Models Trained", "0", delta="Train models")
#             with col4:
#                 avg_study = df['Study_Hours'].mean()
#                 st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
            
#             # Download cleaned data
#             st.markdown("---")
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 csv = df.to_csv(index=False).encode('utf-8')
#                 st.download_button(
#                     label="⬇️ Download Cleaned Data",
#                     data=csv,
#                     file_name="cleaned_student_data.csv",
#                     mime="text/csv",
#                     use_container_width=True
#                 )
#             with col2:
#                 encoded_csv = df_encoded.to_csv(index=False).encode('utf-8')
#                 st.download_button(
#                     label="⬇️ Download Encoded Data",
#                     data=encoded_csv,
#                     file_name="encoded_student_data.csv",
#                     mime="text/csv",
#                     use_container_width=True
#                 )
            
#             st.markdown("---")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("📊 Performance Distribution")
#                 performance_counts = df['Performance'].value_counts()
#                 fig = px.pie(
#                     values=performance_counts.values,
#                     names=performance_counts.index,
#                     title='Student Performance Distribution',
#                     color_discrete_sequence=['#667eea', '#f87171'],
#                     hole=0.4
#                 )
#                 fig.update_traces(textposition='inside', textinfo='percent+label')
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             with col2:
#                 st.subheader("📈 Study Hours vs Performance")
#                 fig = px.box(
#                     df, 
#                     x='Performance', 
#                     y='Study_Hours',
#                     color='Performance',
#                     title='Study Hours Impact on Performance',
#                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
#                 )
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             st.markdown("---")
            
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 st.subheader("🎯 Attendance Analysis")
#                 fig = px.histogram(
#                     df, 
#                     x='Attendance', 
#                     color='Performance',
#                     marginal='box',
#                     title='Attendance Distribution by Performance',
#                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
#                 )
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             with col2:
#                 st.subheader("📚 Previous Grade Analysis")
#                 fig = px.violin(
#                     df, 
#                     x='Performance', 
#                     y='Previous_Grade',
#                     color='Performance',
#                     title='Previous Grade Distribution',
#                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
#                     box=True
#                 )
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             st.markdown("---")
#             st.subheader("📋 Student Data Overview")
#             st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

#         # Model Training Page
#         elif page == "📈 Model Training":
#             st.header("🤖 Advanced Machine Learning Model Training")
            
#             col1, col2 = st.columns([1, 2])
            
#             with col1:
#                 st.subheader("⚙️ Model Configuration")
                
#                 model_choice = st.selectbox(
#                     "Select ML Algorithm",
#                     ["Random Forest", "Logistic Regression", "Support Vector Machine", 
#                      "K-Nearest Neighbors", "Gradient Boosting", "Naive Bayes", "Decision Tree"]
#                 )
                
#                 enable_tuning = st.checkbox("🎯 Enable Hyperparameter Tuning", value=False)
                
#                 if enable_tuning:
#                     tuning_method = st.radio("Tuning Method", ["Grid Search", "Random Search"])
#                     n_iter = st.slider("Number of iterations", 5, 50, 10) if tuning_method == "Random Search" else None
                
#                 test_size = st.slider("Test Set Size (%)", 10, 50, 30) / 100
                
#                 enable_feature_selection = st.checkbox("🎯 Enable Feature Selection", value=False)
#                 if enable_feature_selection:
#                     n_features = st.slider("Number of features to select", 1, len(feature_columns), len(feature_columns))
                
#                 if st.button("🚀 Train Model", use_container_width=True):
#                     with st.spinner(f"Training {model_choice}..."):
#                         progress_bar = st.progress(0)
                        
#                         # Feature selection
#                         if enable_feature_selection:
#                             selector = SelectKBest(f_classif, k=n_features)
#                             X_selected = selector.fit_transform(X, y)
#                             selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
#                             st.info(f"Selected features: {', '.join(selected_features)}")
#                         else:
#                             X_selected = X
#                             selected_features = feature_columns
                        
#                         progress_bar.progress(20)
                        
#                         # Split data
#                         X_train, X_test, y_train, y_test = train_test_split(
#                             X_selected, y, test_size=test_size, random_state=42
#                         )
                        
#                         progress_bar.progress(30)
                        
#                         # Scale features
#                         scaler = StandardScaler()
#                         X_train_scaled = scaler.fit_transform(X_train)
#                         X_test_scaled = scaler.transform(X_test)
                        
#                         progress_bar.progress(40)
                        
#                         # Model selection and training
#                         if model_choice == "Random Forest":
#                             if enable_tuning:
#                                 param_grid = {
#                                     'n_estimators': [50, 100, 200],
#                                     'max_depth': [10, 20, 30, None],
#                                     'min_samples_split': [2, 5, 10]
#                                 }
#                                 base_model = RandomForestClassifier(random_state=42)
#                             else:
#                                 model = RandomForestClassifier(n_estimators=100, random_state=42)
                        
#                         elif model_choice == "Logistic Regression":
#                             if enable_tuning:
#                                 param_grid = {
#                                     'C': [0.001, 0.01, 0.1, 1, 10],
#                                     'penalty': ['l2'],
#                                     'solver': ['lbfgs', 'liblinear']
#                                 }
#                                 base_model = LogisticRegression(random_state=42, max_iter=1000)
#                             else:
#                                 model = LogisticRegression(random_state=42, max_iter=1000)
                        
#                         elif model_choice == "Support Vector Machine":
#                             if enable_tuning:
#                                 param_grid = {
#                                     'C': [0.1, 1, 10],
#                                     'kernel': ['linear', 'rbf'],
#                                     'gamma': ['scale', 'auto']
#                                 }
#                                 base_model = SVC(random_state=42, probability=True)
#                             else:
#                                 model = SVC(random_state=42, probability=True)
                        
#                         elif model_choice == "K-Nearest Neighbors":
#                             if enable_tuning:
#                                 param_grid = {
#                                     'n_neighbors': [3, 5, 7, 9, 11],
#                                     'weights': ['uniform', 'distance'],
#                                     'metric': ['euclidean', 'manhattan']
#                                 }
#                                 base_model = KNeighborsClassifier()
#                             else:
#                                 model = KNeighborsClassifier(n_neighbors=5)
                        
#                         elif model_choice == "Gradient Boosting":
#                             if enable_tuning:
#                                 param_grid = {
#                                     'n_estimators': [50, 100, 200],
#                                     'learning_rate': [0.01, 0.1, 0.2],
#                                     'max_depth': [3, 5, 7]
#                                 }
#                                 base_model = GradientBoostingClassifier(random_state=42)
#                             else:
#                                 model = GradientBoostingClassifier(random_state=42)
                        
#                         elif model_choice == "Naive Bayes":
#                             model = GaussianNB()
#                             enable_tuning = False
                        
#                         elif model_choice == "Decision Tree":
#                             if enable_tuning:
#                                 param_grid = {
#                                     'max_depth': [5, 10, 20, None],
#                                     'min_samples_split': [2, 5, 10],
#                                     'criterion': ['gini', 'entropy']
#                                 }
#                                 base_model = DecisionTreeClassifier(random_state=42)
#                             else:
#                                 model = DecisionTreeClassifier(random_state=42)
                        
#                         progress_bar.progress(50)
                        
#                         # Hyperparameter tuning
#                         if enable_tuning and model_choice != "Naive Bayes":
#                             if tuning_method == "Grid Search":
#                                 search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#                             else:
#                                 search = RandomizedSearchCV(base_model, param_grid, cv=5, 
#                                                           scoring='accuracy', n_iter=n_iter, random_state=42, n_jobs=-1)
                            
#                             search.fit(X_train_scaled, y_train)
#                             model = search.best_estimator_
#                             best_params = search.best_params_
#                             st.success(f"✅ Best parameters found: {best_params}")
#                         else:
#                             model.fit(X_train_scaled, y_train)
#                             best_params = None
                        
#                         progress_bar.progress(80)
                        
#                         # Predictions
#                         y_pred = model.predict(X_test_scaled)
#                         y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                        
#                         # Metrics
#                         accuracy = accuracy_score(y_test, y_pred)
#                         cm = confusion_matrix(y_test, y_pred)
#                         precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                        
#                         progress_bar.progress(100)
                        
#                         # Store model
#                         st.session_state.trained_models[model_choice] = {
#                             'model': model,
#                             'scaler': scaler,
#                             'accuracy': accuracy,
#                             'precision': precision,
#                             'recall': recall,
#                             'f1': f1,
#                             'cm': cm,
#                             'features': selected_features,
#                             'best_params': best_params,
#                             'X_test': X_test_scaled,
#                             'y_test': y_test,
#                             'y_pred': y_pred,
#                             'y_pred_proba': y_pred_proba
#                         }
                        
#                         st.success(f"✅ {model_choice} trained successfully!")
#                         st.balloons()
#                         progress_bar.empty()
                        
#                         # Save model option
#                         model_bytes = io.BytesIO()
#                         joblib.dump({'model': model, 'scaler': scaler, 'features': selected_features}, model_bytes)
#                         model_bytes.seek(0)
                        
#                         st.download_button(
#                             label=f"💾 Download {model_choice} Model",
#                             data=model_bytes,
#                             file_name=f"{model_choice.lower().replace(' ', '_')}_model.pkl",
#                             mime="application/octet-stream",
#                             use_container_width=True
#                         )
            
#             with col2:
#                 st.subheader("📊 Model Comparison Dashboard")
                
#                 if st.session_state.trained_models:
#                     # Create comparison dataframe
#                     comparison_data = []
#                     for model_name, model_data in st.session_state.trained_models.items():
#                         comparison_data.append({
#                             'Model': model_name,
#                             'Accuracy': f"{model_data['accuracy']*100:.2f}%",
#                             'Precision': f"{model_data['precision']*100:.2f}%",
#                             'Recall': f"{model_data['recall']*100:.2f}%",
#                             'F1-Score': f"{model_data['f1']*100:.2f}%"
#                         })
                    
#                     comparison_df = pd.DataFrame(comparison_data)
#                     st.dataframe(comparison_df, use_container_width=True)
                    
#                     # Accuracy comparison chart
#                     fig = px.bar(
#                         comparison_df,
#                         x='Model',
#                         y=[float(x.strip('%')) for x in comparison_df['Accuracy']],
#                         title='Model Accuracy Comparison',
#                         labels={'y': 'Accuracy (%)'},
#                         color=[float(x.strip('%')) for x in comparison_df['Accuracy']],
#                         color_continuous_scale='Viridis'
#                     )
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.info("👈 Train a model to see results here")
            
#             # Detailed results for trained models
#             if st.session_state.trained_models:
#                 st.markdown("---")
#                 st.subheader("📈 Detailed Model Results")
                
#                 tabs = st.tabs([name for name in st.session_state.trained_models.keys()])
                
#                 for idx, (model_name, model_data) in enumerate(st.session_state.trained_models.items()):
#                     with tabs[idx]:
#                         col1, col2, col3 = st.columns(3)
                        
#                         with col1:
#                             st.metric("Accuracy", f"{model_data['accuracy']*100:.2f}%")
#                             st.metric("Precision", f"{model_data['precision']*100:.2f}%")
                        
#                         with col2:
#                             st.metric("Recall", f"{model_data['recall']*100:.2f}%")
#                             st.metric("F1-Score", f"{model_data['f1']*100:.2f}%")
                        
#                         with col3:
#                             if model_data['best_params']:
#                                 st.write("**Best Parameters:**")
#                                 for param, value in model_data['best_params'].items():
#                                     st.write(f"• {param}: {value}")
                        
#                         col1, col2 = st.columns(2)
                        
#                         with col1:
#                             st.subheader("🔥 Confusion Matrix")
#                             cm = model_data['cm']
#                             fig = px.imshow(
#                                 cm,
#                                 labels=dict(x="Predicted", y="Actual", color="Count"),
#                                 x=['Fail', 'Pass'],
#                                 y=['Fail', 'Pass'],
#                                 color_continuous_scale='Blues',
#                                 text_auto=True,
#                                 title=f'{model_name} Confusion Matrix'
#                             )
#                             fig.update_layout(height=400)
#                             st.plotly_chart(fig, use_container_width=True)
                        
#                         with col2:
#                             st.subheader("📊 Classification Report")
#                             try:
#                                 report = classification_report(
#                                     model_data['y_test'], 
#                                     model_data['y_pred'],
#                                     target_names=['Fail', 'Pass'],
#                                     output_dict=True,
#                                     zero_division=0
#                                 )
#                                 report_df = pd.DataFrame(report).transpose()
#                                 st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
#                             except:
#                                 st.write("Classification report unavailable")
                        
#                         # ROC Curve
#                         if model_data['y_pred_proba'] is not None:
#                             st.subheader("📈 ROC Curve")
#                             fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'][:, 1])
#                             roc_auc = auc(fpr, tpr)
                            
#                             fig = go.Figure()
#                             fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
#                                                     name=f'ROC Curve (AUC = {roc_auc:.2f})',
#                                                     line=dict(color='#667eea', width=3)))
#                             fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
#                                                     name='Random Classifier',
#                                                     line=dict(color='gray', width=2, dash='dash')))
#                             fig.update_layout(
#                                 title=f'{model_name} ROC Curve',
#                                 xaxis_title='False Positive Rate',
#                                 yaxis_title='True Positive Rate',
#                                 height=400
#                             )
#                             st.plotly_chart(fig, use_container_width=True)
                        
#                         # Feature Importance
#                         if hasattr(model_data['model'], 'feature_importances_'):
#                             st.subheader("🌟 Feature Importance")
#                             importance_df = pd.DataFrame({
#                                 'Feature': model_data['features'],
#                                 'Importance': model_data['model'].feature_importances_
#                             }).sort_values('Importance', ascending=False)
                            
#                             fig = px.bar(
#                                 importance_df,
#                                 x='Importance',
#                                 y='Feature',
#                                 orientation='h',
#                                 title=f'{model_name} Feature Importance',
#                                 color='Importance',
#                                 color_continuous_scale='Viridis'
#                             )
#                             fig.update_layout(height=400)
#                             st.plotly_chart(fig, use_container_width=True)

#         # Prediction Page
#         elif page == "🔮 Prediction":
#             st.header("🔮 Student Performance Prediction")
            
#             if not st.session_state.trained_models:
#                 st.warning("⚠️ Please train at least one model first!")
#                 if st.button("Go to Model Training"):
#                     st.session_state.page = "📈 Model Training"
#                     st.rerun()
#             else:
#                 col1, col2 = st.columns([2, 1])
                
#                 with col2:
#                     st.subheader("⚙️ Prediction Settings")
#                     selected_model = st.selectbox(
#                         "Select Model for Prediction",
#                         list(st.session_state.trained_models.keys())
#                     )
                    
#                     model_info = st.session_state.trained_models[selected_model]
#                     st.metric("Model Accuracy", f"{model_info['accuracy']*100:.2f}%")
                
#                 with col1:
#                     st.markdown("### Enter Student Information")
                    
#                     col1a, col1b, col1c = st.columns(3)
                    
#                     with col1a:
#                         gender = st.selectbox("Gender", df['Gender'].unique())
                        
#                         study_min = int(df['Study_Hours'].min())
#                         study_max = int(df['Study_Hours'].max())
#                         study_mean = int(df['Study_Hours'].mean())
                        
#                         if study_min == study_max:
#                             st.info(f"Study Hours: {study_min} (constant)")
#                             study_hours = study_min
#                         else:
#                             study_hours = st.slider("Study Hours per Day", study_min, study_max, study_mean)
                    
#                     with col1b:
#                         att_min = int(df['Attendance'].min())
#                         att_max = int(df['Attendance'].max())
#                         att_mean = int(df['Attendance'].mean())
                        
#                         if att_min == att_max:
#                             st.info(f"Attendance: {att_min}%")
#                             attendance = att_min
#                         else:
#                             attendance = st.slider("Attendance %", att_min, att_max, att_mean)
                        
#                         grade_min = int(df['Previous_Grade'].min())
#                         grade_max = int(df['Previous_Grade'].max())
#                         grade_mean = int(df['Previous_Grade'].mean())
                        
#                         if grade_min == grade_max:
#                             st.info(f"Previous Grade: {grade_min}")
#                             previous_grade = grade_min
#                         else:
#                             previous_grade = st.slider("Previous Grade", grade_min, grade_max, grade_mean)
                    
#                     with col1c:
#                         parent_education = st.selectbox("Parent Education", df['Parent_Education'].unique())
#                         internet_access = st.selectbox("Internet Access", df['Internet_Access'].unique())
                
#                 if st.button("🎯 Predict Performance", use_container_width=True):
#                     # Encode inputs
#                     gender_encoded = le_gender.transform([gender])[0]
#                     parent_encoded = le_parent.transform([parent_education])[0]
#                     internet_encoded = le_internet.transform([internet_access])[0]
                    
#                     input_dict = {
#                         'Gender': gender_encoded,
#                         'Study_Hours': study_hours,
#                         'Attendance': attendance,
#                         'Previous_Grade': previous_grade,
#                         'Parent_Education': parent_encoded,
#                         'Internet_Access': internet_encoded
#                     }
                    
#                     # Use model's features
#                     input_data = np.array([[input_dict[col] for col in model_info['features']]])
#                     input_data_scaled = model_info['scaler'].transform(input_data)
                    
#                     prediction = model_info['model'].predict(input_data_scaled)[0]
#                     prediction_proba = model_info['model'].predict_proba(input_data_scaled)[0] if hasattr(model_info['model'], 'predict_proba') else [0.5, 0.5]
                    
#                     result = le_performance.inverse_transform([prediction])[0]
                    
#                     st.markdown("---")
#                     st.markdown("### 🎊 Prediction Results")
                    
#                     col1, col2, col3 = st.columns([1, 2, 1])
                    
#                     with col2:
#                         if result == 'Pass':
#                             st.success(f"### ✅ Predicted Performance: PASS")
#                             st.balloons()
#                         else:
#                             st.error(f"### ❌ Predicted Performance: FAIL")
                        
#                         confidence = max(prediction_proba) * 100
#                         st.markdown(f"**Confidence:** {confidence:.2f}%")
#                         st.markdown(f"**Model Used:** {selected_model}")
                        
#                         # Confidence visualization
#                         fig = go.Figure()
#                         fig.add_trace(go.Bar(
#                             x=prediction_proba,
#                             y=['Fail', 'Pass'],
#                             orientation='h',
#                             marker=dict(color=['#f87171', '#667eea']),
#                             text=[f'{p*100:.1f}%' for p in prediction_proba],
#                             textposition='auto'
#                         ))
#                         fig.update_layout(
#                             title='Prediction Confidence',
#                             xaxis_title='Probability',
#                             height=300,
#                             showlegend=False
#                         )
#                         st.plotly_chart(fig, use_container_width=True)
                        
#                         # Recommendations
#                         st.markdown("### 💡 Recommendations")
#                         if result == 'Fail':
#                             recommendations = []
#                             if study_hours < df[df['Performance']=='Pass']['Study_Hours'].mean():
#                                 target_hours = df[df['Performance']=='Pass']['Study_Hours'].mean()
#                                 recommendations.append(f"📚 Increase study hours to at least {target_hours:.1f} hours per day")
#                             if attendance < df[df['Performance']=='Pass']['Attendance'].mean():
#                                 target_att = df[df['Performance']=='Pass']['Attendance'].mean()
#                                 recommendations.append(f"📅 Improve attendance to above {target_att:.1f}%")
#                             if previous_grade < df[df['Performance']=='Pass']['Previous_Grade'].mean():
#                                 recommendations.append("📖 Focus on strengthening foundational concepts")
                            
#                             for rec in recommendations:
#                                 st.warning(rec)
#                         else:
#                             st.success("🎉 Great! Keep up the excellent work!")
#                             st.info("💪 Continue maintaining your study habits and attendance")

#         # Data Analysis Page
#         elif page == "📊 Data Analysis":
#             st.header("📊 Advanced Data Analysis")
            
#             tab1, tab2, tab3, tab4 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights", "📉 Trends"])
            
#             with tab1:
#                 st.subheader("Correlation Analysis")
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     correlation = df_encoded.corr()
#                     fig = px.imshow(
#                         correlation,
#                         labels=dict(color="Correlation"),
#                         x=correlation.columns,
#                         y=correlation.columns,
#                         color_continuous_scale='RdBu',
#                         zmin=-1, zmax=1,
#                         title='Feature Correlation Heatmap',
#                         text_auto='.2f'
#                     )
#                     fig.update_layout(height=600)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 with col2:
#                     st.write("### Key Correlations")
#                     perf_corr = correlation['Performance'].abs().sort_values(ascending=False)[1:]
#                     st.dataframe(perf_corr.to_frame('Correlation'), use_container_width=True)
            
#             with tab2:
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.subheader("Attendance Distribution")
#                     fig = px.histogram(
#                         df, 
#                         x='Attendance', 
#                         color='Performance',
#                         marginal='violin',
#                         title='Attendance Impact on Performance',
#                         color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
#                         nbins=20
#                     )
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 with col2:
#                     st.subheader("Previous Grade Distribution")
#                     fig = px.histogram(
#                         df, 
#                         x='Previous_Grade', 
#                         color='Performance',
#                         marginal='box',
#                         title='Previous Grade Impact',
#                         color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
#                         nbins=20
#                     )
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 st.subheader("Study Hours Analysis")
#                 fig = px.scatter(
#                     df,
#                     x='Study_Hours',
#                     y='Previous_Grade',
#                     color='Performance',
#                     size='Attendance',
#                     hover_data=['Gender', 'Parent_Education'],
#                     title='Multi-dimensional Performance Analysis',
#                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
#                 )
#                 fig.update_layout(height=500)
#                 st.plotly_chart(fig, use_container_width=True)
            
#             with tab3:
#                 st.subheader("📌 Statistical Insights")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     avg_study_pass = df[df['Performance']=='Pass']['Study_Hours'].mean()
#                     avg_study_fail = df[df['Performance']=='Fail']['Study_Hours'].mean()
#                     avg_att_pass = df[df['Performance']=='Pass']['Attendance'].mean()
#                     avg_att_fail = df[df['Performance']=='Fail']['Attendance'].mean()
#                     avg_grade_pass = df[df['Performance']=='Pass']['Previous_Grade'].mean()
#                     avg_grade_fail = df[df['Performance']=='Fail']['Previous_Grade'].mean()
                    
#                     st.info(f"""
#                     **Academic Performance Insights:**
                    
#                     📚 **Study Hours:**
#                     - Pass: {avg_study_pass:.1f} hours
#                     - Fail: {avg_study_fail:.1f} hours
#                     - Difference: {avg_study_pass - avg_study_fail:.1f} hours
                    
#                     📅 **Attendance:**
#                     - Pass: {avg_att_pass:.1f}%
#                     - Fail: {avg_att_fail:.1f}%
#                     - Difference: {avg_att_pass - avg_att_fail:.1f}%
                    
#                     📖 **Previous Grades:**
#                     - Pass: {avg_grade_pass:.1f}
#                     - Fail: {avg_grade_fail:.1f}
#                     - Difference: {avg_grade_pass - avg_grade_fail:.1f}
#                     """)
                
#                 with col2:
#                     # Gender performance
#                     gender_perf = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
#                     fig = px.bar(
#                         gender_perf,
#                         title='Performance by Gender',
#                         labels={'value': 'Percentage (%)', 'variable': 'Performance'},
#                         color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
#                         barmode='group'
#                     )
#                     fig.update_layout(height=350)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 st.markdown("---")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     # Parent education impact
#                     parent_perf = pd.crosstab(df['Parent_Education'], df['Performance'], normalize='index') * 100
#                     fig = px.bar(
#                         parent_perf,
#                         title='Performance by Parent Education',
#                         labels={'value': 'Percentage (%)', 'variable': 'Performance'},
#                         color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
#                         barmode='group'
#                     )
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 with col2:
#                     # Internet access impact
#                     internet_perf = pd.crosstab(df['Internet_Access'], df['Performance'], normalize='index') * 100
#                     fig = px.bar(
#                         internet_perf,
#                         title='Performance by Internet Access',
#                         labels={'value': 'Percentage (%)', 'variable': 'Performance'},
#                         color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
#                         barmode='group'
#                     )
#                     fig.update_layout(height=400)
#                     st.plotly_chart(fig, use_container_width=True)
            
#             with tab4:
#                 st.subheader("📉 Performance Trends")
                
#                 # Create synthetic trend data
#                 metrics_df = pd.DataFrame({
#                     'Metric': ['Study Hours', 'Attendance', 'Previous Grade'] * 2,
#                     'Performance': ['Pass'] * 3 + ['Fail'] * 3,
#                     'Average': [
#                         df[df['Performance']=='Pass']['Study_Hours'].mean(),
#                         df[df['Performance']=='Pass']['Attendance'].mean(),
#                         df[df['Performance']=='Pass']['Previous_Grade'].mean(),
#                         df[df['Performance']=='Fail']['Study_Hours'].mean(),
#                         df[df['Performance']=='Fail']['Attendance'].mean(),
#                         df[df['Performance']=='Fail']['Previous_Grade'].mean()
#                     ]
#                 })
                
#                 fig = px.line(
#                     metrics_df,
#                     x='Metric',
#                     y='Average',
#                     color='Performance',
#                     markers=True,
#                     title='Average Metrics Comparison',
#                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
#                 )
#                 fig.update_layout(height=400)
#                 st.plotly_chart(fig, use_container_width=True)
                
#                 # Sunburst chart
#                 st.subheader("🌅 Hierarchical Performance Analysis")
#                 sunburst_df = df.groupby(['Gender', 'Parent_Education', 'Performance']).size().reset_index(name='Count')
#                 fig = px.sunburst(
#                     sunburst_df,
#                     path=['Gender', 'Parent_Education', 'Performance'],
#                     values='Count',
#                     title='Hierarchical Performance Breakdown',
#                     color='Performance',
#                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
#                 )
#                 fig.update_layout(height=600)
#                 st.plotly_chart(fig, use_container_width=True)

#         # AI Explainability Page
#         elif page == "🧠 AI Explainability":
#             st.header("🧠 AI Model Explainability")
            
#             if not st.session_state.trained_models:
#                 st.warning("⚠️ Please train at least one model first!")
#             else:
#                 st.info("💡 **SHAP Values** help explain which features most influenced each prediction")
                
#                 selected_model = st.selectbox(
#                     "Select Model to Explain",
#                     [name for name in st.session_state.trained_models.keys() 
#                      if hasattr(st.session_state.trained_models[name]['model'], 'feature_importances_')]
#                 )
                
#                 if selected_model:
#                     model_info = st.session_state.trained_models[selected_model]
                    
#                     st.subheader(f"📊 {selected_model} Feature Importance")
                    
#                     if hasattr(model_info['model'], 'feature_importances_'):
#                         importance_df = pd.DataFrame({
#                             'Feature': model_info['features'],
#                             'Importance': model_info['model'].feature_importances_
#                         }).sort_values('Importance', ascending=False)
                        
#                         col1, col2 = st.columns([2, 1])
                        
#                         with col1:
#                             fig = px.treemap(
#                                 importance_df,
#                                 path=['Feature'],
#                                 values='Importance',
#                                 title=f'{selected_model} Feature Importance (Treemap)',
#                                 color='Importance',
#                                 color_continuous_scale='Viridis'
#                             )
#                             fig.update_layout(height=500)
#                             st.plotly_chart(fig, use_container_width=True)
                        
#                         with col2:
#                             st.dataframe(
#                                 importance_df.style.background_gradient(cmap='Viridis', subset=['Importance']),
#                                 use_container_width=True,
#                                 height=500
#                             )
                        
#                         st.markdown("---")
                        
#                         # Feature impact analysis
#                         st.subheader("🎯 Feature Impact Analysis")
                        
#                         feature_to_analyze = st.selectbox("Select Feature to Analyze", model_info['features'])
                        
#                         if feature_to_analyze in df.columns:
#                             col1, col2 = st.columns(2)
                            
#                             with col1:
#                                 fig = px.box(
#                                     df,
#                                     x='Performance',
#                                     y=feature_to_analyze,
#                                     color='Performance',
#                                     title=f'{feature_to_analyze} Distribution by Performance',
#                                     color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
#                                 )
#                                 fig.update_layout(height=400)
#                                 st.plotly_chart(fig, use_container_width=True)
                            
#                             with col2:
#                                 # Statistics
#                                 st.write(f"**{feature_to_analyze} Statistics:**")
#                                 pass_mean = df[df['Performance']=='Pass'][feature_to_analyze].mean()
#                                 fail_mean = df[df['Performance']=='Fail'][feature_to_analyze].mean()
                                
#                                 st.metric("Pass Average", f"{pass_mean:.2f}")
#                                 st.metric("Fail Average", f"{fail_mean:.2f}")
#                                 st.metric("Difference", f"{pass_mean - fail_mean:.2f}")
#                     else:
#                         st.warning("This model doesn't support feature importance visualization")

#         # Student Clustering Page
#         elif page == "👥 Student Clustering":
#             st.header("👥 Student Segmentation & Clustering")
            
#             st.info("💡 Use K-Means clustering to identify different student groups based on their characteristics")
            
#             col1, col2 = st.columns([1, 2])
            
#             with col1:
#                 st.subheader("⚙️ Clustering Settings")
#                 n_clusters = st.slider("Number of Clusters", 2, 6, 3)
                
#                 features_for_clustering = st.multiselect(
#                     "Select Features for Clustering",
#                     feature_columns,
#                     default=feature_columns[:3]
#                 )
                
#                 if st.button("🎯 Perform Clustering", use_container_width=True):
#                     if len(features_for_clustering) < 2:
#                         st.error("Please select at least 2 features")
#                     else:
#                         with st.spinner("Performing clustering..."):
#                             # Prepare data
#                             X_cluster = df_encoded[features_for_clustering]
#                             scaler = StandardScaler()
#                             X_scaled = scaler.fit_transform(X_cluster)
                            
#                             # Perform clustering
#                             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#                             clusters = kmeans.fit_predict(X_scaled)
                            
#                             # Add clusters to dataframe
#                             df['Cluster'] = clusters
#                             df['Cluster'] = df['Cluster'].apply(lambda x: f"Group {x+1}")
                            
#                             st.session_state.clusters = clusters
#                             st.session_state.cluster_centers = kmeans.cluster_centers_
#                             st.session_state.clustering_features = features_for_clustering
                            
#                             st.success("✅ Clustering completed!")
            
#             with col2:
#                 if 'clusters' in st.session_state:
#                     st.subheader("📊 Clustering Results")
                    
#                     # Cluster distribution
#                     cluster_counts = df['Cluster'].value_counts()
#                     fig = px.pie(
#                         values=cluster_counts.values,
#                         names=cluster_counts.index,
#                         title='Student Distribution by Cluster',
#                         hole=0.4
#                     )
#                     st.plotly_chart(fig, use_container_width=True)
#                 else:
#                     st.info("👈 Configure and run clustering to see results")
            
#             if 'clusters' in st.session_state:
#                 st.markdown("---")
                
#                 # Visualization
#                 if len(st.session_state.clustering_features) >= 2:
#                     st.subheader("🎨 Cluster Visualization")
                    
#                     feat1 = st.session_state.clustering_features[0]
#                     feat2 = st.session_state.clustering_features[1]
                    
#                     fig = px.scatter(
#                         df,
#                         x=feat1,
#                         y=feat2,
#                         color='Cluster',
#                         symbol='Performance',
#                         title=f'Student Clusters: {feat1} vs {feat2}',
#                         hover_data=['Study_Hours', 'Attendance', 'Previous_Grade']
#                     )
#                     fig.update_layout(height=500)
#                     st.plotly_chart(fig, use_container_width=True)
                
#                 # Cluster characteristics
#                 st.subheader("📋 Cluster Characteristics")
                
#                 cluster_stats = df.groupby('Cluster').agg({
#                     'Study_Hours': 'mean',
#                     'Attendance': 'mean',
#                     'Previous_Grade': 'mean',
#                     'Performance': lambda x: (x == 'Pass').sum() / len(x) * 100
#                 }).round(2)
                
#                 cluster_stats.columns = ['Avg Study Hours', 'Avg Attendance', 'Avg Previous Grade', 'Pass Rate (%)']
#                 st.dataframe(cluster_stats, use_container_width=True)
                
#                 # Cluster profiles
#                 st.subheader("👤 Cluster Profiles")
                
#                 tabs = st.tabs([f"Group {i+1}" for i in range(n_clusters)])
                
#                 for idx, tab in enumerate(tabs):
#                     with tab:
#                         cluster_name = f"Group {idx+1}"
#                         cluster_data = df[df['Cluster'] == cluster_name]
                        
#                         col1, col2, col3, col4 = st.columns(4)
                        
#                         with col1:
#                             st.metric("Students", len(cluster_data))
#                         with col2:
#                             pass_rate = (cluster_data['Performance'] == 'Pass').sum() / len(cluster_data) * 100
#                             st.metric("Pass Rate", f"{pass_rate:.1f}%")
#                         with col3:
#                             st.metric("Avg Study", f"{cluster_data['Study_Hours'].mean():.1f}h")
#                         with col4:
#                             st.metric("Avg Attendance", f"{cluster_data['Attendance'].mean():.1f}%")
                        
#                         st.markdown("**Cluster Description:**")
#                         if pass_rate > 70:
#                             st.success("🌟 **High Achievers**: This group shows excellent performance with strong study habits.")
#                         elif pass_rate > 40:
#                             st.info("📚 **Moderate Performers**: This group has potential for improvement with targeted support.")
#                         else:
#                             st.warning("⚠️ **At-Risk Students**: This group needs immediate intervention and support.")

#         # Advanced Settings Page
#         elif page == "⚙️ Advanced Settings":
#             st.header("⚙️ Advanced Settings & Model Management")
            
#             tab1, tab2, tab3 = st.tabs(["💾 Model Management", "📊 Data Export", "🔧 System Info"])
            
#             with tab1:
#                 st.subheader("💾 Model Management")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("### Save Models")
#                     if st.session_state.trained_models:
#                         for model_name, model_data in st.session_state.trained_models.items():
#                             model_bytes = io.BytesIO()
#                             joblib.dump({
#                                 'model': model_data['model'],
#                                 'scaler': model_data['scaler'],
#                                 'features': model_data['features']
#                             }, model_bytes)
#                             model_bytes.seek(0)
                            
#                             st.download_button(
#                                 label=f"💾 Download {model_name}",
#                                 data=model_bytes,
#                                 file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
#                                 mime="application/octet-stream",
#                                 use_container_width=True
#                             )
#                     else:
#                         st.info("No trained models available")
                
#                 with col2:
#                     st.markdown("### Load Model")
#                     uploaded_model = st.file_uploader("Upload trained model (.pkl)", type=['pkl'])
                    
#                     if uploaded_model:
#                         try:
#                             model_data = joblib.load(uploaded_model)
#                             st.success("✅ Model loaded successfully!")
#                             st.write("**Model Info:**")
#                             st.write(f"- Features: {', '.join(model_data['features'])}")
#                         except Exception as e:
#                             st.error(f"Error loading model: {str(e)}")
                
#                 st.markdown("---")
                
#                 if st.session_state.trained_models:
#                     st.markdown("### Clear Models")
#                     if st.button("🗑️ Clear All Trained Models", use_container_width=True):
#                         st.session_state.trained_models = {}
#                         gc.collect()
#                         st.success("✅ All models cleared!")
#                         st.rerun()
            
#             with tab2:
#                 st.subheader("📊 Data Export Options")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("### Export Data")
                    
#                     # Original data
#                     csv = df.to_csv(index=False).encode('utf-8')
#                     st.download_button(
#                         label="📥 Download Original Data",
#                         data=csv,
#                         file_name="student_data.csv",
#                         mime="text/csv",
#                         use_container_width=True
#                     )
                    
#                     # Encoded data
#                     encoded_csv = df_encoded.to_csv(index=False).encode('utf-8')
#                     st.download_button(
#                         label="📥 Download Encoded Data",
#                         data=encoded_csv,
#                         file_name="encoded_student_data.csv",
#                         mime="text/csv",
#                         use_container_width=True
#                     )
                    
#                     # Cluster data (if available)
#                     if 'Cluster' in df.columns:
#                         cluster_csv = df.to_csv(index=False).encode('utf-8')
#                         st.download_button(
#                             label="📥 Download Clustered Data",
#                             data=cluster_csv,
#                             file_name="clustered_student_data.csv",
#                             mime="text/csv",
#                             use_container_width=True
#                         )
                
#                 with col2:
#                     st.markdown("### Export Reports")
                    
#                     if st.session_state.trained_models:
#                         # Create comprehensive report
#                         report_data = []
#                         for model_name, model_data in st.session_state.trained_models.items():
#                             report_data.append({
#                                 'Model': model_name,
#                                 'Accuracy': f"{model_data['accuracy']*100:.2f}%",
#                                 'Precision': f"{model_data['precision']*100:.2f}%",
#                                 'Recall': f"{model_data['recall']*100:.2f}%",
#                                 'F1-Score': f"{model_data['f1']*100:.2f}%"
#                             })
                        
#                         report_df = pd.DataFrame(report_data)
#                         report_csv = report_df.to_csv(index=False).encode('utf-8')
                        
#                         st.download_button(
#                             label="📊 Download Model Report",
#                             data=report_csv,
#                             file_name="model_comparison_report.csv",
#                             mime="text/csv",
#                             use_container_width=True
#                         )
#                     else:
#                         st.info("Train models first to generate reports")
            
#             with tab3:
#                 st.subheader("🔧 System Information")
                
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown("### Dataset Info")
#                     st.write(f"**Total Records:** {len(df)}")
#                     st.write(f"**Features:** {len(feature_columns)}")
#                     st.write(f"**Pass Count:** {(df['Performance'] == 'Pass').sum()}")
#                     st.write(f"**Fail Count:** {(df['Performance'] == 'Fail').sum()}")
#                     st.write(f"**Data Quality:** {(df.isna().sum().sum() == 0) and '✅ Clean' or '⚠️ Contains missing values'}")
                
#                 with col2:
#                     st.markdown("### Model Info")
#                     st.write(f"**Trained Models:** {len(st.session_state.trained_models)}")
#                     if st.session_state.trained_models:
#                         best_model = max(st.session_state.trained_models.items(), 
#                                        key=lambda x: x[1]['accuracy'])
#                         st.write(f"**Best Model:** {best_model[0]}")
#                         st.write(f"**Best Accuracy:** {best_model[1]['accuracy']*100:.2f}%")
                
#                 st.markdown("---")
                
#                 st.markdown("### Feature Statistics")
#                 stats_df = df[['Study_Hours', 'Attendance', 'Previous_Grade']].describe().T
#                 st.dataframe(stats_df, use_container_width=True)
                
#                 st.markdown("---")
                
#                 st.markdown("### Session State")
#                 with st.expander("🔍 View Session State"):
#                     st.write({
#                         'data_loaded': st.session_state.data_loaded,
#                         'models_trained': len(st.session_state.trained_models),
#                         'dark_mode': st.session_state.dark_mode,
#                         'clusters_created': 'clusters' in st.session_state
#                     })

#     except Exception as e:
#         st.error(f"❌ Error during processing: {str(e)}")
#         st.write("Please check your data and try again.")
        
#         with st.expander("🔍 Debug Information"):
#             st.write("**Available columns:**", list(df.columns))
#             st.write("**Data types:**")
#             st.write(df.dtypes)
#             st.write("**First few rows:**")
#             st.write(df.head())
#             st.write("**Error Details:**")
#             import traceback
#             st.code(traceback.format_exc())

# # Footer
# st.markdown("---")
# st.markdown("""
#     <div style='text-align: center; color: #64748b; padding: 20px; font-family: "Poppins", sans-serif;'>
#         <p style='font-size: 1.1rem; font-weight: 600;'><strong>Advanced Student Performance Prediction System</strong></p>
#         <p style='font-size: 0.95rem;'>Powered by Multiple ML Models | Hyperparameter Tuning | AI Explainability</p>
#         <p style='font-size: 0.9rem; margin-top: 10px;'>
#             Developed by <a href='https://www.linkedin.com/in/rashid-ali-619671357/' target='_blank' style='color: #0077B5; text-decoration: none; font-weight: 600;'>Rashid Ali</a>
#         </p>
#         <p style='font-size: 0.85rem; color: #94a3b8; margin-top: 5px;'>
#             © 2025 | PITP & Gexton Educational Excellence
#         </p>
#     </div>
#     """, unsafe_allow_html=True)













#Dark mode fixed and final one without Errors

import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import gc
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state FIRST before any checks
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Custom CSS with dark mode support
dark_mode_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Montserrat:wght@700;900&display=swap');

    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        font-family: 'Poppins', sans-serif;
        color: #e2e8f0;
    }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }

    .stMarkdown {
        color: #e2e8f0;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #818cf8 !important;
    }

    .stDataFrame {
        color: #e2e8f0;
    }

    [data-testid="stExpander"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
"""

light_mode_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Montserrat:wght@700;900&display=swap');

    .main {
        background: linear-gradient(135deg, #74ABE2 0%, #5563DE 50%, #A683E3 100%);
        font-family: 'Poppins', sans-serif;
        color: #1e293b;
    }

    .stApp {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
"""

# Apply the appropriate CSS based on dark mode state
if st.session_state.dark_mode:
    st.markdown(dark_mode_css, unsafe_allow_html=True)
else:
    st.markdown(light_mode_css, unsafe_allow_html=True)

# Common CSS for both modes
st.markdown("""
    <style>

    .metric-card {
        background: """ + ("linear-gradient(145deg, #1e293b 0%, #334155 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #ffffff 0%, #f3f4f6 100%)") + """;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, """ + ("0.3" if st.session_state.dark_mode else "0.1") + """);
        margin: 15px 0;
        transition: all 0.4s ease-in-out;
        border-image: linear-gradient(90deg, #3B82F6, #9333EA) 1;
    }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 12px 24px rgba(0, 0, 0, """ + ("0.4" if st.session_state.dark_mode else "0.15") + """);
    }

    .title-text {
        font-family: 'Montserrat', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 10px;
        letter-spacing: -2px;
        text-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        animation: titlePulse 3s ease-in-out infinite;
    }
    
    @keyframes titlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
.title-text:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 119, 181, 0.5);
        border-radius: 100px;
    }
    .subtitle-text {
        font-family: 'Poppins', sans-serif;
        color: """ + ("#cbd5e1" if st.session_state.dark_mode else "#475569") + """;
        font-size: 1.25rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 35px;
        letter-spacing: 0.6px;
        animation: titlePulse 3s ease-in-out infinite;
    }
 @keyframes subtitlePulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: """ + ("#818cf8" if st.session_state.dark_mode else "#4338ca") + """;
        text-shadow: 1px 1px 8px rgba(99, 102, 241, 0.25);
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: """ + ("#cbd5e1" if st.session_state.dark_mode else "#334155") + """;
    }

    .stButton>button {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(90deg, #6EE7B7 0%, #3B82F6 50%, #9333EA 100%);
        color: white;
        border: none;
        padding: 14px 38px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.6px;
        transition: all 0.3s ease;
        background-size: 200% 200%;
        animation: gradientShift 5s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.4);
    }

    .upload-section {
        background: """ + ("linear-gradient(145deg, #1e293b 0%, #334155 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #ffffff 0%, #f1f5f9 100%)") + """;
        padding: 32px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, """ + ("0.3" if st.session_state.dark_mode else "0.08") + """);
        margin: 25px 0;
        border-left: 6px solid #6366f1;
        transition: all 0.4s ease-in-out;
    }
    
    .upload-section:hover {
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: """ + ("#e2e8f0" if st.session_state.dark_mode else "#374151") + """;
        background: """ + ("linear-gradient(90deg, #1e293b 0%, #334155 100%)" if st.session_state.dark_mode else "linear-gradient(90deg, #f9fafb 0%, #e2e8f0 100%)") + """;
        border-radius: 12px;
        margin-right: 6px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.25);
    }

    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        color: """ + ("#f1f5f9" if st.session_state.dark_mode else "#1e293b") + """;
    }

    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #818cf8, #a78bfa);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #6366f1, #8b5cf6);
    }

    .linkedin-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #0077B5 0%, #00A0DC 100%);
        padding: 12px 20px;
        border-radius: 50px;
        box-shadow: 0 4px 15px rgba(0, 119, 181, 0.3);
        transition: all 0.3s ease;
        z-index: 1000;
        animation: titlePulse 3s ease-in-out infinite;
    }
    
      @keyframes linkedinPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .linkedin-badge:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 119, 181, 0.5);
    }
    
    .linkedin-badge a {
        color: white;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
  

    
    </style>
""", unsafe_allow_html=True)

# LinkedIn Badge
st.markdown("""
    <div class="linkedin-badge">
        <a href="https://www.linkedin.com/in/rashid-ali-619671357/" target="_blank">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
                <path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/>
            </svg>
            Connect Rashid on LinkedIn
        </a>
    </div>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="title-text">🎓 Students Performance Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">It is the best for PITP and Gexton Education Excellence</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
    st.title("📊 Navigation")
    
    # Dark Mode Toggle
    dark_mode = st.toggle("🌗 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    if st.session_state.data_loaded:
        page = st.radio("Select Page:", [
            "🏠 Dashboard", 
            "📈 Model Training", 
            "🔮 Prediction", 
            "📊 Data Analysis",
            "🧠 AI Explainability",
            "👥 Student Clustering",
            "⚙️ Advanced Settings"
        ])
    else:
        page = "📁 Data Upload"
    
    st.markdown("---")
    st.info("**About:** Advanced ML system with multiple algorithms, hyperparameter tuning, and SHAP explainability for predicting student performance.")
    
    if st.session_state.data_loaded:
        st.success(f"✅ Data Loaded: {len(st.session_state.df)} records")
        
        # Model Management
        if st.session_state.trained_models:
            st.markdown("### 🤖 Trained Models")
            for model_name in st.session_state.trained_models.keys():
                st.write(f"✓ {model_name}")
        
        if st.button("🔄 Upload New Dataset"):
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.feature_columns = None
            st.session_state.trained_models = {}
            gc.collect()
            st.rerun()
    
    st.markdown("---")
    st.markdown("Thank You Sir Arham and ma'am Mona Shah for support and guidance.")

# Sample dataset generator
def generate_sample_dataset():
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Study_Hours': np.random.randint(1, 12, n_samples),
        'Attendance': np.random.randint(50, 100, n_samples),
        'Previous_Grade': np.random.randint(40, 100, n_samples),
        'Parent_Education': np.random.choice(['High School', 'Graduate', 'Postgraduate'], n_samples),
        'Internet_Access': np.random.choice(['Yes', 'No'], n_samples),
    }
    
    df = pd.DataFrame(data)
    df['Performance'] = df.apply(
        lambda row: 'Pass' if (row['Study_Hours'] * 0.3 + row['Attendance'] * 0.4 + row['Previous_Grade'] * 0.3) > 60 else 'Fail',
        axis=1
    )
    
    return df

# Data Upload Page
if not st.session_state.data_loaded:
    st.header("📁 Upload Student Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class='upload-section'>
                <h3>📋 Dataset Requirements</h3>
                <p>Please ensure your CSV file contains the following columns:</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        **Required Columns:**
        - Gender (Male/Female)
        - Study_Hours (numeric)
        - Attendance (numeric, 0-100)
        - Previous_Grade (numeric, 0-100)
        - Parent_Education (High School/Graduate/Postgraduate)
        - Internet_Access (Yes/No)
        - Performance (Pass/Fail)
        """)
    
    with col2:
        st.markdown("### 📥 Download Sample Dataset")
        sample_df = generate_sample_dataset()
        csv = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=csv,
            file_name="sample_student_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # File size validation
            file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
            if file_size > 50:
                st.error("❌ File size exceeds 50MB limit. Please upload a smaller file.")
            else:
                df = pd.read_csv(uploaded_file)
                
                required_columns = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
                                  'Parent_Education', 'Internet_Access', 'Performance']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                    
                    # Auto column mapping suggestion
                    st.markdown("### 🔄 Column Mapping Assistant")
                    st.info("Try mapping your columns to the required format:")
                    
                    mapping = {}
                    for req_col in missing_columns:
                        available_cols = [col for col in df.columns if col not in mapping.values()]
                        mapping[req_col] = st.selectbox(f"Map '{req_col}' to:", [''] + available_cols, key=req_col)
                    
                    if st.button("Apply Mapping"):
                        for req_col, user_col in mapping.items():
                            if user_col:
                                df[req_col] = df[user_col]
                        st.success("✅ Columns mapped successfully!")
                        st.rerun()
                else:
                    st.success("✅ Dataset uploaded successfully!")
                    st.toast("🎉 Dataset validated successfully!")
                    
                    st.subheader("📊 Dataset Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Features", len(df.columns)-1)
                    with col3:
                        pass_count = (df['Performance'] == 'Pass').sum()
                        st.metric("Pass Count", pass_count)
                    with col4:
                        fail_count = (df['Performance'] == 'Fail').sum()
                        st.metric("Fail Count", fail_count)
                    
                    if st.button("✅ Proceed with Analysis", use_container_width=True):
                        st.session_state.df = df
                        st.session_state.data_loaded = True
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

else:
    df = st.session_state.df.copy()
    
    REQUIRED_FEATURES = ['Gender', 'Study_Hours', 'Attendance', 'Previous_Grade', 
                        'Parent_Education', 'Internet_Access']
    
    # Data Cleaning with progress bar
    if 'data_cleaned' not in st.session_state:
        st.write("---")
        st.header("🧹 Data Cleaning in Progress...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Column selection
        status_text.text("Step 1/5: Selecting required columns...")
        progress_bar.progress(20)
        available_features = [col for col in REQUIRED_FEATURES if col in df.columns]
        df = df[available_features + ['Performance']]
        
        # Step 2: Performance standardization
        status_text.text("Step 2/5: Standardizing performance values...")
        progress_bar.progress(40)
        if 'Performance' in df.columns:
            df['Performance'] = df['Performance'].astype(str).str.strip().str.lower()
            pass_values = ['pass', 'passed', 'success', 'good', 'excellent', 'yes', '1', 'true']
            fail_values = ['fail', 'failed', 'failure', 'poor', 'bad', 'no', '0', 'false']
            
            df['Performance'] = df['Performance'].apply(lambda x: 
                'Pass' if any(val in str(x).lower() for val in pass_values)
                else 'Fail' if any(val in str(x).lower() for val in fail_values)
                else x
            )
            
            try:
                numeric_perf = pd.to_numeric(df['Performance'], errors='coerce')
                if numeric_perf.notna().any():
                    threshold = numeric_perf.median()
                    df.loc[numeric_perf.notna(), 'Performance'] = numeric_perf[numeric_perf.notna()].apply(
                        lambda x: 'Pass' if x >= threshold else 'Fail'
                    )
            except:
                pass
            
            valid_performance = df['Performance'].isin(['Pass', 'Fail'])
            if not valid_performance.all():
                df = df[valid_performance]
        
        # Step 3: Numeric columns cleaning
        status_text.text("Step 3/5: Cleaning numeric columns...")
        progress_bar.progress(60)
        categorical_columns = ['Gender', 'Parent_Education', 'Internet_Access', 'Performance']
        numeric_columns = ['Study_Hours', 'Attendance', 'Previous_Grade']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                date_formats = ['%d-%b', '%d-%B', '%b-%d', '%B-%d', '%d/%m', '%m/%d']
                date_converted = False
                
                for date_format in date_formats:
                    try:
                        parsed_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                        if parsed_dates.notna().sum() > 0:
                            df[col] = parsed_dates.dt.day
                            date_converted = True
                            break
                    except:
                        continue
                
                if not date_converted:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
        
        # Step 4: Categorical columns handling
        status_text.text("Step 4/5: Processing categorical columns...")
        progress_bar.progress(80)
        for col in categorical_columns:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        # Step 5: Final validation
        status_text.text("Step 5/5: Final validation...")
        progress_bar.progress(100)
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
        
        st.session_state.data_cleaned = True
        st.success("✅ Data cleaned successfully!")
        st.toast("✅ Data cleaning completed!")
        progress_bar.empty()
        status_text.empty()
    
    # Encode categorical variables
    @st.cache_data
    def encode_data(data):
        le_gender = LabelEncoder()
        le_parent = LabelEncoder()
        le_internet = LabelEncoder()
        le_performance = LabelEncoder()
        
        data_encoded = data.copy()
        data_encoded['Gender'] = le_gender.fit_transform(data['Gender'])
        data_encoded['Parent_Education'] = le_parent.fit_transform(data['Parent_Education'])
        data_encoded['Internet_Access'] = le_internet.fit_transform(data['Internet_Access'])
        data_encoded['Performance'] = le_performance.fit_transform(data['Performance'])
        
        return data_encoded, le_gender, le_parent, le_internet, le_performance

    try:
        df_encoded, le_gender, le_parent, le_internet, le_performance = encode_data(df)
        
        feature_columns = [col for col in df_encoded.columns if col != 'Performance']
        st.session_state.feature_columns = feature_columns
        
        X = df_encoded[feature_columns]
        y = df_encoded['Performance']
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Dashboard Page
        if page == "🏠 Dashboard":
            st.header("📊 Performance Overview Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", len(df), delta="Active")
            with col2:
                pass_rate = (df['Performance'] == 'Pass').sum() / len(df) * 100
                st.metric("Pass Rate", f"{pass_rate:.1f}%", delta=f"{pass_rate-50:.1f}%")
            with col3:
                if st.session_state.trained_models:
                    best_model = max(st.session_state.trained_models.items(), 
                                   key=lambda x: x[1]['accuracy'])
                    st.metric("Best Model Accuracy", f"{best_model[1]['accuracy']*100:.1f}%", 
                            delta=best_model[0])
                else:
                    st.metric("Models Trained", "0", delta="Train models")
            with col4:
                avg_study = df['Study_Hours'].mean()
                st.metric("Avg Study Hours", f"{avg_study:.1f}h", delta="Good")
            
            # Download cleaned data
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Cleaned Data",
                    data=csv,
                    file_name="cleaned_student_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                encoded_csv = df_encoded.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Encoded Data",
                    data=encoded_csv,
                    file_name="encoded_student_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Performance Distribution")
                performance_counts = df['Performance'].value_counts()
                fig = px.pie(
                    values=performance_counts.values,
                    names=performance_counts.index,
                    title='Student Performance Distribution',
                    color_discrete_sequence=['#667eea', '#f87171'],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📈 Study Hours vs Performance")
                fig = px.box(
                    df, 
                    x='Performance', 
                    y='Study_Hours',
                    color='Performance',
                    title='Study Hours Impact on Performance',
                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Attendance Analysis")
                fig = px.histogram(
                    df, 
                    x='Attendance', 
                    color='Performance',
                    marginal='box',
                    title='Attendance Distribution by Performance',
                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📚 Previous Grade Analysis")
                fig = px.violin(
                    df, 
                    x='Performance', 
                    y='Previous_Grade',
                    color='Performance',
                    title='Previous Grade Distribution',
                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
                    box=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📋 Student Data Overview")
            st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)

        # Model Training Page
        elif page == "📈 Model Training":
            st.header("🤖 Advanced Machine Learning Model Training")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("⚙️ Model Configuration")
                
                model_choice = st.selectbox(
                    "Select ML Algorithm",
                    ["Random Forest", "Logistic Regression", "Support Vector Machine", 
                     "K-Nearest Neighbors", "Gradient Boosting", "Naive Bayes", "Decision Tree"]
                )
                
                enable_tuning = st.checkbox("🎯 Enable Hyperparameter Tuning", value=False)
                
                if enable_tuning:
                    tuning_method = st.radio("Tuning Method", ["Grid Search", "Random Search"])
                    n_iter = st.slider("Number of iterations", 5, 50, 10) if tuning_method == "Random Search" else None
                
                test_size = st.slider("Test Set Size (%)", 10, 50, 30) / 100
                
                enable_feature_selection = st.checkbox("🎯 Enable Feature Selection", value=False)
                if enable_feature_selection:
                    n_features = st.slider("Number of features to select", 1, len(feature_columns), len(feature_columns))
                
                if st.button("🚀 Train Model", use_container_width=True):
                    with st.spinner(f"Training {model_choice}..."):
                        progress_bar = st.progress(0)
                        
                        # Feature selection
                        if enable_feature_selection:
                            selector = SelectKBest(f_classif, k=n_features)
                            X_selected = selector.fit_transform(X, y)
                            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
                            st.info(f"Selected features: {', '.join(selected_features)}")
                        else:
                            X_selected = X
                            selected_features = feature_columns
                        
                        progress_bar.progress(20)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_selected, y, test_size=test_size, random_state=42
                        )
                        
                        progress_bar.progress(30)
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        progress_bar.progress(40)
                        
                        # Model selection and training
                        if model_choice == "Random Forest":
                            if enable_tuning:
                                param_grid = {
                                    'n_estimators': [50, 100, 200],
                                    'max_depth': [10, 20, 30, None],
                                    'min_samples_split': [2, 5, 10]
                                }
                                base_model = RandomForestClassifier(random_state=42)
                            else:
                                model = RandomForestClassifier(n_estimators=100, random_state=42)
                        
                        elif model_choice == "Logistic Regression":
                            if enable_tuning:
                                param_grid = {
                                    'C': [0.001, 0.01, 0.1, 1, 10],
                                    'penalty': ['l2'],
                                    'solver': ['lbfgs', 'liblinear']
                                }
                                base_model = LogisticRegression(random_state=42, max_iter=1000)
                            else:
                                model = LogisticRegression(random_state=42, max_iter=1000)
                        
                        elif model_choice == "Support Vector Machine":
                            if enable_tuning:
                                param_grid = {
                                    'C': [0.1, 1, 10],
                                    'kernel': ['linear', 'rbf'],
                                    'gamma': ['scale', 'auto']
                                }
                                base_model = SVC(random_state=42, probability=True)
                            else:
                                model = SVC(random_state=42, probability=True)
                        
                        elif model_choice == "K-Nearest Neighbors":
                            if enable_tuning:
                                param_grid = {
                                    'n_neighbors': [3, 5, 7, 9, 11],
                                    'weights': ['uniform', 'distance'],
                                    'metric': ['euclidean', 'manhattan']
                                }
                                base_model = KNeighborsClassifier()
                            else:
                                model = KNeighborsClassifier(n_neighbors=5)
                        
                        elif model_choice == "Gradient Boosting":
                            if enable_tuning:
                                param_grid = {
                                    'n_estimators': [50, 100, 200],
                                    'learning_rate': [0.01, 0.1, 0.2],
                                    'max_depth': [3, 5, 7]
                                }
                                base_model = GradientBoostingClassifier(random_state=42)
                            else:
                                model = GradientBoostingClassifier(random_state=42)
                        
                        elif model_choice == "Naive Bayes":
                            model = GaussianNB()
                            enable_tuning = False
                        
                        elif model_choice == "Decision Tree":
                            if enable_tuning:
                                param_grid = {
                                    'max_depth': [5, 10, 20, None],
                                    'min_samples_split': [2, 5, 10],
                                    'criterion': ['gini', 'entropy']
                                }
                                base_model = DecisionTreeClassifier(random_state=42)
                            else:
                                model = DecisionTreeClassifier(random_state=42)
                        
                        progress_bar.progress(50)
                        
                        # Hyperparameter tuning
                        if enable_tuning and model_choice != "Naive Bayes":
                            if tuning_method == "Grid Search":
                                search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                            else:
                                search = RandomizedSearchCV(base_model, param_grid, cv=5, 
                                                          scoring='accuracy', n_iter=n_iter, random_state=42, n_jobs=-1)
                            
                            search.fit(X_train_scaled, y_train)
                            model = search.best_estimator_
                            best_params = search.best_params_
                            st.success(f"✅ Best parameters found: {best_params}")
                        else:
                            model.fit(X_train_scaled, y_train)
                            best_params = None
                        
                        progress_bar.progress(80)
                        
                        # Predictions
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                        
                        # Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        cm = confusion_matrix(y_test, y_pred)
                        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                        
                        progress_bar.progress(100)
                        
                        # Store model
                        st.session_state.trained_models[model_choice] = {
                            'model': model,
                            'scaler': scaler,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'cm': cm,
                            'features': selected_features,
                            'best_params': best_params,
                            'X_test': X_test_scaled,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba
                        }
                        
                        st.success(f"✅ {model_choice} trained successfully!")
                        st.balloons()
                        progress_bar.empty()
                        
                        # Save model option
                        model_bytes = io.BytesIO()
                        joblib.dump({'model': model, 'scaler': scaler, 'features': selected_features}, model_bytes)
                        model_bytes.seek(0)
                        
                        st.download_button(
                            label=f"💾 Download {model_choice} Model",
                            data=model_bytes,
                            file_name=f"{model_choice.lower().replace(' ', '_')}_model.pkl",
                            mime="application/octet-stream",
                            use_container_width=True
                        )
            
            with col2:
                st.subheader("📊 Model Comparison Dashboard")
                
                if st.session_state.trained_models:
                    # Create comparison dataframe
                    comparison_data = []
                    for model_name, model_data in st.session_state.trained_models.items():
                        comparison_data.append({
                            'Model': model_name,
                            'Accuracy': f"{model_data['accuracy']*100:.2f}%",
                            'Precision': f"{model_data['precision']*100:.2f}%",
                            'Recall': f"{model_data['recall']*100:.2f}%",
                            'F1-Score': f"{model_data['f1']*100:.2f}%"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Accuracy comparison chart
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y=[float(x.strip('%')) for x in comparison_df['Accuracy']],
                        title='Model Accuracy Comparison',
                        labels={'y': 'Accuracy (%)'},
                        color=[float(x.strip('%')) for x in comparison_df['Accuracy']],
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("👈 Train a model to see results here")
            
            # Detailed results for trained models
            if st.session_state.trained_models:
                st.markdown("---")
                st.subheader("📈 Detailed Model Results")
                
                tabs = st.tabs([name for name in st.session_state.trained_models.keys()])
                
                for idx, (model_name, model_data) in enumerate(st.session_state.trained_models.items()):
                    with tabs[idx]:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{model_data['accuracy']*100:.2f}%")
                            st.metric("Precision", f"{model_data['precision']*100:.2f}%")
                        
                        with col2:
                            st.metric("Recall", f"{model_data['recall']*100:.2f}%")
                            st.metric("F1-Score", f"{model_data['f1']*100:.2f}%")
                        
                        with col3:
                            if model_data['best_params']:
                                st.write("**Best Parameters:**")
                                for param, value in model_data['best_params'].items():
                                    st.write(f"• {param}: {value}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🔥 Confusion Matrix")
                            cm = model_data['cm']
                            fig = px.imshow(
                                cm,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Fail', 'Pass'],
                                y=['Fail', 'Pass'],
                                color_continuous_scale='Blues',
                                text_auto=True,
                                title=f'{model_name} Confusion Matrix'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("📊 Classification Report")
                            try:
                                report = classification_report(
                                    model_data['y_test'], 
                                    model_data['y_pred'],
                                    target_names=['Fail', 'Pass'],
                                    output_dict=True,
                                    zero_division=0
                                )
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                            except:
                                st.write("Classification report unavailable")
                        
                        # ROC Curve
                        if model_data['y_pred_proba'] is not None:
                            st.subheader("📈 ROC Curve")
                            fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'][:, 1])
                            roc_auc = auc(fpr, tpr)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                                    name=f'ROC Curve (AUC = {roc_auc:.2f})',
                                                    line=dict(color='#667eea', width=3)))
                            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                    name='Random Classifier',
                                                    line=dict(color='gray', width=2, dash='dash')))
                            fig.update_layout(
                                title=f'{model_name} ROC Curve',
                                xaxis_title='False Positive Rate',
                                yaxis_title='True Positive Rate',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Feature Importance
                        if hasattr(model_data['model'], 'feature_importances_'):
                            st.subheader("🌟 Feature Importance")
                            importance_df = pd.DataFrame({
                                'Feature': model_data['features'],
                                'Importance': model_data['model'].feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title=f'{model_name} Feature Importance',
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

        # Prediction Page
        elif page == "🔮 Prediction":
            st.header("🔮 Student Performance Prediction")
            
            if not st.session_state.trained_models:
                st.warning("⚠️ Please train at least one model first!")
                if st.button("Go to Model Training"):
                    st.session_state.page = "📈 Model Training"
                    st.rerun()
            else:
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    st.subheader("⚙️ Prediction Settings")
                    selected_model = st.selectbox(
                        "Select Model for Prediction",
                        list(st.session_state.trained_models.keys())
                    )
                    
                    model_info = st.session_state.trained_models[selected_model]
                    st.metric("Model Accuracy", f"{model_info['accuracy']*100:.2f}%")
                
                with col1:
                    st.markdown("### Enter Student Information")
                    
                    col1a, col1b, col1c = st.columns(3)
                    
                    with col1a:
                        gender = st.selectbox("Gender", df['Gender'].unique())
                        
                        study_min = int(df['Study_Hours'].min())
                        study_max = int(df['Study_Hours'].max())
                        study_mean = int(df['Study_Hours'].mean())
                        
                        if study_min == study_max:
                            st.info(f"Study Hours: {study_min} (constant)")
                            study_hours = study_min
                        else:
                            study_hours = st.slider("Study Hours per Day", study_min, study_max, study_mean)
                    
                    with col1b:
                        att_min = int(df['Attendance'].min())
                        att_max = int(df['Attendance'].max())
                        att_mean = int(df['Attendance'].mean())
                        
                        if att_min == att_max:
                            st.info(f"Attendance: {att_min}%")
                            attendance = att_min
                        else:
                            attendance = st.slider("Attendance %", att_min, att_max, att_mean)
                        
                        grade_min = int(df['Previous_Grade'].min())
                        grade_max = int(df['Previous_Grade'].max())
                        grade_mean = int(df['Previous_Grade'].mean())
                        
                        if grade_min == grade_max:
                            st.info(f"Previous Grade: {grade_min}")
                            previous_grade = grade_min
                        else:
                            previous_grade = st.slider("Previous Grade", grade_min, grade_max, grade_mean)
                    
                    with col1c:
                        parent_education = st.selectbox("Parent Education", df['Parent_Education'].unique())
                        internet_access = st.selectbox("Internet Access", df['Internet_Access'].unique())
                
                if st.button("🎯 Predict Performance", use_container_width=True):
                    # Encode inputs
                    gender_encoded = le_gender.transform([gender])[0]
                    parent_encoded = le_parent.transform([parent_education])[0]
                    internet_encoded = le_internet.transform([internet_access])[0]
                    
                    input_dict = {
                        'Gender': gender_encoded,
                        'Study_Hours': study_hours,
                        'Attendance': attendance,
                        'Previous_Grade': previous_grade,
                        'Parent_Education': parent_encoded,
                        'Internet_Access': internet_encoded
                    }
                    
                    # Use model's features
                    input_data = np.array([[input_dict[col] for col in model_info['features']]])
                    input_data_scaled = model_info['scaler'].transform(input_data)
                    
                    prediction = model_info['model'].predict(input_data_scaled)[0]
                    prediction_proba = model_info['model'].predict_proba(input_data_scaled)[0] if hasattr(model_info['model'], 'predict_proba') else [0.5, 0.5]
                    
                    result = le_performance.inverse_transform([prediction])[0]
                    
                    st.markdown("---")
                    st.markdown("### 🎊 Prediction Results")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        if result == 'Pass':
                            st.success(f"### ✅ Predicted Performance: PASS")
                            st.balloons()
                        else:
                            st.error(f"### ❌ Predicted Performance: FAIL")
                        
                        confidence = max(prediction_proba) * 100
                        st.markdown(f"**Confidence:** {confidence:.2f}%")
                        st.markdown(f"**Model Used:** {selected_model}")
                        
                        # Confidence visualization
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=prediction_proba,
                            y=['Fail', 'Pass'],
                            orientation='h',
                            marker=dict(color=['#f87171', '#667eea']),
                            text=[f'{p*100:.1f}%' for p in prediction_proba],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title='Prediction Confidence',
                            xaxis_title='Probability',
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations
                        st.markdown("### 💡 Recommendations")
                        if result == 'Fail':
                            recommendations = []
                            if study_hours < df[df['Performance']=='Pass']['Study_Hours'].mean():
                                target_hours = df[df['Performance']=='Pass']['Study_Hours'].mean()
                                recommendations.append(f"📚 Increase study hours to at least {target_hours:.1f} hours per day")
                            if attendance < df[df['Performance']=='Pass']['Attendance'].mean():
                                target_att = df[df['Performance']=='Pass']['Attendance'].mean()
                                recommendations.append(f"📅 Improve attendance to above {target_att:.1f}%")
                            if previous_grade < df[df['Performance']=='Pass']['Previous_Grade'].mean():
                                recommendations.append("📖 Focus on strengthening foundational concepts")
                            
                            for rec in recommendations:
                                st.warning(rec)
                        else:
                            st.success("🎉 Great! Keep up the excellent work!")
                            st.info("💪 Continue maintaining your study habits and attendance")

        # Data Analysis Page
        elif page == "📊 Data Analysis":
            st.header("📊 Advanced Data Analysis")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Correlations", "🎯 Distributions", "🔍 Insights", "📉 Trends"])
            
            with tab1:
                st.subheader("Correlation Analysis")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    correlation = df_encoded.corr()
                    fig = px.imshow(
                        correlation,
                        labels=dict(color="Correlation"),
                        x=correlation.columns,
                        y=correlation.columns,
                        color_continuous_scale='RdBu',
                        zmin=-1, zmax=1,
                        title='Feature Correlation Heatmap',
                        text_auto='.2f'
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("### Key Correlations")
                    perf_corr = correlation['Performance'].abs().sort_values(ascending=False)[1:]
                    st.dataframe(perf_corr.to_frame('Correlation'), use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Attendance Distribution")
                    fig = px.histogram(
                        df, 
                        x='Attendance', 
                        color='Performance',
                        marginal='violin',
                        title='Attendance Impact on Performance',
                        color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
                        nbins=20
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Previous Grade Distribution")
                    fig = px.histogram(
                        df, 
                        x='Previous_Grade', 
                        color='Performance',
                        marginal='box',
                        title='Previous Grade Impact',
                        color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
                        nbins=20
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Study Hours Analysis")
                fig = px.scatter(
                    df,
                    x='Study_Hours',
                    y='Previous_Grade',
                    color='Performance',
                    size='Attendance',
                    hover_data=['Gender', 'Parent_Education'],
                    title='Multi-dimensional Performance Analysis',
                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("📌 Statistical Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_study_pass = df[df['Performance']=='Pass']['Study_Hours'].mean()
                    avg_study_fail = df[df['Performance']=='Fail']['Study_Hours'].mean()
                    avg_att_pass = df[df['Performance']=='Pass']['Attendance'].mean()
                    avg_att_fail = df[df['Performance']=='Fail']['Attendance'].mean()
                    avg_grade_pass = df[df['Performance']=='Pass']['Previous_Grade'].mean()
                    avg_grade_fail = df[df['Performance']=='Fail']['Previous_Grade'].mean()
                    
                    st.info(f"""
                    **Academic Performance Insights:**
                    
                    📚 **Study Hours:**
                    - Pass: {avg_study_pass:.1f} hours
                    - Fail: {avg_study_fail:.1f} hours
                    - Difference: {avg_study_pass - avg_study_fail:.1f} hours
                    
                    📅 **Attendance:**
                    - Pass: {avg_att_pass:.1f}%
                    - Fail: {avg_att_fail:.1f}%
                    - Difference: {avg_att_pass - avg_att_fail:.1f}%
                    
                    📖 **Previous Grades:**
                    - Pass: {avg_grade_pass:.1f}
                    - Fail: {avg_grade_fail:.1f}
                    - Difference: {avg_grade_pass - avg_grade_fail:.1f}
                    """)
                
                with col2:
                    # Gender performance
                    gender_perf = pd.crosstab(df['Gender'], df['Performance'], normalize='index') * 100
                    fig = px.bar(
                        gender_perf,
                        title='Performance by Gender',
                        labels={'value': 'Percentage (%)', 'variable': 'Performance'},
                        color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
                        barmode='group'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Parent education impact
                    parent_perf = pd.crosstab(df['Parent_Education'], df['Performance'], normalize='index') * 100
                    fig = px.bar(
                        parent_perf,
                        title='Performance by Parent Education',
                        labels={'value': 'Percentage (%)', 'variable': 'Performance'},
                        color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
                        barmode='group'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Internet access impact
                    internet_perf = pd.crosstab(df['Internet_Access'], df['Performance'], normalize='index') * 100
                    fig = px.bar(
                        internet_perf,
                        title='Performance by Internet Access',
                        labels={'value': 'Percentage (%)', 'variable': 'Performance'},
                        color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'},
                        barmode='group'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("📉 Performance Trends")
                
                # Create synthetic trend data
                metrics_df = pd.DataFrame({
                    'Metric': ['Study Hours', 'Attendance', 'Previous Grade'] * 2,
                    'Performance': ['Pass'] * 3 + ['Fail'] * 3,
                    'Average': [
                        df[df['Performance']=='Pass']['Study_Hours'].mean(),
                        df[df['Performance']=='Pass']['Attendance'].mean(),
                        df[df['Performance']=='Pass']['Previous_Grade'].mean(),
                        df[df['Performance']=='Fail']['Study_Hours'].mean(),
                        df[df['Performance']=='Fail']['Attendance'].mean(),
                        df[df['Performance']=='Fail']['Previous_Grade'].mean()
                    ]
                })
                
                fig = px.line(
                    metrics_df,
                    x='Metric',
                    y='Average',
                    color='Performance',
                    markers=True,
                    title='Average Metrics Comparison',
                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Sunburst chart
                st.subheader("🌅 Hierarchical Performance Analysis")
                sunburst_df = df.groupby(['Gender', 'Parent_Education', 'Performance']).size().reset_index(name='Count')
                fig = px.sunburst(
                    sunburst_df,
                    path=['Gender', 'Parent_Education', 'Performance'],
                    values='Count',
                    title='Hierarchical Performance Breakdown',
                    color='Performance',
                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

        # AI Explainability Page
        elif page == "🧠 AI Explainability":
            st.header("🧠 AI Model Explainability")
            
            if not st.session_state.trained_models:
                st.warning("⚠️ Please train at least one model first!")
            else:
                st.info("💡 **SHAP Values** help explain which features most influenced each prediction")
                
                selected_model = st.selectbox(
                    "Select Model to Explain",
                    [name for name in st.session_state.trained_models.keys() 
                     if hasattr(st.session_state.trained_models[name]['model'], 'feature_importances_')]
                )
                
                if selected_model:
                    model_info = st.session_state.trained_models[selected_model]
                    
                    st.subheader(f"📊 {selected_model} Feature Importance")
                    
                    if hasattr(model_info['model'], 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': model_info['features'],
                            'Importance': model_info['model'].feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig = px.treemap(
                                importance_df,
                                path=['Feature'],
                                values='Importance',
                                title=f'{selected_model} Feature Importance (Treemap)',
                                color='Importance',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.dataframe(
                                importance_df.style.background_gradient(cmap='Viridis', subset=['Importance']),
                                use_container_width=True,
                                height=500
                            )
                        
                        st.markdown("---")
                        
                        # Feature impact analysis
                        st.subheader("🎯 Feature Impact Analysis")
                        
                        feature_to_analyze = st.selectbox("Select Feature to Analyze", model_info['features'])
                        
                        if feature_to_analyze in df.columns:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = px.box(
                                    df,
                                    x='Performance',
                                    y=feature_to_analyze,
                                    color='Performance',
                                    title=f'{feature_to_analyze} Distribution by Performance',
                                    color_discrete_map={'Pass': '#667eea', 'Fail': '#f87171'}
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Statistics
                                st.write(f"**{feature_to_analyze} Statistics:**")
                                pass_mean = df[df['Performance']=='Pass'][feature_to_analyze].mean()
                                fail_mean = df[df['Performance']=='Fail'][feature_to_analyze].mean()
                                
                                st.metric("Pass Average", f"{pass_mean:.2f}")
                                st.metric("Fail Average", f"{fail_mean:.2f}")
                                st.metric("Difference", f"{pass_mean - fail_mean:.2f}")
                    else:
                        st.warning("This model doesn't support feature importance visualization")

        # Student Clustering Page
        elif page == "👥 Student Clustering":
            st.header("👥 Student Segmentation & Clustering")
            
            st.info("💡 Use K-Means clustering to identify different student groups based on their characteristics")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("⚙️ Clustering Settings")
                n_clusters = st.slider("Number of Clusters", 2, 6, 3)
                
                features_for_clustering = st.multiselect(
                    "Select Features for Clustering",
                    feature_columns,
                    default=feature_columns[:3]
                )
                
                if st.button("🎯 Perform Clustering", use_container_width=True):
                    if len(features_for_clustering) < 2:
                        st.error("Please select at least 2 features")
                    else:
                        with st.spinner("Performing clustering..."):
                            # Prepare data
                            X_cluster = df_encoded[features_for_clustering]
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_cluster)
                            
                            # Perform clustering
                            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(X_scaled)
                            
                            # Add clusters to dataframe
                            df['Cluster'] = clusters
                            df['Cluster'] = df['Cluster'].apply(lambda x: f"Group {x+1}")
                            
                            st.session_state.clusters = clusters
                            st.session_state.cluster_centers = kmeans.cluster_centers_
                            st.session_state.clustering_features = features_for_clustering
                            
                            st.success("✅ Clustering completed!")
            
            with col2:
                if 'clusters' in st.session_state:
                    st.subheader("📊 Clustering Results")
                    
                    # Cluster distribution
                    cluster_counts = df['Cluster'].value_counts()
                    fig = px.pie(
                        values=cluster_counts.values,
                        names=cluster_counts.index,
                        title='Student Distribution by Cluster',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("👈 Configure and run clustering to see results")
            
            if 'clusters' in st.session_state:
                st.markdown("---")
                
                # Visualization
                if len(st.session_state.clustering_features) >= 2:
                    st.subheader("🎨 Cluster Visualization")
                    
                    feat1 = st.session_state.clustering_features[0]
                    feat2 = st.session_state.clustering_features[1]
                    
                    fig = px.scatter(
                        df,
                        x=feat1,
                        y=feat2,
                        color='Cluster',
                        symbol='Performance',
                        title=f'Student Clusters: {feat1} vs {feat2}',
                        hover_data=['Study_Hours', 'Attendance', 'Previous_Grade']
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster characteristics
                st.subheader("📋 Cluster Characteristics")
                
                cluster_stats = df.groupby('Cluster').agg({
                    'Study_Hours': 'mean',
                    'Attendance': 'mean',
                    'Previous_Grade': 'mean',
                    'Performance': lambda x: (x == 'Pass').sum() / len(x) * 100
                }).round(2)
                
                cluster_stats.columns = ['Avg Study Hours', 'Avg Attendance', 'Avg Previous Grade', 'Pass Rate (%)']
                st.dataframe(cluster_stats, use_container_width=True)
                
                # Cluster profiles
                st.subheader("👤 Cluster Profiles")
                
                tabs = st.tabs([f"Group {i+1}" for i in range(n_clusters)])
                
                for idx, tab in enumerate(tabs):
                    with tab:
                        cluster_name = f"Group {idx+1}"
                        cluster_data = df[df['Cluster'] == cluster_name]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Students", len(cluster_data))
                        with col2:
                            pass_rate = (cluster_data['Performance'] == 'Pass').sum() / len(cluster_data) * 100
                            st.metric("Pass Rate", f"{pass_rate:.1f}%")
                        with col3:
                            st.metric("Avg Study", f"{cluster_data['Study_Hours'].mean():.1f}h")
                        with col4:
                            st.metric("Avg Attendance", f"{cluster_data['Attendance'].mean():.1f}%")
                        
                        st.markdown("**Cluster Description:**")
                        if pass_rate > 70:
                            st.success("🌟 **High Achievers**: This group shows excellent performance with strong study habits.")
                        elif pass_rate > 40:
                            st.info("📚 **Moderate Performers**: This group has potential for improvement with targeted support.")
                        else:
                            st.warning("⚠️ **At-Risk Students**: This group needs immediate intervention and support.")

        # Advanced Settings Page
        elif page == "⚙️ Advanced Settings":
            st.header("⚙️ Advanced Settings & Model Management")
            
            tab1, tab2, tab3 = st.tabs(["💾 Model Management", "📊 Data Export", "🔧 System Info"])
            
            with tab1:
                st.subheader("💾 Model Management")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Save Models")
                    if st.session_state.trained_models:
                        for model_name, model_data in st.session_state.trained_models.items():
                            model_bytes = io.BytesIO()
                            joblib.dump({
                                'model': model_data['model'],
                                'scaler': model_data['scaler'],
                                'features': model_data['features']
                            }, model_bytes)
                            model_bytes.seek(0)
                            
                            st.download_button(
                                label=f"💾 Download {model_name}",
                                data=model_bytes,
                                file_name=f"{model_name.lower().replace(' ', '_')}_model.pkl",
                                mime="application/octet-stream",
                                use_container_width=True
                            )
                    else:
                        st.info("No trained models available")
                
                with col2:
                    st.markdown("### Load Model")
                    uploaded_model = st.file_uploader("Upload trained model (.pkl)", type=['pkl'])
                    
                    if uploaded_model:
                        try:
                            model_data = joblib.load(uploaded_model)
                            st.success("✅ Model loaded successfully!")
                            st.write("**Model Info:**")
                            st.write(f"- Features: {', '.join(model_data['features'])}")
                        except Exception as e:
                            st.error(f"Error loading model: {str(e)}")
                
                st.markdown("---")
                
                if st.session_state.trained_models:
                    st.markdown("### Clear Models")
                    if st.button("🗑️ Clear All Trained Models", use_container_width=True):
                        st.session_state.trained_models = {}
                        gc.collect()
                        st.success("✅ All models cleared!")
                        st.rerun()
            
            with tab2:
                st.subheader("📊 Data Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Export Data")
                    
                    # Original data
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Original Data",
                        data=csv,
                        file_name="student_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Encoded data
                    encoded_csv = df_encoded.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Encoded Data",
                        data=encoded_csv,
                        file_name="encoded_student_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Cluster data (if available)
                    if 'Cluster' in df.columns:
                        cluster_csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Clustered Data",
                            data=cluster_csv,
                            file_name="clustered_student_data.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
                with col2:
                    st.markdown("### Export Reports")
                    
                    if st.session_state.trained_models:
                        # Create comprehensive report
                        report_data = []
                        for model_name, model_data in st.session_state.trained_models.items():
                            report_data.append({
                                'Model': model_name,
                                'Accuracy': f"{model_data['accuracy']*100:.2f}%",
                                'Precision': f"{model_data['precision']*100:.2f}%",
                                'Recall': f"{model_data['recall']*100:.2f}%",
                                'F1-Score': f"{model_data['f1']*100:.2f}%"
                            })
                        
                        report_df = pd.DataFrame(report_data)
                        report_csv = report_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="📊 Download Model Report",
                            data=report_csv,
                            file_name="model_comparison_report.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.info("Train models first to generate reports")
            
            with tab3:
                st.subheader("🔧 System Information")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Dataset Info")
                    st.write(f"**Total Records:** {len(df)}")
                    st.write(f"**Features:** {len(feature_columns)}")
                    st.write(f"**Pass Count:** {(df['Performance'] == 'Pass').sum()}")
                    st.write(f"**Fail Count:** {(df['Performance'] == 'Fail').sum()}")
                    st.write(f"**Data Quality:** {(df.isna().sum().sum() == 0) and '✅ Clean' or '⚠️ Contains missing values'}")
                
                with col2:
                    st.markdown("### Model Info")
                    st.write(f"**Trained Models:** {len(st.session_state.trained_models)}")
                    if st.session_state.trained_models:
                        best_model = max(st.session_state.trained_models.items(), 
                                       key=lambda x: x[1]['accuracy'])
                        st.write(f"**Best Model:** {best_model[0]}")
                        st.write(f"**Best Accuracy:** {best_model[1]['accuracy']*100:.2f}%")
                
                st.markdown("---")
                
                st.markdown("### Feature Statistics")
                stats_df = df[['Study_Hours', 'Attendance', 'Previous_Grade']].describe().T
                st.dataframe(stats_df, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### Session State")
                with st.expander("🔍 View Session State"):
                    st.write({
                        'data_loaded': st.session_state.data_loaded,
                        'models_trained': len(st.session_state.trained_models),
                        'dark_mode': st.session_state.dark_mode,
                        'clusters_created': 'clusters' in st.session_state
                    })

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.write("Please check your data and try again.")
        
        with st.expander("🔍 Debug Information"):
            st.write("**Available columns:**", list(df.columns))
            st.write("**Data types:**")
            st.write(df.dtypes)
            st.write("**First few rows:**")
            st.write(df.head())
            st.write("**Error Details:**")
            import traceback
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px; font-family: "Poppins", sans-serif;'>
        <p style='font-size: 1.1rem; font-weight: 600;'><strong>Advanced Student Performance Prediction System</strong></p>
        <p style='font-size: 1.1rem; font-weight: 600;'>Developed by Rashid Ali Soomro</p>
    </div>
    """, unsafe_allow_html=True) 


