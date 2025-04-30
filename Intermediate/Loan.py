# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
def load_data(file_path):
    """Load the dataset from CSV file"""
    return pd.read_csv(file_path)

# Exploratory Data Analysis
def perform_eda(df):
    """Perform exploratory data analysis"""
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    print("Dataset Information:")
    print(df.info())
    
    print("\nSummary Statistics:")
    print(df.describe(include='all'))
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Visualizations
    plt.figure(figsize=(15, 10))
    
    # Target variable distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='loan_status', data=df)
    plt.title('Loan Status Distribution')
    
    # Numerical features distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df['income_annum'], kde=True)
    plt.title('Annual Income Distribution')
    
    plt.subplot(2, 2, 3)
    sns.histplot(df['loan_amount'], kde=True)
    plt.title('Loan Amount Distribution')
    
    plt.subplot(2, 2, 4)
    sns.histplot(df['cibil_score'], kde=True)
    plt.title('CIBIL Score Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Categorical features
    cat_cols = ['education', 'self_employed']
    
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(1, 2, i)
        sns.countplot(x=col, hue='loan_status', data=df)
        plt.title(f'{col} vs Loan Status')
    plt.tight_layout()
    plt.show()

# Data Preprocessing
def preprocess_data(df):
    """Preprocess the data for modeling"""
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Check unique values in loan_status
    print("\nUnique values in loan_status:", df['loan_status'].unique())
    
    # Convert target variable to binary (1 for Approved, 0 for Rejected)
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Verify no NaN values in target
    print("\nNaN values in loan_status after mapping:", df['loan_status'].isna().sum())
    
    # Drop rows where loan_status couldn't be mapped (if any)
    df = df.dropna(subset=['loan_status'])
    
    # Feature engineering
    df['total_assets'] = (df['residential_assets_value'] + 
                         df['commercial_assets_value'] + 
                         df['luxury_assets_value'] + 
                         df['bank_asset_value'])
    
    df['asset_to_loan_ratio'] = df['total_assets'] / df['loan_amount']
    
    # Replace inf values with large number
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    
    # Drop unnecessary columns
    df.drop(['loan_id'], axis=1, inplace=True)
    
    return df

# Feature Selection
def select_features(df):
    """Select features for modeling"""
    # Define categorical and numerical features
    categorical_features = ['education', 'self_employed']
    numerical_features = ['no_of_dependents', 'income_annum', 'loan_amount', 
                         'loan_term', 'cibil_score', 'residential_assets_value',
                         'commercial_assets_value', 'luxury_assets_value',
                         'bank_asset_value', 'total_assets', 'asset_to_loan_ratio']
    
    return categorical_features, numerical_features

# Build Pipeline
def build_pipeline(categorical_features, numerical_features):
    """Build the preprocessing and modeling pipeline"""
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])
    
    return preprocessor

# Train and Evaluate Models
def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    """Train and evaluate different models"""
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'report': report,
            'confusion_matrix': cm,
            'model': pipeline
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
    
    return results

# Hyperparameter Tuning
def tune_model(X_train, y_train, preprocessor):
    """Perform hyperparameter tuning for the best model"""
    # We'll tune Random Forest as it often performs well
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(random_state=42))])
    
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    print("Best ROC AUC score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

# Main function
def main():
    # Load data
    data_path = 'loan_approval_dataset.csv'  # Replace with your dataset path
    df = load_data(data_path)
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Verify no NaN values in target
    print("\nNaN values in target after preprocessing:", df['loan_status'].isna().sum())
    
    # Split data into features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Verify shapes
    print("\nShapes before train_test_split:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Select features
    categorical_features, numerical_features = select_features(df)
    
    # Build preprocessing pipeline
    preprocessor = build_pipeline(categorical_features, numerical_features)
    
    # Train and evaluate models
    results = train_and_evaluate(X_train_res, X_test, y_train_res, y_test, preprocessor)
    
    # Tune the best model
    best_model = tune_model(X_train_res, y_train_res, preprocessor)
    
    # Evaluate tuned model
    y_pred_tuned = best_model.predict(X_test)
    y_proba_tuned = best_model.predict_proba(X_test)[:, 1]
    
    print("\nTuned Model Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba_tuned):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_tuned))
    
    # Save the best model
    joblib.dump(best_model, 'loan_approval_model.pkl')
    print("\nModel saved as 'loan_approval_model.pkl'")

if __name__ == "__main__":
    main()