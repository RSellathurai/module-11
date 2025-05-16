# Used Car Price Analysis and Regression Modeling
# Following CRISP-DM Framework

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set the aesthetic style of the plots
plt.style.use('ggplot')
sns.set_style('whitegrid')

# 1. Business Understanding
# The goal is to understand what factors influence used car prices
# This can help buyers and sellers make better decisions

print("="*80)
print("CRISP-DM PHASE 1: BUSINESS UNDERSTANDING")
print("="*80)
print("Objective: Determine which factors most significantly affect used car prices")
print("Key questions to answer:")
print("1. Which features most strongly influence car prices?")
print("2. How do different manufacturers, conditions, and years affect pricing?")
print("3. Which model provides the most accurate predictions of car prices?")
print("4. What insights can we derive for buyers and sellers in the used car market?")
print("\n")

# 2. Data Understanding
print("="*80)
print("CRISP-DM PHASE 2: DATA UNDERSTANDING")
print("="*80)

# Load both datasets
df1 = pd.read_csv('vehicles_1.csv')
df2 = pd.read_csv('vehicles_2.csv')

# Display basic information about the datasets
print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)

# Combine the datasets
print("\nCombining the datasets...")
df = pd.concat([df1, df2], ignore_index=True)
print("Combined dataset shape:", df.shape)

# Check for duplicates based on id
duplicate_count = df['id'].duplicated().sum()
print(f"Number of duplicate IDs: {duplicate_count}")

# Remove duplicates if any
if duplicate_count > 0:
    df = df.drop_duplicates(subset=['id'])
    print(f"After removing duplicates, dataset shape: {df.shape}")

# Summary statistics
print("\nSummary statistics for numeric columns:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage.round(2)
})
print(missing_info[missing_info['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# Distribution of categorical variables
print("\nDistribution of manufacturers:")
print(df['manufacturer'].value_counts().head(10))

print("\nDistribution of condition:")
print(df['condition'].value_counts())

print("\nDistribution of fuel types:")
print(df['fuel'].value_counts())

print("\nDistribution of transmission types:")
print(df['transmission'].value_counts())

# Visualize the price distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of Car Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: ${df["price"].mean():.2f}')
plt.axvline(df['price'].median(), color='green', linestyle='--', label=f'Median: ${df["price"].median():.2f}')
plt.legend()
plt.savefig('price_distribution.png')
plt.close()

# Visualize relationship between price and year
plt.figure(figsize=(12, 6))
sns.boxplot(x='year', y='price', data=df.dropna(subset=['year']).sample(10000))
plt.title('Price Distribution by Year')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.xticks(rotation=90)
plt.savefig('price_by_year.png')
plt.close()

# Visualize relationship between price and manufacturer
plt.figure(figsize=(14, 8))
top_manufacturers = df['manufacturer'].value_counts().head(15).index
manufacturer_data = df[df['manufacturer'].isin(top_manufacturers)]
sns.boxplot(x='manufacturer', y='price', data=manufacturer_data.sample(min(10000, len(manufacturer_data))))
plt.title('Price Distribution by Manufacturer')
plt.xlabel('Manufacturer')
plt.ylabel('Price ($)')
plt.xticks(rotation=90)
plt.savefig('price_by_manufacturer.png')
plt.close()

# Correlation matrix for numeric variables
numeric_columns = df.select_dtypes(include=['number']).columns
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Variables')
plt.savefig('correlation_matrix.png')
plt.close()

# 3. Data Preparation
print("="*80)
print("CRISP-DM PHASE 3: DATA PREPARATION")
print("="*80)

# Remove outliers in price
q1 = df['price'].quantile(0.01)
q3 = df['price'].quantile(0.99)
iqr = q3 - q1
price_lower_bound = q1 - 1.5 * iqr
price_upper_bound = q3 + 1.5 * iqr

# Filter the dataset
df_filtered = df[(df['price'] >= price_lower_bound) & (df['price'] <= price_upper_bound)]
print(f"Dataset shape after removing price outliers: {df_filtered.shape}")

# Filter invalid years (assuming cars from 1900 to present)
current_year = 2025
df_filtered = df_filtered[(df_filtered['year'] >= 1900) & (df_filtered['year'] <= current_year)]
print(f"Dataset shape after filtering invalid years: {df_filtered.shape}")

# Handle missing values
# For categorical columns, fill with 'unknown'
categorical_columns = ['manufacturer', 'model', 'condition', 'cylinders', 'fuel', 
                       'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color']
for col in categorical_columns:
    df_filtered[col] = df_filtered[col].fillna('unknown')

# For odometer, fill with median
df_filtered['odometer'] = df_filtered['odometer'].fillna(df_filtered['odometer'].median())

# Convert year to age
df_filtered['age'] = current_year - df_filtered['year']

# Create price categories for visualization and analysis
df_filtered['price_category'] = pd.cut(df_filtered['price'], 
                                      bins=[0, 5000, 10000, 20000, 50000, 100000, float('inf')],
                                      labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Luxury'])

# Feature engineering
# Group less frequent manufacturers as 'Other'
top_manufacturers = df_filtered['manufacturer'].value_counts().head(20).index
df_filtered['manufacturer_grouped'] = df_filtered['manufacturer'].apply(
    lambda x: x if x in top_manufacturers else 'Other')

# Create mileage categories
df_filtered['mileage_category'] = pd.cut(df_filtered['odometer'], 
                                        bins=[0, 20000, 50000, 100000, 150000, float('inf')],
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Split the data into features and target
X = df_filtered.drop(['price', 'price_category', 'id', 'VIN', 'model', 'region', 'state'], axis=1)
y = df_filtered['price']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("\nFeatures for modeling:")
print(f"Categorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")

# Create train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# 4. Modeling
print("="*80)
print("CRISP-DM PHASE 4: MODELING")
print("="*80)

# Preprocessing for numerical and categorical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor.XGBRegressor(random_state=42)
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor):
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions on test data
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV R² (mean)': cv_scores.mean(),
        'CV R² (std)': cv_scores.std()
    }

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test, preprocessor)

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)

# Find the best model based on test R²
best_model_name = results_df['R²'].idxmax()
print(f"\nBest performing model: {best_model_name} with R² = {results_df.loc[best_model_name, 'R²']:.4f}")

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning for the best model...")

if best_model_name == 'Linear Regression':
    param_grid = {'model__fit_intercept': [True, False]}
    
elif best_model_name == 'Ridge Regression':
    param_grid = {
        'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        'model__solver': ['auto', 'svd', 'cholesky']
    }
    
elif best_model_name == 'Lasso Regression':
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'model__selection': ['cyclic', 'random']
    }
    
elif best_model_name == 'ElasticNet':
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5]
    }
    
elif best_model_name == 'XGBoost':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5],
        'model__colsample_bytree': [0.7, 0.8, 0.9]
    }

# Create the best model pipeline
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Perform grid search with cross-validation
grid_search = GridSearchCV(
    best_pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model on the test set
tuned_y_pred = grid_search.predict(X_test)
tuned_r2 = r2_score(y_test, tuned_y_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_y_pred))
tuned_mae = mean_absolute_error(y_test, tuned_y_pred)

print("\nTuned Model Performance:")
print(f"R²: {tuned_r2:.4f}")
print(f"RMSE: {tuned_rmse:.2f}")
print(f"MAE: {tuned_mae:.2f}")

# 5. Evaluation
print("="*80)
print("CRISP-DM PHASE 5: EVALUATION")
print("="*80)

# Compare the performance of all models with the tuned best model
final_results = results_df.copy()
final_results.loc['Tuned ' + best_model_name] = [tuned_r2, tuned_rmse, tuned_mae, grid_search.best_score_, 0]

print("\nFinal Model Comparison:")
print(final_results.sort_values('R²', ascending=False))

# Plot model comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=final_results.index, y=final_results['R²'])
plt.title('Model Comparison - R² Score')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# For the best model, analyze feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    # For tree-based models, we can get feature importance directly
    best_model_fitted = grid_search.best_estimator_.named_steps['model']
    
    # Get feature names from the preprocessor
    preprocessor_fitted = grid_search.best_estimator_.named_steps['preprocessor']
    
    # For categorical features, OneHotEncoder creates multiple columns
    ohe = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    
    # Numerical features keep their names
    num_feature_names = numerical_cols
    
    # Combine all feature names in the right order
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    # Get feature importances
    if best_model_name == 'XGBoost':
        importances = best_model_fitted.feature_importances_
    else:
        importances = best_model_fitted.feature_importances_
    
    # Sort importances
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = np.array(all_feature_names)[sorted_indices]
    
    # Plot top 20 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x=sorted_importances[:20], y=sorted_features[:20])
    plt.title(f'Top 20 Important Features ({best_model_name})')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nTop 10 Important Features:")
    for i in range(10):
        print(f"{sorted_features[i]}: {sorted_importances[i]:.4f}")
        
elif best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
    # For linear models, we can get coefficients
    best_model_fitted = grid_search.best_estimator_.named_steps['model']
    
    # Get feature names from the preprocessor
    preprocessor_fitted = grid_search.best_estimator_.named_steps['preprocessor']
    
    # For categorical features, OneHotEncoder creates multiple columns
    ohe = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    
    # Numerical features keep their names
    num_feature_names = numerical_cols
    
    # Combine all feature names in the right order
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    # Get coefficients
    coefficients = best_model_fitted.coef_
    
    # Sort coefficients by absolute value
    sorted_indices = np.argsort(np.abs(coefficients))[::-1]
    sorted_coefficients = coefficients[sorted_indices]
    sorted_features = np.array(all_feature_names)[sorted_indices]
    
    # Plot top 20 coefficients
    plt.figure(figsize=(12, 8))
    colors = ['red' if coef < 0 else 'blue' for coef in sorted_coefficients[:20]]
    sns.barplot(x=sorted_coefficients[:20], y=sorted_features[:20], palette=colors)
    plt.title(f'Top 20 Feature Coefficients ({best_model_name})')
    plt.xlabel('Coefficient')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_coefficients.png')
    plt.close()
    
    print("\nTop 10 Important Features (by coefficient magnitude):")
    for i in range(10):
        print(f"{sorted_features[i]}: {sorted_coefficients[i]:.4f}")

# Analyze residuals
plt.figure(figsize=(12, 6))
residuals = y_test - tuned_y_pred
sns.histplot(residuals, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title('Residuals Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.savefig('residuals_distribution.png')
plt.close()

# Plot predicted vs actual values
plt.figure(figsize=(10, 10))
sns.scatterplot(x=y_test, y=tuned_y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# 6. Deployment
print("="*80)
print("CRISP-DM PHASE 6: DEPLOYMENT")
print("="*80)
print("Summary of Findings and Recommendations:")

# Summarize the model performance
print(f"\n1. Model Performance:")
print(f"   - Best model: {best_model_name} with tuned parameters")
print(f"   - R² Score: {tuned_r2:.4f} (explains {tuned_r2*100:.1f}% of price variation)")
print(f"   - RMSE: ${tuned_rmse:.2f}")
print(f"   - MAE: ${tuned_mae:.2f}")

# Summarize key price drivers
print("\n2. Key Factors Affecting Used Car Prices:")
if 'sorted_features' in locals():
    for i in range(min(5, len(sorted_features))):
        if best_model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            effect = "positive" if sorted_coefficients[i] > 0 else "negative"
            print(f"   - {sorted_features[i]}: {effect} impact (coefficient: {sorted_coefficients[i]:.4f})")
        else:
            print(f"   - {sorted_features[i]}: importance score of {sorted_importances[i]:.4f}")

# Add business recommendations
print("\n3. Business Recommendations:")
print("   - For sellers: Focus on highlighting the most influential positive features in listings")
print("   - For buyers: Be aware of the key factors driving prices to negotiate better deals")
print("   - For dealers: Use the model to identify underpriced or overpriced vehicles in the market")
print("   - For platform owners: Implement price recommendations based on the model to help users")

# Example function to make predictions on new data
print("\n4. Model Deployment Example:")
print("""
# Function to make predictions on new car data
def predict_car_price(new_car_data, model=grid_search):
    # Ensure new_car_data has the right format
    # Make prediction
    predicted_price = model.predict(new_car_data)
    return predicted_price[0]

# Example usage:
# predict_car_price(pd.DataFrame({
#    'year': [2018],
#    'manufacturer': ['toyota'],
#    'condition': ['good'],
#    'odometer': [50000],
#    ...
# }))
""")

print("\nAnalysis complete!")
