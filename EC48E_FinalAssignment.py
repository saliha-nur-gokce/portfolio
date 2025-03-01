import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import statsmodels.api as sm
import shap

# ----------------------------------------------------------------------
# 1) Data
# ----------------------------------------------------------------------
def load_data(file_path):
    return pd.read_excel(file_path)

# ----------------------------------------------------------------------
# 2) "1970 Q1" -> year, quarter
# ----------------------------------------------------------------------
def parse_quarter_column(df, date_col_name='observation_date'):
    df[['year','quarter']] = df[date_col_name].str.extract(r'(\d{4}) Q(\d)')
    df['year'] = df['year'].astype(int)
    df['quarter'] = df['quarter'].astype(int)
    return df

# ----------------------------------------------------------------------
# 3) Preprocess (RF, GBM)
# ----------------------------------------------------------------------
def preprocess_data(df, target_column, columns_to_drop=None, pca_components=None):
    if columns_to_drop:
        df = df.drop(df.columns[columns_to_drop], axis=1)

    
    if 'GDP growth annual' in df.columns:
        df.drop(columns=['GDP growth annual'], inplace=True)

    
    for col in df.columns:
        if col != target_column:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col].str.extract(r'(\-?\d+\.?\d*)', expand=False))
                except (ValueError, AttributeError):
                    df.drop(columns=[col], inplace=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    if pca_components:
        numeric_transformer.steps.append(('pca', PCA(n_components=pca_components)))

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return X, y, preprocessor

# ----------------------------------------------------------------------
# 4) Correlation Matrix
# ----------------------------------------------------------------------
def plot_correlation_matrix(df, exclude_columns_indices=None):
    if exclude_columns_indices:
        df = df.drop(df.columns[exclude_columns_indices], axis=1)
    numeric_df = df.select_dtypes(include=['int64','float64'])
    corr = numeric_df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    plt.show()

# ----------------------------------------------------------------------
# 5) RF and GB Models (GridSearchCV)
# ----------------------------------------------------------------------
def train_model(X_train, y_train, preprocessor, model_type="random_forest", random_state=42):
    if model_type == "random_forest":
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=random_state))
        ])
        param_grid = {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
    elif model_type == "gradient_boosting":
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=random_state))
        ])
        param_grid = {
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5]
        }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

# ----------------------------------------------------------------------
# 6) Evaluation (MSE, R2)
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{model_name} (Test) - Mean Squared Error: {mse:.3f}")
    print(f"{model_name} (Test) - R² Score: {r2:.3f}")

# ----------------------------------------------------------------------
# 7) Feature Importances (RF/GBM)
# ----------------------------------------------------------------------
def compute_feature_importance(model, feature_names, model_name="model"):
    if hasattr(model.best_estimator_['regressor'], 'feature_importances_'):
        importances = model.best_estimator_['regressor'].feature_importances_
        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df.sort_values("Importance", ascending=False, inplace=True)

        print(f"\n{model_name} - Feature Importance:")
        print(fi_df.head(10))

        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=fi_df.head(10))
        plt.title(f"{model_name} - Top 10 Important Features")
        plt.tight_layout()
        plt.show()
        return fi_df
    else:
        print(f"{model_name} does not expose feature_importances_")
        return None

# ----------------------------------------------------------------------
# 8) Actual vs Predicted + Train/Test Split
# ----------------------------------------------------------------------
def plot_actual_vs_predicted_with_split(data, actual, predicted, train_end_year=2013, model_name="default"):
    data = data.copy()
    data['x_value'] = data['year'] + (data['quarter'] - 1)/4.0

    plt.figure().set_figwidth(20)
    plt.plot(data['x_value'], actual, label='Actual', marker='o', color='red')
    plt.plot(data['x_value'], predicted, label='Predicted', marker='8', alpha=0.5, color='blue')

    plt.axvline(x=train_end_year, color='gray', linestyle='--', label='Train/Test Split')
    plt.title(f"{model_name} - Actual vs. Predicted (Train/Test Split)")
    plt.xlabel("Time (year.quarter)")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------------------------------------------------
# KDE Subplots (3x2)
# ----------------------------------------------------------------------
def plot_kde_subplots_3x2(data, feature_columns, target_column, threshold, model_name="KDE Subplots"):
    data = data.copy()
    data['Growth Category'] = data[target_column].apply(lambda x: 1 if x > threshold else 0)

    num_feats = len(feature_columns)
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 14))
    axes = axes.flatten()  # 6 axes

    for i, feature in enumerate(feature_columns):
        if i >= 6:
            print("Warning: more than 6 features, subplots limited to 6.")
            break
        ax = axes[i]
        if not pd.api.types.is_numeric_dtype(data[feature]):
            ax.text(0.1, 0.5, f"'{feature}' is not numeric, skipping", color='red')
            ax.axis('off')
            continue

        sns.kdeplot(data=data[data['Growth Category'] == 1], x=feature,
                    label='Good', color='green', fill=True, alpha=0.5, ax=ax)
        sns.kdeplot(data=data[data['Growth Category'] == 0], x=feature,
                    label='Bad', color='red', fill=True, alpha=0.5, ax=ax)

        ax.set_title(f"{model_name}: {feature}")
        ax.grid(True)
        ax.legend()

    
    if num_feats < 6:
        for j in range(num_feats, 6):
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_kde_for_intersect_top6_features(rf_fi_df, gb_fi_df, data, target_column, threshold=None):
    top_rf = rf_fi_df.head(6)['Feature'].tolist() if rf_fi_df is not None else []
    top_gb = gb_fi_df.head(6)['Feature'].tolist() if gb_fi_df is not None else []

    common_feats = list(set(top_rf).intersection(set(top_gb)))
    print(f"\n=== Common Top-6 Features (RF ∩ GB) ===")
    print(common_feats)

    if threshold is None:
        threshold = data[target_column].mean()

    plot_kde_subplots_3x2(
        data=data,
        feature_columns=common_feats,
        target_column=target_column,
        threshold=threshold,
        model_name="Common Top6"
    )
# ----------------------------------------------------------------------
# 10) PCA
# ----------------------------------------------------------------------
def perform_pca_analysis(X, n_components=None):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_std)
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(explained_variance)+1), explained_variance,
             marker='o', linestyle='--')
    plt.title("Explained Variance by Principal Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.xticks(range(1, len(explained_variance)+1))
    plt.grid()
    plt.show()

    loadings = pd.DataFrame(pca.components_.T, columns = [f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    return pca_df, explained_variance

# ----------------------------------------------------------------------
# 11) Elastic Net
# ----------------------------------------------------------------------
def train_elastic_net(data, target_column):
    df = data.copy()
    drop_cols = [
        target_column,
        'year',
        'quarter',
        'observation_date'
    ]
    
    X = df.drop(columns=drop_cols, errors='ignore').dropna()
    Y = df[target_column].loc[X.index]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    elastic_net_model = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    )

    elastic_net_model.fit(X_train, Y_train)
    train_score = elastic_net_model.score(X_train, Y_train)
    test_score = elastic_net_model.score(X_test, Y_test)

    print(f"\n--- Elastic Net Results ---")
    print(f"Training R^2 Score: {train_score:.3f}")
    print(f"Testing R^2 Score:  {test_score:.3f}")

    coefficients = elastic_net_model.named_steps['elasticnet'].coef_
    print("Coefficients:", coefficients)

    return elastic_net_model

# ----------------------------------------------------------------------
# 12) OLS (Top 6 Features)
# ----------------------------------------------------------------------
def perform_ols_on_top_features(rf_feature_imp_df, gb_feature_imp_df, X, y, top_n=6):
    if rf_feature_imp_df is not None:
        top_rf = rf_feature_imp_df.head(top_n)['Feature'].tolist()
    else:
        top_rf = []
    if gb_feature_imp_df is not None:
        top_gb = gb_feature_imp_df.head(top_n)['Feature'].tolist()
    else:
        top_gb = []
    
    top_features_union = list(set(top_rf + top_gb))
    print(f"\n[OLS] Seçilecek feature'lar (union): {top_features_union}")

    available_feats = [f for f in top_features_union if f in X.columns]
    print(f"[OLS] Kodda mevcut feature'lar: {available_feats}")

    X_ols = sm.add_constant(X[available_feats].copy())
    y_ols = y.copy()

    df_ols = pd.concat([X_ols, y_ols], axis=1).dropna()
    y_ols = df_ols[y_ols.name]
    X_ols = df_ols.drop(columns=[y_ols.name])

    ols_model = sm.OLS(y_ols, X_ols).fit()
    print("\n[OLS] Regression Summary on top features:")
    print(ols_model.summary())

    return ols_model

# ----------------------------------------------------------------------
# 13) SHAP Analysis
# ----------------------------------------------------------------------
def shap_analysis(model_pipeline, X, feature_names, model_name="default"):
    if hasattr(model_pipeline, 'named_steps'):
        model = model_pipeline.named_steps['regressor']
    else:
        model = model_pipeline

    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", feature_names=feature_names)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    
    shap.initjs()
    for i in range(min(len(shap_values), 5)):
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X.iloc[i, :],
            feature_names=feature_names,
            matplotlib=True
        )
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.show()

# ----------------------------------------------------------------------
# 14) Classification (Logistic + ROC)
# ----------------------------------------------------------------------
def perform_classification_analysis(df,
                                    target_gdp_column='Real Gross Domestic Product Percent Change from Preceding Period Quarterly Seasonally Adjusted Annual Rate',
                                    threshold=2.8):
    df = df.copy()

    df['Good Growth'] = (df[target_gdp_column] > threshold).astype(int)

    X = df.select_dtypes(include=['float64','int64']).drop(columns=['Good Growth', target_gdp_column, "observation_date", "year", "quarter"], errors='ignore')
    X.fillna(X.mean(), inplace=True)

    y_binary = df['Good Growth'][X.index]

    corr_matrix = X.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix (Classification Features)')
    plt.show()

    threshold_corr = 0.8
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold_corr:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    print(f"Highly correlated pairs (|corr| > {threshold_corr}):")
    for pair in high_corr_pairs:
        print(pair)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

    log_reg_model = LogisticRegression(max_iter=2000, solver='lbfgs')
    log_reg_model.fit(X_train, y_train)

    y_pred = log_reg_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    class_rpt = classification_report(y_test, y_pred)

    print(f"\n[Classification] Logistic Regression => threshold={threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_mat)
    print("Classification Report:")
    print(class_rpt)

    
    if len(np.unique(y_test)) > 1:
        y_prob = log_reg_model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc_value = roc_auc_score(y_test, y_prob)

        print(f"ROC AUC Score: {auc_value:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC (AUC = {auc_value:.3f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    else:
        print("Warning: Only one class present in test set, skipping ROC/AUC calculation.")

    return log_reg_model

# ----------------------------------------------------------------------
# 15) AR(4) Model 
# ----------------------------------------------------------------------
def train_ar4_model(data, target_column, train_end_year=2013):
    df = data.copy()

    
    df.sort_values(by=['year', 'quarter'], inplace=True)

    
    for lag in range(1, 5):  
        df[f'lag{lag}'] = df[target_column].shift(lag)

    
    train_df = df[df['year'] <= train_end_year].copy()
    test_df = df[(df['year'] > train_end_year) & (df['year'] <= 2020)].copy()

    
    train_df.dropna(subset=[target_column] + [f'lag{i}' for i in range(1, 5)], inplace=True)
    test_df.dropna(subset=[target_column] + [f'lag{i}' for i in range(1, 5)], inplace=True)

    
    X_train_ar4 = sm.add_constant(train_df[[f'lag{i}' for i in range(1, 5)]])
    y_train_ar4 = train_df[target_column]

    
    ar4_model = sm.OLS(y_train_ar4, X_train_ar4).fit()

    
    X_test_ar4 = sm.add_constant(test_df[[f'lag{i}' for i in range(1, 5)]])
    y_test_ar4 = test_df[target_column]
    
    
    test_pred_ar4 = ar4_model.predict(X_test_ar4)

    
    mse_ar4_test = mean_squared_error(y_test_ar4, test_pred_ar4)
    r2_ar4_test = r2_score(y_test_ar4, test_pred_ar4)

    print("\n--- AR(4) Model Summary ---")
    print(ar4_model.summary())
    print(f"AR(4) Test MSE: {mse_ar4_test:.3f}")
    print(f"AR(4) Test R²: {r2_ar4_test:.3f}")

    
    df_ar4 = df.dropna(subset=[f'lag{i}' for i in range(1, 5)]).copy()
    X_full_ar4 = sm.add_constant(df_ar4[[f'lag{i}' for i in range(1, 5)]])
    full_pred_ar4 = ar4_model.predict(X_full_ar4)
    
    df_ar4['ar4_pred'] = full_pred_ar4

    return ar4_model, df_ar4

# ----------------------------------------------------------------------
# 16) All in One Graph (AR(4), RF, GB vs. Actual)
# ----------------------------------------------------------------------
def plot_forecasted_vs_actual_multiple(data, 
                                       target_column,
                                       df_ar4,
                                       rf_preds_all,
                                       gb_preds_all,
                                       train_end_year=2013):
    data = data.copy()
    data.sort_values(by=['year','quarter'], inplace=True)
    data['x_val'] = data['year'] + (data['quarter'] - 1)/4.0

    df_ar4 = df_ar4.copy()
    df_ar4.sort_values(by=['year','quarter'], inplace=True)
    df_ar4['x_val'] = df_ar4['year'] + (df_ar4['quarter'] - 1)/4.0

    merged = pd.merge(data, df_ar4[['year','quarter','ar4_pred']], 
                      on=['year','quarter'], how='left')
    merged['rf_pred'] = rf_preds_all
    merged['gb_pred'] = gb_preds_all
    merged.sort_values(by=['year','quarter'], inplace=True)

    plt.figure().set_figwidth(20)

    
    plt.scatter(merged['x_val'], merged[target_column], 
                color='red', alpha = 0.8, label='Actual')

    
    plt.scatter(merged['x_val'], merged['ar4_pred'], 
             marker ='o', color='orange', alpha = 0.4, label='AR(4)')

    
    plt.scatter(merged['x_val'], merged['rf_pred'], 
             marker ='o', color='purple', alpha = 0.4, label='Random Forest')

    
    plt.plot(merged['x_val'], merged['gb_pred'], 
             marker ='o', color='blue', alpha = 0.3, label='Gradient Boosting')

    
    plt.axvline(x=train_end_year, color='gray', linestyle='--', label='Train/Test Split')

    plt.title("Forecasted vs Actual GDP Growth (Quarterly)")
    plt.xlabel("Year")
    plt.ylabel("GDP Growth (%)")
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()


# ----------------------------------------------------------------------
# 17) Main Function
# ----------------------------------------------------------------------
def main(file_path,
         target_column,
         date_col='observation_date',
         columns_to_drop=None,
         pca_components=None,
         train_end_year=2013):

    
    data = load_data(file_path)
    data = parse_quarter_column(data, date_col_name=date_col)

    
    data = data[data['year'] < 2020] 
    print("Data shape after removing 2020+:", data.shape)

    
    plot_correlation_matrix(data, exclude_columns_indices=[-1,-2])

    
    train_data = data[data['year'] <= train_end_year]
    holdout_data = data[(data['year'] > train_end_year) & (data['year'] < 2020)]

    X_train, y_train, preprocessor = preprocess_data(train_data, target_column, columns_to_drop, pca_components)
    X_holdout, y_holdout, _ = preprocess_data(holdout_data, target_column, columns_to_drop, pca_components)

    print("Train set shape:", X_train.shape)
    print("Test set shape: ", X_holdout.shape)

    # 1) Random Forest
    rf_model = train_model(X_train, y_train, preprocessor, model_type="random_forest")
    print("Random Forest - Best Params:", rf_model.best_params_)
    rf_feature_imp_df = compute_feature_importance(rf_model, X_train.columns, "Random Forest")

    print("\nRandom Forest - Train Performance:")
    evaluate_model(rf_model, X_train, y_train, "Random Forest")
    print("\nRandom Forest - Test Performance:")
    evaluate_model(rf_model, X_holdout, y_holdout, "Random Forest")

    X_all, y_all, _ = preprocess_data(data, target_column, columns_to_drop, pca_components)
    rf_all_preds = rf_model.predict(X_all)

    plot_actual_vs_predicted_with_split(
        data=data,
        actual=y_all,
        predicted=rf_all_preds,
        train_end_year=train_end_year,
        model_name="Random Forest"
    )

    # 2) Gradient Boosting
    gb_model = train_model(X_train, y_train, preprocessor, model_type="gradient_boosting")
    print("\nGradient Boosting - Best Params:", gb_model.best_params_)
    gb_feature_imp_df = compute_feature_importance(gb_model, X_train.columns, "Gradient Boosting")

    print("\nGradient Boosting - Train Performance:")
    evaluate_model(gb_model, X_train, y_train, "Gradient Boosting")
    print("\nGradient Boosting - Test Performance:")
    evaluate_model(gb_model, X_holdout, y_holdout, "Gradient Boosting")

    gb_all_preds = gb_model.predict(X_all)
    plot_actual_vs_predicted_with_split(
        data=data,
        actual=y_all,
        predicted=gb_all_preds,
        train_end_year=train_end_year,
        model_name="Gradient Boosting"
    )
    
    # -------------------------------------------------------------
    # RF and GB Top 6 Features Combined KDE Graph
    # -------------------------------------------------------------
    
    plot_kde_for_intersect_top6_features(
        rf_fi_df=rf_feature_imp_df,
        gb_fi_df=gb_feature_imp_df,
        data=data,
        target_column=target_column,
        threshold=y_all.mean()
    )

    # 3) PCA 
    pca_df, explained_variance = perform_pca_analysis(X_all, n_components=pca_components)
    print("PCA Explained Variance:", explained_variance)

    # 4) Elastic Net 
    enet_model = train_elastic_net(data, target_column)

    # 5) OLS -> top 6 features
    perform_ols_on_top_features(rf_feature_imp_df, gb_feature_imp_df, X_all, y_all, top_n=6)

    # 6) SHAP
    print("\n=== SHAP Analysis: Random Forest ===")
    shap_analysis(rf_model.best_estimator_, X_all, X_all.columns, model_name="Random Forest")

    print("\n=== SHAP Analysis: Gradient Boosting ===")
    shap_analysis(gb_model.best_estimator_, X_all, X_all.columns, model_name="Gradient Boosting")

    # 7) Classification (Logistic + ROC)
    print("\n=== Classification Analysis (Logistic Regression + ROC) ===")
    classification_model = perform_classification_analysis(data, 
                                                           target_gdp_column=target_column,
                                                           threshold=0.69)

    # 8) AR(4) Model
    ar4_model, df_ar4 = train_ar4_model(data, target_column, train_end_year=train_end_year)
    

    # 9) All in One Graph (AR(4), RF, GB, Actual)
    plot_forecasted_vs_actual_multiple(
        data=data,
        target_column=target_column,
        df_ar4=df_ar4, 
        rf_preds_all=rf_all_preds,
        gb_preds_all=gb_all_preds,
        train_end_year=train_end_year
    )

    return pca_df, explained_variance, enet_model, classification_model, ar4_model, df_ar4

# ----------------------------------------------------------------------
# 18) Script 
# ----------------------------------------------------------------------
if __name__ == "__main__":
    file_path = "https://github.com/s-gkce/EC48E_finalProjectData/raw/refs/heads/main/US_Growth_Data.xlsx"
    target_column = "Real Gross Domestic Product, Percent Change, Quarterly, Seasonally Adjusted Annual Rate"

    columns_to_drop = None  
    pca_components = None

    results = main(
        file_path,
        target_column,
        date_col='observation_date',
        columns_to_drop=columns_to_drop,
        pca_components=pca_components,
        train_end_year=2013
    )

