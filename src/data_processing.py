import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, target_column, features_to_exclude=None):
    """
    Performs data cleaning, feature engineering, and prepares data for modeling.
    Returns preprocessed DataFrame and preprocessor pipeline.
    """
    df = df.copy()

    # Handle missing values (basic strategy: fill numerical NaNs with mean, categorical with mode)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean()) # Or median, or imputation model

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0]) # Or add 'Missing' category

    # Feature Engineering Examples (based on prompt's feature sets)
    # Combine location features
    df['Location_Combined'] = df['Province'].astype(str) + '_' + df['MainCrestaZone'].astype(str)

    # Age of Vehicle
    current_year = pd.Timestamp.now().year # Or use a fixed reference year from data
    df['VehicleAge'] = current_year - df['RegistrationYear']
    df['VehicleAge'].fillna(df['VehicleAge'].mean(), inplace=True) # Fill NaNs if any

    # Ratio features
    df['SumInsured_per_CubicCapacity'] = df['SumInsured'] / (df['Cubiccapacity'] + 1e-6) # Avoid division by zero
    df['SumInsured_per_CubicCapacity'].fillna(0, inplace=True) # Fill any resulting NaNs

    # Temporal Features (from TransactionDate)
    df['TransactionMonthNum'] = df['TransactionDate'].dt.month
    df['TransactionDayOfWeek'] = df['TransactionDate'].dt.dayofweek
    df['TransactionQuarter'] = df['TransactionDate'].dt.quarter

    # Select features for modeling
    # Exclude IDs, raw dates, and potentially target-related columns if not used as features directly
    # Ensure 'TransactionDate' is handled correctly before passing to ColumnTransformer
    if features_to_exclude is None:
        features_to_exclude = ['UnderwrittenCoverID', 'PolicyID', 'TransactionDate', 'PostalCode',
                               'Bank', 'AccountType', 'Product', 'Section', 'CoverCategory',
                               'CoverType', 'CoverGroup', 'StatutoryClass', 'StatutoryRiskType', # Potentially too high cardinality or leakage
                               'TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm', # Exclude if they are targets
                               'LossRatio', 'ClaimOccurred', 'ClaimFrequency', 'ClaimSeverity', 'Margin' # Exclude derived metrics
                              ]

    features = [col for col in df.columns if col not in features_to_exclude and col != target_column]

    # Identify categorical and numerical features
    categorical_features = df[features].select_dtypes(include=['object']).columns
    numerical_features = df[features].select_dtypes(include=np.number).columns

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Already handled above, but good for robust pipeline
        ('scaler', StandardScaler())
    ])

    # Handle high cardinality for some categorical features if needed (e.g., Make, Model)
    # For simplicity, using OneHotEncoder for all categoricals, but might be too many columns
    # Consider target encoding or reducing cardinality for high-cardinality features.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Already handled above
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns that are not transformed
    )

    return df[features], df[target_column], preprocessor, features

def split_data(X, y, test_size=0.3, random_state=42):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test