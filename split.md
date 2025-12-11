# Séparation du dataframe en 2 parties : variables numériques et variables catégorielles
# Imaginons qu'on ait deja:
# drop les variables deja correlees et les variables non utiles
# transformé les variables categorielles en one hot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0 — concatenate numeric and categorical features
X_global = pd.concat([vars_numeriques, variables_categorielles_ohe], axis=1).to_numpy()
Y = data["RiskScore"].to_numpy()

# 1 — split first
X_train, X_test, y_train, y_test = train_test_split(
    X_global, Y,
    test_size=0.2,
    random_state=42
)

# 2 — define preprocessing
numeric_features = vars_numeriques.columns.tolist()
categorical_features = variables_categorielles_ohe.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)
    ]
)

# 3 — fit only on training data
X_train_scaled = preprocessor.fit_transform(X_train)

# 4 — transform test data with the same fitted transformer
X_test_scaled = preprocessor.transform(X_test)
