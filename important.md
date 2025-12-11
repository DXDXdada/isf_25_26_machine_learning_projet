# Preprocessing des variables catégorielles en one hot encoding
#One hot encoding des variables catégorielles
preproc_ohe = preproc.OneHotEncoder(handle_unknown='ignore')
preproc_ohe = preproc.OneHotEncoder(drop='first', sparse_output = False).fit(vars_categorielles) 

variables_categorielles_ohe = preproc_ohe.transform(vars_categorielles)
variables_categorielles_ohe = pd.DataFrame(variables_categorielles_ohe, 
                                           columns = preproc_ohe.get_feature_names_out(vars_categorielles.columns))
variables_categorielles_ohe.head()

# Normalisation des variables numériques 
preproc_scale = preproc.StandardScaler(with_mean=True, with_std=True)
preproc_scale.fit(vars_numeriques)

vars_numeriques_scaled = preproc_scale.transform(vars_numeriques)
vars_numeriques_scaled = pd.DataFrame(vars_numeriques_scaled, 
                            columns = vars_numeriques.columns)
vars_numeriques_scaled.head()

# Sampling 
X_global = vars_numeriques_scaled.merge(variables_categorielles_ohe,
                            left_index = True,
                            right_index = True)
#Réorganisation des données 
X = X_global.to_numpy()
Y = data_model["CM"]

#Oversampling
#Appliquer le suréchantillonnage à la classe minoritaire
sampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)


#Sampling en 80% train et 20% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Observation de la distribution sur Y_train
df = pd.DataFrame(y_train_resampled, columns = ["SINISTRE"])
fig = px.histogram(df, 
                   x="SINISTRE",
                  title="Distribution de la variable Y_train_resampled")
fig.show()

# Kfold 
#Initialisation
#Nombre de sous-échantillons pour la cross-validation
num_splits = 5

#Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

#Initialisation du KFold cross-validation splitter
kf = KFold(n_splits=num_splits)

#Listes pour enregistrer les performances du modèle
MAE_scores = []
MSE_scores = []
RMSE_scores = []
___
#Entrainement avec cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    
    # Fitting
    rf_regressor.fit(X_train, y_train)
    
    # Evaluation du modèle
    y_pred_test = rf_regressor.predict(X_test) 
    
    MAE = metrics.mean_absolute_error(y_test, y_pred_test)
    MSE = metrics.mean_squared_error(y_test, y_pred_test)
    RMSE = metrics.root_mean_squared_error(y_test, y_pred_test)
    
    #Concaténation des résultats
    MAE_scores.append(MAE)
    MSE_scores.append(MSE)
    RMSE_scores.append(RMSE)
___
#Calcul des métriques sur tous les folds

#MAE
for fold, mae in enumerate(MAE_scores, start=1):
    print(f"Fold {fold} MAE:", mae)

#MSE
for fold, mse in enumerate(MSE_scores, start=1):
    print(f"Fold {fold} MSE:", mse)

#RMSE
for fold, rmse in enumerate(RMSE_scores, start=1):
    print(f"Fold {fold} RMSE:", rmse)

# Grid search pour hyperparamètres (grid search x Kfold)
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error

#Sampling en 80% train et 20% test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
___
#Supposons que vous ayez des données d'entraînement X_train et y_train

#Définir la grille d'hyperparamètres à rechercher
param_grid = {
    'n_estimators': [60,65,70 ,75],
    'max_depth': [None,1,2,3],
    'min_samples_split': [5,8,10,11,13, 14,15]
}
#Nombre de folds pour la validation croisée
num_folds = 5

#Initialisation du modèle RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

#Création de l'objet GridSearchCV pour la recherche sur grille avec validation croisée
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=KFold(n_splits=num_folds, shuffle=True, random_state=42),  # Validation croisée avec 5 folds
    scoring='neg_mean_squared_error',  # Métrique d'évaluation (moins c'est mieux)
    n_jobs=-1  # Utiliser tous les cœurs du processeur
)

#Exécution de la recherche sur grille
grid_search.fit(X_train, y_train)

#Afficher les meilleurs hyperparamètres
best_params = grid_search.best_params_
print("Meilleurs hyperparamètres : ", best_params)

#Initialiser le modèle final avec les meilleurs hyperparamètres
best_rf = RandomForestRegressor(random_state=42, **best_params)

# Cross validation
#RMSE de chaque fold
rmse_scores = cross_val_score(best_rf, X_train, y_train, cv=num_folds, scoring='neg_root_mean_squared_error')

# Afficher les scores pour chaque fold
for i, score in enumerate(rmse_scores):
    print(f"RMSE pour le fold {i + 1}: {score}")
    
#MSE de chaque fold
mse_scores = cross_val_score(best_rf, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error')

# Afficher les scores pour chaque fold
print("\n")
for i, score in enumerate(mse_scores):
    print(f"MSE pour le fold {i + 1}: {score}")
    
#MAE de chaque fold
mae_scores = cross_val_score(best_rf, X_train, y_train, cv=num_folds, scoring='neg_mean_absolute_error')

#Afficher les scores pour chaque fold
print("\n")
for i, score in enumerate(mae_scores):
    print(f"MAE pour le fold {i + 1}: {score}")

#Entraîner le modèle final sur toute la base
best_rf.fit(X_train, y_train)

#Faire des prédictions sur l'ensemble de test
y_pred = best_rf.predict(X_test)

#Calculer la métrique de performance (dans ce cas, RMSE)
rmse = metrics.root_mean_squared_error(y_test, y_pred)
print(f"RMSE : {rmse}")

#Calculer la métrique de performance (dans ce cas, MSE)
mse = metrics.mean_squared_error(y_test, y_pred)
print(f"MSE : {mse}")

#Calculer la métrique de performance (dans ce cas, MAE)
mae = metrics.mean_absolute_error(y_test, y_pred)
print(f"MAE : {mae}")

# Calcul des métriques pour les prédictions en classification 
# Matrice de confusion
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
__
#Calculer le recall des prédictions
recall = metrics.recall_score(y_test, y_pred)
print(f"Recall : {recall}")
__
#Calculer l'accuracy des prédictions
acc = metrics.accuracy_score(y_test, y_pred)
print(f"Exactitude : {acc}")
__
#Calculer le precision des prédictions
precision = metrics.precision_score(y_test, y_pred)
print(f"Precision : {precision}")

# Courbe ROC
# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_baseline)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'CatBoost (AUC = {roc_auc_score(y_test, y_pred_proba_baseline):.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Hasard')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.legend()
plt.grid(True)
plt.show()

# Importance des variables

print("\n=== Importance des variables ===\n")

feature_importance = model_baseline.get_feature_importance(train_pool)
feature_names = X_train.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df)

#Visualisation
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
plt.xlabel('Importance')
plt.title('Top 15 variables les plus importantes')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()