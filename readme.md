# Prédiction du Risque de Crédit

## 📋 Description

Projet de Machine Learning développé pour une banque visant à automatiser l'évaluation du risque associé aux demandes de financement.

## 🏗️ Structure du Projet

```
├── data/
│   ├── data_raw/              # Données brutes
│   └── data_processed/        # Données prétraitées
├── figs/                      # Figures de code source
├── models/                    # Modèles sauvegardés
├── notebooks/
│   ├── 02_feature_engineering.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_interpretability.ipynb
│   └── 06_deployment_strategy.ipynb
└── rapport.pdf             # Rapport détaillé
```

## 🔬 Méthodologie

1. **Feature Engineering** : Création de 40+ variables (ratios financiers, indicateurs de crédit, interactions)
2. **Preprocessing** : 3 pipelines adaptés (modèles linéaires, arbres, catégoriel)
3. **Modélisation** : 8 algorithmes testés avec hyperparameter tuning et CV 5-fold
4. **Évaluation** : RMSE, MAE, R² sur train/test split (80/20)

## 📈 Résultats

| Modèle | R² Test | RMSE Test | MAE Test |
|--------|---------|-----------|----------|
| Ridge/Lasso/ElasticNet | 0.79 | 3.65 | 2.81 |
| Random Forest | 0.83 | 3.28 | 2.51 |
| **XGBoost** | **0.85** | 3.15 | 2.43 |
| **LightGBM** | **0.85** | **3.12** | **2.41** |
| **CatBoost** | **0.85** | 3.14 | 2.42 |
| CNN | 0.83 | 3.35 | 2.58 |