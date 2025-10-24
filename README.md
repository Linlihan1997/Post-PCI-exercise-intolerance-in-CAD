# Post-PCI-exercise-intolerance-in-CAD
This repository provides core scripts for developing and validating 7 ML models to predict post-PCI exercise intolerance based on routinely collected clinical indicators. It includes data preprocessing, model training, and evaluation.
# 1. Project Overview
  This repository contains the core scripts for developing and validating machine learning (ML) models to predict exercise intolerance after percutaneous coronary intervention (PCI), based on routinely collected clinical and laboratory parameters.  
  We constructed and compared seven supervised ML algorithms (Logistic Regression, Random Forest, Support Vector Machine, K-Nearest Neighbors, Extreme Gradient Boosting, Multilayer Perceptron, and LightGBM). The workflow includes data preprocessing, feature engineering, model training, internal validation, and clinical interpretability analysis

# 2.📁Project structure

- `README.md` – Project overview and instructions
- `Feature selection/`
  - `feature selection.R`
- `Model Construction and Evaluation/`
  - `Model construction.py`
  - `ROC_and_Calibration.py`
  - `Decision_Curve_Analysis.py` 
- `Interpretation/`
  - `SHAP_Interpretation.py` 
- `Deployment/`
  - `mlp.rar`

# 3. Methods
## 3.1 Data Preprocessing
Clinical and demographic variables were standardized before modeling.  
Outcome variable (exercise_tolerance) was coded as binary:  
· _0 = Exercise tolerance_  
· _1 = Exercise intolerance_    
Missing values were handled using appropriate imputation methods.  
Continuous variables were standardized (z-score) for algorithms requiring scaling.  
## 3.2 Feature Selection
We performed in two stages. First, the least absolute shrinkage and selection operator (LASSO) regression with 10-fold cross-validation was used to select predictors by shrinking irrelevant coefficients to zero.   
To assess robustness, the Boruta algorithm was applied to evaluate the relative importance of all candidate variables. Multicollinearity was assessed using the variance inflation factor (VIF), with VIF <5 considered acceptable.  
## 3.3 Model Construction
| Model                                          | Description                                                   | Key Parameters                                                                                                                                                                     |
| :--------------------------------------------- | :------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression (LR)**                   | Linear baseline classifier with logit link                    | `alpha`, `lambda`                                                                                                                                                                  |
| **Random Forest (RF)**                         | Ensemble of decision trees reducing variance                  | `ntree`, `mtry`, `nodesize`, `class_weight`, `n_jobs`                                                                                                                              |
| **Support Vector Machine (SVM)**               | Kernel-based classifier maximizing the margin between classes | `C`, `kernel`, `gamma`                                                                                                                                                             |
| **K-Nearest Neighbors (KNN)**                  | Predicts class by majority vote of nearest neighbors          | `k`, `distance`, `kernel`                                                                                                                                                          |
| **Extreme Gradient Boosting (XGB)**            | Gradient-boosted trees with regularization                    | `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_lambda`, `booster`                                                |
| **Multilayer Perceptron (MLP)**                | Feedforward neural network for nonlinear mapping              | `hidden_layer_sizes`, `activation`, `learning_rate`, `max_iter`, `alpha`, `epoch`                                                                                                  |
| **Light Gradient Boosting Machine (LightGBM)** | Histogram-based gradient boosting framework                   | `num_leaves`, `max_depth`, `min_child_samples`, `learning_rate`, `n_estimators`, `reg_alpha`, `reg_lambda`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `class_weight` |

# 4. Model Construction
**Discrimination**: ROC curves and AUROC with 95% CI using bootstrapping (n=1000).  
**Calibration**: Reliability curves (LOWESS smoothed), Brier score, and Hosmer–Lemeshow test.  
**Performance Metrics**: Accuracy, recall, specificity, PPV, NPV, and F1 score at representative thresholds (default, F1-optimal).  
**Clinical Utility**: Decision curve analysis (DCA) for net benefit comparison.  
**Interpretability**: Feature importance via SHAP (global and local explanations).  

# 5. Deployment
An **interactive web calculator** was developed using **Shiny (R)** and **reticulate** (Python interface) to facilitate clinical application of the MLP model.  
Healthcare professionals can input eight clinical variables (**Age, BMI, Diabetes, Gender, Hb, RBC, RHR, Smoking**) to obtain individualized risk probabilities and visualize variable contributions.  
Access the online calculator:  
👉 https://postPCIEI.shinyapps.io/shinyapp/

# 6. Deployment
All scripts are implemented in Python 3.9 and R 4.3.  
Required dependencies include numpy, pandas, scikit-learn, xgboost, lightgbm, and matplotlib.  
For the Shiny deployment, ensure that the reticulate package correctly links to the local Python environment  
