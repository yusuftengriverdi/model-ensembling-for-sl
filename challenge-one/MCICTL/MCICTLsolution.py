import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, recall_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real

# Get Data on MCICTL.
MCICTLtrain = pd.read_csv("data/MCICTLtrain.csv")
MCICTLtest = pd.read_csv("data/MCICTLtest.csv")

# Process data. Whatever you do, apply to test data set too.
MCICTLtrain = MCICTLtrain.drop(columns=["ID"])
MCICTLtest = MCICTLtest.drop(columns=["ID"])

# If necessary, we can use X-y formula.
y = MCICTLtrain["Label"]
X = MCICTLtrain.drop(columns=["Label"])

# Preprocess pipeline.
preproc = StandardScaler()
X_train_val = preproc.fit_transform(X)
X_test = preproc.transform(MCICTLtest)

# Feature reduction with PCA.
pca = PCA()
X_train_val_pca = pca.fit_transform(X_train_val)
X_test_pca = pca.transform(X_test)


train_val_set = pd.DataFrame(X_train_val_pca)
train_val_set.loc[:, "Label"] = y
# print(train_val_set.shape)
train_val_set['Label'] = train_val_set['Label'].astype('category')
train_val_set['Label'] = train_val_set['Label'].cat.codes
train_val_set['Label'] = train_val_set['Label'].astype(int)

# Set the random seed for reproducibility
np.random.seed(123)
# Calculate the sample size.
# Perform random sampling with class distribution preservation.
train_set, val_set = train_test_split(train_val_set, test_size=0.30, stratify=y)

print(train_set.shape, val_set.shape)

def custom_matthews_corrcoef(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred.astype(int))

def custom_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred.astype(int))

def custom_f1(y_true, y_pred):
    return f1_score(y_true, y_pred.astype(int))

def custom_recall(y_true, y_pred):
    return recall_score(y_true, y_pred.astype(int))

def custom_ap(y_true, y_pred):
    return average_precision_score(y_true, y_pred.astype(int))

cv_metrics = {
    'accuracy_score': make_scorer(custom_accuracy),
    'roc_auc_score': make_scorer(roc_auc_score),
    'matthews_corr': make_scorer(custom_matthews_corrcoef),
    'mean_squared_error': make_scorer(mean_squared_error),
    'r2_score': make_scorer(r2_score),
    'f1_score': make_scorer(custom_f1),
    'recall': make_scorer(custom_recall),
    'average_precision': make_scorer(average_precision_score),
}

np.random.seed(107)

# Create cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=15, n_repeats=5, random_state=107)

# Define custom scoring function

def get_results(model_list, X, y):
    
    fitted_models = []
    for i, model in tqdm(enumerate(model_list)):
        model_name = type(model).__name__
        scores = cross_validate(model, X, y, cv=cv, scoring=cv_metrics, 
                                return_train_score=True,
                                return_estimator=True)
        fitted_models.append(scores['estimator'])
        scores = pd.DataFrame.from_dict(scores)
        scores = scores.mean(numeric_only=True)
        scores["Model"] = model_name
        if i == 0:
            results = scores.to_frame().transpose()
        else:
            results = pd.concat([results, scores.to_frame().transpose()])

    return results, fitted_models


# Assuming X_train and y_train are the training features and labels respectively
# and X_test and y_test are the testing features and labels respectively

models = [ ('MLPClassifier', MLPClassifier(learning_rate='adaptive', learning_rate_init=0.01, activation='logistic')), 
           ('LogisticRegression', LogisticRegression()),
           ('XGBClassifier', XGBClassifier(booster='gblinear')),
           ('SVCrbf', SVC(kernel='rbf', probability=True)),
           ('KNeighborsClassifier', KNeighborsClassifier()),
           ('SVCpoly', SVC(kernel='poly', probability=True)),
           ('SVClinear', SVC(kernel='linear', probability=True))
]
search_spaces = [
    {
        'hidden_layer_sizes': Integer(50, 200),
        'alpha': Real(1e-6, 1e-2, prior='log-uniform'),
        'learning_rate_init': Real(0.001, 0.1),
        'activation': Categorical(['logistic', 'relu', 'tanh'])
    },
    {
        'penalty': Categorical(['l1', 'l2']),
        'C': Real(0.01, 10, prior='log-uniform'),
        'solver': Categorical(['liblinear', 'saga'])
    },
    {
        'n_components': Integer(1, 10),
        'scale': Categorical([True, False])
    },
    {
        'C': Real(0.01, 10, prior='log-uniform'),
        'gamma': Real(1e-6, 1e-2, prior='log-uniform')
    },
    {
        'n_neighbors': Integer(1, 20),
        'weights': Categorical(['uniform', 'distance']),
        'p': Integer(1, 5)
    },
    {
        'C': Real(0.01, 10, prior='log-uniform'),
        'gamma': Real(1e-6, 1e-2, prior='log-uniform'),
        'degree': Integer(2, 5),
        'coef0': Real(0, 1)
    },
    {
        'C': Real(0.01, 10, prior='log-uniform')
    }
]

best_models = []
# Perform Bayesian search
for i, (model_name, model) in tqdm(enumerate(models)):
    bayes_search = BayesSearchCV(
        model, search_spaces[i], cv=5, scoring=make_scorer(custom_matthews_corrcoef), n_jobs=-1
    )
    bayes_search.fit(train_set.iloc[:, :-1], train_set["Label"])
    
    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    
    best_models.append((model_name, best_model, best_params))

best_models = np.asarray(best_models)[:, 1]

def get_results(model_list, X, y):
    fitted_models = []
    for i, model in tqdm(enumerate(model_list)):
        model_name = type(model).__name__
        scores = cross_validate(model, X, y, cv=cv, scoring=cv_metrics, 
                                return_train_score=True,
                                return_estimator=True)
        fitted_models.append(scores['estimator'])
        scores = pd.DataFrame.from_dict(scores)
        scores = scores.mean(numeric_only=True)
        scores["Model"] = model_name
        if i == 0:
            results = scores.to_frame().transpose()
        else:
            results = pd.concat([results, scores.to_frame().transpose()])

    return results, fitted_models

train_results, fitted_models = get_results(best_models, train_set.iloc[:, :-1], train_set["Label"])
train_results.to_csv("train_results.csv")

# Exclude Bayesian Ridge, as it has really poor performance.
flat_models = np.array(fitted_models).flatten()
flat_models = flat_models[75:]

val_preds = np.zeros((len(val_set), len(flat_models)))
for i, model in tqdm(enumerate(flat_models)):
    preds = model.predict(val_set.iloc[:, :-1])
    if len(preds.shape) == 2:
        preds = np.squeeze(preds)
    val_preds[:, i] = preds

# Separate the features and target variable
X_stacked = val_preds.copy()
y_stacked = val_set.iloc[:, -1]

# Split the stacked data into training and testing sets
X_train_stacked, X_test_stacked, y_train_stacked, y_test_stacked = train_test_split(
    X_stacked, y_stacked, test_size=0.5, random_state=42
)

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'max_depth': [None, 5, 10],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for the best split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
    'random_state': [42]  # Random seed for reproducibility
}

# Initialize the meta-model (Gradient Boosting classifier was the best empirically)
# Initialize the base model
base_model = RandomForestClassifier()

# Perform random search
meta_model = RandomizedSearchCV(base_model, param_grid, n_iter=10, scoring=cv_metrics, refit='matthews_corr', cv=5, random_state=42)

# Train the meta-model
meta_model.fit(X_train_stacked, y_train_stacked)

optimized_gb_model = meta_model.best_estimator_

# Evaluate the meta-model on the test set
meta_val_test_preds = optimized_gb_model.predict(X_test_stacked)

def custom_metrics(y_true, y_pred, name):
    metrics = {}
    for metric_name, scorer in cv_metrics.items():
        metrics[metric_name] = scorer._score_func(y_true, y_pred)
    metrics = dict(sorted(metrics.items()))  # Sort metrics by keys
    df = pd.DataFrame(metrics, index=[0])
    df['Model'] = name
    return df

meta_results = custom_metrics(y_test_stacked, meta_val_test_preds, "meta_gbc")
meta_results.to_csv("meta_gradient_boosting_results.csv")

MCICTLtest = pd.read_csv("data/MCICTLtest.csv")

def get_preds_probs(ensembled_model, flat_models, newdata=X_test):
    test_set = pd.DataFrame(newdata)
    test_preds = np.zeros((len(test_set), len(flat_models)))
    test_probs = np.zeros((len(test_set), len(flat_models)))
    for i, model in tqdm(enumerate(flat_models)):
        preds = model.predict(test_set)        
        if len(preds.shape) == 2:
            preds = np.squeeze(preds)
        test_preds[:, i] = preds


    final_preds = ensembled_model.predict(test_preds)
    final_probs = ensembled_model.predict_proba(test_preds)
    pd.DataFrame(final_preds, index=MCICTLtest["ID"], columns=["Label"]).to_csv("final_preds_MCICTL.csv")
    pd.DataFrame(final_probs, index=MCICTLtest["ID"], columns=["AD", "CTL"]).to_csv("final_probs_MCICTL.csv")
    return final_probs

get_preds_probs(optimized_gb_model, flat_models, newdata=X_test_pca)
