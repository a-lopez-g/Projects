import argparse
import logging
import time
from datetime import datetime
import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_classif, SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

np.random.seed(42)
feat_select_dict = {
    "AN": "ANOVA",
    "MI": "Mutual Information" ,
    "L1": "log-reg L1 regularization"
}
model_dict = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XG": "XGBoost"
}

def select_features(X_tr, y_tr, X_tst, mode="AN", ratio=0.2):
    k = round(X_tr.shape[1]*ratio)
    if mode == "AN":
        feat_selector = SelectKBest(score_func=f_classif, k=k)
        X_tr_ = feat_selector.fit_transform(X_tr, y_tr)
        X_tst_ = feat_selector.transform(X_tst)
    elif mode == "MI":
        feat_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_tr_ = feat_selector.fit_transform(X_tr, y_tr)
        X_tst_ = feat_selector.transform(X_tst)    
    elif mode == "L1":
        #TODO
        X_tr_ = X_tr
        X_tst_ = X_tst
    
    return X_tr_, X_tst_

def select_classifier(classifier):
    if classifier == "LR":
        model = LogisticRegression(penalty="l1", solver='liblinear', class_weight="balanced")
        params = {'C':np.logspace(-3, 0, 50)}
    elif classifier == "RF":
        model = RandomForestClassifier()
        params = { 
            'n_estimators': np.arange(30, 401, 10),
            'max_depth' : np.arange(4, 41, 4),
            }
    elif classifier == "XG":
        model = XGBClassifier(
            objective= 'multi:softprob',
            tree_method='gpu_hist',
            gpu_id=0,
            subsample=0.8,
            colsample_bytree=0.8)
        params = {
            "n_estimators": np.arange(20, 151, 10),
            "learning_rate": np.logspace(-2, 0, 5),
        }

    return model, params

def scale_data(X_tr, X_tst):
    scaler = StandardScaler()
    X_tr_ = scaler.fit_transform(X_tr)
    X_tst_ = scaler.transform(X_tst)

    return X_tr_, X_tst_

def main(feat_select: str, classifier: str, iters: str):
    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")
    logging.basicConfig(filename=f"monograms/{feat_select}_{classifier}_{dt_string}.log", encoding="utf-8", level=logging.DEBUG)
    logging.info(f"Model: {model_dict[classifier]}")
    logging.info(f"Feature selection: {feat_select_dict[feat_select]}")
    logging.info(f"Experiment iterations = {iters}")
    logging.info(f"Date & time of execution = {now.strftime('%Y/%m/%d %H:%M:%S')}")
    n_iters = int(iters)
    if classifier == "XG": 
        n_jobs = 1
    else:        
        n_jobs = -1
        
    # Load data
    df = pd.read_csv("../../data/sentiment_analysis/tf_idf/labelled_convers_tf_idf_monograms.csv", index_col=0)
    conv_ids = df.conversation_id.values
    df.drop(["conversation_id"], axis=1, inplace=True)
    X  = df.values[:, :-1]
    y = df.primary_label.values
    n, d = X.shape
    label_encoder = LabelEncoder()
    y_ = label_encoder.fit_transform(y)
    classes = [word.title() for word in label_encoder.classes_]
    logging.info(f"Input samples = {n}")
    logging.info(f"Input features = {d}")
    logging.info(f"Labels: {classes}")

    # Run n_iter experiments
    result_dict = {
        "y_tst_list": [],
        "ids_tst_list": [],
        "y_pred_list": [],
        "y_proba_list": [],
        "val_score_list": [],
        "best_param_list": [],
        "tst_score_list": [],
        "conf_matrix_list": [],
    }
    for iter in range(n_iters):
        # Partition data
        X_tr, X_tst, y_tr, y_tst, _, ids_tst = train_test_split(X, y_, conv_ids, test_size=0.1)

        # Remove constant features
        feat_vars = np.var(X_tr, axis=0)
        X_tr_ = X_tr[:, feat_vars != 0]
        X_tst_ = X_tst[:, feat_vars != 0]

        # Feature selection
        X_tr_, X_tst_ = select_features(X_tr_, y_tr, X_tst_, mode=feat_select)

        # Model validation
        if classifier == "LR":
            # Standardize data for logistic regression
            X_tr_, X_tst_ = scale_data(X_tr_, X_tst_)

        model, params = select_classifier(classifier)
        clf = GridSearchCV(model, params, scoring='roc_auc_ovr', n_jobs=n_jobs, verbose=1, refit=True)
        start = time.perf_counter()
        clf.fit(X_tr_, y_tr)
        stop = time.perf_counter()
        best_params = clf.best_params_
        best_val_score = clf.best_score_
        best_estimator = clf.best_estimator_

        # Model evaluation
        y_pred = best_estimator.predict(X_tst_)
        y_proba = best_estimator.predict_proba(X_tst_)
        mean_auc = roc_auc_score(y_tst, y_proba, average="weighted", multi_class='ovr')
        conf_mat = confusion_matrix(y_tst, y_pred)

        logging.info(f"EXPERIMENT ITERATION {iter + 1}")
        logging.info(f"Nº features post-selection = {X_tr_.shape[1]}")
        logging.info(f"Validation time = {(stop - start):.2f} seconds.")
        logging.info(f"Best validation score (OvR AUC) = {best_val_score:.3f}")
        logging.info(f"Best model parameters after validation = {best_params}")
        logging.info(f"Test score (OvR AUC) = {mean_auc:.3f}")

        result_dict["y_tst_list"].append(y_tst)
        result_dict["ids_tst_list"].append(ids_tst)
        result_dict["y_pred_list"].append(y_pred)
        result_dict["y_proba_list"].append(y_proba)
        result_dict["val_score_list"].append(best_val_score)
        result_dict["best_param_list"].append(best_params)
        result_dict["tst_score_list"].append(mean_auc)
        result_dict["conf_matrix_list"].append(conf_mat)

    # Save results
    with open(f"monograms/{feat_select}_{classifier}_{dt_string}.pickle", "wb") as f:
        pickle.dump(result_dict, f)

    logging.info(f"Average validation score (OvR AUC) = {np.mean(result_dict['val_score_list'])}")
    logging.info(f"Average test score (OvR AUC) = {np.mean(result_dict['tst_score_list'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_select", help="Variable selection strategy.", type=str)
    parser.add_argument("classifier", help="Classifier.", type=str)
    parser.add_argument("iters", help="Nnumber of independent iterations for the experiment.", type=str)
    args = parser.parse_args()

    main(args.feat_select, args.classifier, args.iters)