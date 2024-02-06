import copy
import numpy as np
from joblib import dump, load
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer, accuracy_score, confusion_matrix, plot_confusion_matrix, roc_auc_score, brier_score_loss


def build_ensemble(models, path):
    ensemble = []
    for m in models:
        ensemble.append((m, load(path+f"{m}.joblib")))
    
    return ensemble


def predict_ensemble(ensemble, X, y, threshold=0.5):
    y_proba = []
    for m in ensemble:
        y_proba.append(m.predict_proba(X))
    y_proba = np.mean(y_proba, axis=0)
    y_pred = y_proba[:, 1] > threshold
    
    return y_proba, y_pred


def evaluate_ensemble(ensemble, X, y, threshold=0.5, verbose=True):
    y_proba, y_pred = predict_ensemble(ensemble, X, y)
    if verbose:
        print(classification_report(y, y_pred, digits=3))
        print(f"auroc {roc_auc_score(y, y_proba[:, 1]):.3f}")
        print(f"brier {brier_score_loss(y, y_proba[:, 1]):.3f}")
        print(confusion_matrix(y, y_pred))
  
    return f1_score(y, y_pred)


def find_best_ensemble(ensemble, X_valid, y_valid):
    results = []
    while len(ensemble) > 1:
        tmp_res = []
        for m in ensemble:
            tmp = copy.copy(ensemble)
            tmp.remove(m)
            names = [_name for _name, _m in tmp]
            tmp = [_m for _name, _m in tmp]
            acc = evaluate_ensemble(tmp, X_valid, y_valid, verbose=False)
            results.append((names, tmp, acc))
            tmp_res.append((m, acc))

        m, _ = max(tmp_res, key=lambda item:item[1])
        ensemble.remove(m)
        # print(m)
        
    return max(results, key=lambda item:item[2])