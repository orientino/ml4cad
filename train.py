import numpy as np
from sklearn.metrics import classification_report, f1_score, fbeta_score, make_scorer, accuracy_score, confusion_matrix, plot_confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve


def report(results, n_top=3):
    """Utility function to report the best scores."""

    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates[:1]:
            print("Model rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            

def evaluate(pipe, X, y, plot=False):
    """Evaluate models."""

    y_pred = pipe.predict(X)
    print(classification_report(y, y_pred, digits=3))
    print(f"auc macro {roc_auc_score(y, pipe.predict_proba(X)[:, 1]):.3f}")

    if plot:
        plot_confusion_matrix(pipe, X, y, normalize=None, values_format = '')
        plt.grid(False)
    else:
        print("confusion matrix")
        print(confusion_matrix(y, y_pred))


def train_and_evaluate(
    preprocess, 
    model, 
    hyperparams, 
    X_train, 
    y_train, 
    X_valid, 
    y_valid, 
    scoring="f1_macro", 
    iter=5000, 
    save=False, 
    savename=""
):
    """Train and evaluation pipeline."""
    pipe = Pipeline(steps=[
        ('preprocess', preprocess), 
        ('model', model)
    ])

    rand = RandomizedSearchCV(pipe,
                              param_distributions=hyperparams,
                              n_iter=iter,
                              scoring=scoring,
                              cv=2,
                              n_jobs=-1,    # use all processors
                              refit=True,   # refit the best model at the end
                              return_train_score=True,
                              verbose=True).fit(X_train, y_train)
    
    evaluate(rand.best_estimator_, X_train, y_train)
    evaluate(rand.best_estimator_, X_valid, y_valid)
    report(rand.cv_results_, n_top=5)

    if save:
        dump(rand.best_estimator_, f"{path_models}{savename}{suffix}.joblib")
    
    return rand.best_estimator_