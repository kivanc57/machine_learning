import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_train_vect, y_train, X_test_vect, y_test):
  # Evaluate using cross-validation (10-fold)
  cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
  scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
  cross_val_results = cross_validate(model, X_train_vect, y_train, cv=cv, scoring=scoring)

  print(f"Cross-validation results:")
  print(f"Accuracy: {np.mean(cross_val_results['test_accuracy'])}")
  print(f"Precision: {np.mean(cross_val_results['test_precision_macro'])}")
  print(f"Recall: {np.mean(cross_val_results['test_recall_macro'])}")
  print(f"F1 Score: {np.mean(cross_val_results['test_f1_macro'])}")

  y_pred = model.predict(X_test_vect)
  cm = confusion_matrix(y_test, y_pred)
  print("Confusion Matrix:")
  print(cm)

  return cross_val_results, cm
