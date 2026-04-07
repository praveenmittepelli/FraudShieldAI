"""
Evaluation utilities module.

Provides standardized evaluation metrics specifically targeted towards highly imbalanced
datasets (using PR-AUC instead of standard ROC-AUC) and visualization tools like 
confusion matrices and Precision-Recall plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    precision_recall_curve, 
    auc, 
    classification_report
)
import numpy as np

def evaluate_predictions(y_true, y_pred, y_prob=None, model_name="Model", save_dir=None):
    """
    Evaluates predictions and prints standard metrics for imbalanced datasets.
    
    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (optional).
        model_name: Name of the model for display purposes.
        save_dir: Directory to optionally save the classification report.
        
    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    print(f"--- Evaluation for {model_name} ---")
    
    # Standard threshold metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f'classification_report_{model_name.lower()}.txt')
        with open(report_path, 'w') as f:
            f.write(f"--- Evaluation for {model_name} ---\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    # PR-AUC if probabilities are provided
    if y_prob is not None:
        pr_auc = calculate_pr_auc(y_true, y_prob)
        print(f"PR-AUC:    {pr_auc:.4f}")
        metrics['pr_auc'] = pr_auc
        
        if save_dir:
            import os
            report_path = os.path.join(save_dir, f'classification_report_{model_name.lower()}.txt')
            with open(report_path, 'a') as f:
                f.write(f"\nPR-AUC:    {pr_auc:.4f}\n")
                
    return metrics

def calculate_pr_auc(y_true, y_prob):
    """Calculates the Precision-Recall Area Under Curve (PR-AUC)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plots a confusion matrix using Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'], 
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_precision_recall_curve(y_true, y_prob, model_name="Model", save_path=None):
    """
    Plots the Precision-Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    
    # Add a No Skill baseline line
    no_skill = np.sum(y_true == 1) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='red', label='No Skill')
    
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate_models_comparison(results_dict, y_true, save_dir=None, print_metrics=True):
    """
    Evaluates and compares multiple models on a single plot.
    
    Args:
        results_dict: A dictionary of the format {'model_name': {'y_pred': array, 'y_prob': array}}
        y_true: True binary labels
        save_dir: directory to save the output visualizations
        print_metrics: Whether to evaluate and print metrics for each model
    """
    import os
    
    plt.figure(figsize=(10, 8))
    print("=== Model Comparison Curve ===")
    
    for name, result in results_dict.items():
        y_pred = result.get('y_pred')
        y_prob = result.get('y_prob')
        
        # Print metrics to console
        if print_metrics:
            evaluate_predictions(y_true, y_pred, y_prob, model_name=name, save_dir=save_dir)
            print("\n")
        
        # Plot individual Confusion Matrices
        if y_pred is not None:
            cm_path = os.path.join(save_dir, f'cm_{name}.png') if save_dir else None
            plot_confusion_matrix(y_true, y_pred, model_name=name, save_path=cm_path)
            
        # Plot to the shared PR Curve
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')
            
    # Finalize and show shared PR Curve
    no_skill = np.sum(y_true == 1) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'pr_curve_comparison.png'))
    plt.show()
