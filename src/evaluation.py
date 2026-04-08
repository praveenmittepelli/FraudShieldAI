"""
Evaluation Visualization & Reporting Module.

Calculates stringent statistics regarding Imbalanced Detection mapping.
In fraud detection, general Accuracy is useless (a model predicting '0' constantly achieves 99% accuracy).
This module heavily emphasizes Precision/Recall curves, F1-Scores, and confusion matrices instead.
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
    Evaluates absolute mathematical predictions against ground truth limits and establishes bounds.
    
    Args:
        y_true (np.ndarray): True physical dataset labels (1=Fraud, 0=Genuine).
        y_pred (np.ndarray): Binary evaluated system labels dictating a specific cut-off threshold.
        y_prob (np.ndarray, optional): Uncapped numeric probabilities describing raw output.
        model_name (str): The logical system identifier mapped to logs.
        save_dir (str, optional): Physical absolute or relative filesystem directory to push exported `.txt` outputs.
        
    Returns:
        dict: Standard metric numerical aggregates compiled securely.
    """
    print(f"--- Evaluation for {model_name} ---")
    
    # Calculate macro thresholds while enforcing fallback protections
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    
    # The classification_report method yields detailed breakdowns separated independently by True class types
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)
    
    # Synchronize textual export mechanism natively ensuring persistence
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
    
    # If unscaled outputs apply, dynamically construct PR-AUC
    if y_prob is not None:
        pr_auc = calculate_pr_auc(y_true, y_prob)
        print(f"PR-AUC:    {pr_auc:.4f}")
        metrics['pr_auc'] = pr_auc
        
        # Append AUC calculations explicitly backwards into the logged object structure
        if save_dir:
            import os
            report_path = os.path.join(save_dir, f'classification_report_{model_name.lower()}.txt')
            with open(report_path, 'a') as f:
                f.write(f"\nPR-AUC:    {pr_auc:.4f}\n")
                
    return metrics

def calculate_pr_auc(y_true, y_prob):
    """
    Safely calculates the normalized Area Under Curve representation representing Precision vs Recall trade-offs.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Generates a localized visual heatmap representation denoting false positives vs true negatives securely metrics.
    
    Args:
        y_true (np.ndarray): Truth vector.
        y_pred (np.ndarray): Bounded array.
        model_name (str): Identifer overlay mapping.
        save_path (str, optional): Outfile path.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    # Utilize Seaborn mapped colors directly via Matplotlib instance context wrappers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Fraud', 'Fraud'], 
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    # Note: plt.show ignores headless/deploy wrappers inherently
    plt.close() # Close memory handle silently rather than render interactively

def plot_precision_recall_curve(y_true, y_prob, model_name="Model", save_path=None):
    """
    Plots a standard PR Curve evaluating the system tradeoff. Useful for single isolated algorithm validation architectures.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    
    # Establish arbitrary 'No Skill' deterministic boundary tracking relative representation percentages
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
    plt.close()

def evaluate_models_comparison(results_dict, y_true, save_dir=None, print_metrics=True):
    """
    Executes deep comparison evaluations aggregating massive plot overlays and optionally outputting local textual reports.
    
    Args:
        results_dict (dict): Encapsulating master mapped structure defining properties for models. Structure format expected:
                             {'AlgorithmStringDefinition': {'y_pred': [...], 'y_prob': [...]}}
        y_true (np.ndarray): Truth labels.
        save_dir (str): Base root location directory defining visual saves.
        print_metrics (bool): Overwrite capability dictating if identical console printing runs recursively. 
    """
    import os
    
    plt.figure(figsize=(10, 8))
    print("=== Model Comparison Curve ===")
    
    for name, result in results_dict.items():
        y_pred = result.get('y_pred')
        y_prob = result.get('y_prob')
        
        # Trigger internal calculations explicitly if flags are open
        if print_metrics:
            evaluate_predictions(y_true, y_pred, y_prob, model_name=name, save_dir=save_dir)
            print("\n")
        
        # Plot and save internal isolation matrices
        if y_pred is not None:
            cm_path = os.path.join(save_dir, f'cm_{name.lower()}.png') if save_dir else None
            plot_confusion_matrix(y_true, y_pred, model_name=name, save_path=cm_path)
            
        # Draw explicit PR overlapping logic
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{name} (PR-AUC = {pr_auc:.3f})')
            
    # Compile base graphical overlays matching structure bounds for all algorithms loaded securely
    no_skill = np.sum(y_true == 1) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='black', label='No Skill')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'pr_curve_comparison.png'))
    plt.close()
