"""
fairness_audit.py

Author: Amanuel Alemu Zewdu
Group Members: [Add group member names here]

Purpose:
A demonstration fairness audit script for the COMPAS dataset. This script computes common fairness metrics:
 - Disparate Impact Ratio
 - Equal Opportunity Difference
 - Demographic Parity Difference

Requirements:
 - pandas, numpy, scikit-learn, matplotlib
 - (Optional) AI Fairness 360 (aif360) for advanced metrics and mitigation methods
Install:
 pip install pandas numpy scikit-learn matplotlib
 # Optional: pip install aif360
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def generate_synthetic_compas(n=1000, seed=0):
    rng = np.random.RandomState(seed)
    race = rng.choice(['Black','White'], size=n, p=[0.45,0.55])
    base_prob = np.where(race=='Black', 0.35, 0.20)
    label = (rng.rand(n) < base_prob).astype(int)
    pred_noise = rng.rand(n)
    pred = ((label==1) | (pred_noise < (0.05 + (race=='Black')*0.10))).astype(int)
    df = pd.DataFrame({'race': race, 'label': label, 'pred': pred})
    return df

def disparate_impact(df, protected_attr='race', privileged_value='White'):
    groups = df.groupby(protected_attr)
    rates = groups['pred'].mean()
    unprivileged = rates.drop(privileged_value).mean()
    privileged = rates.loc[privileged_value]
    return unprivileged / privileged if privileged > 0 else np.nan

def equal_opportunity_difference(df, protected_attr='race', privileged_value='White'):
    results = {}
    for group, gdf in df.groupby(protected_attr):
        tn, fp, fn, tp = confusion_matrix(gdf['label'], gdf['pred'], labels=[0,1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results[group] = tpr
    unprivileged_groups = [g for g in results.keys() if g != privileged_value]
    tpr_unpriv = np.mean([results[g] for g in unprivileged_groups])
    tpr_priv = results.get(privileged_value, 0.0)
    return tpr_unpriv - tpr_priv, results

def demographic_parity_difference(df, protected_attr='race', privileged_value='White'):
    groups = df.groupby(protected_attr)
    rates = groups['pred'].mean().to_dict()
    unprivileged = np.mean([v for k,v in rates.items() if k != privileged_value])
    privileged = rates.get(privileged_value, 0.0)
    return unprivileged - privileged, rates

def run_audit():
    df = generate_synthetic_compas(n=2000)
    print("Sample counts by group:")
    print(df['race'].value_counts())
    print("\nOverall positive prediction rate:", df['pred'].mean())
    di = disparate_impact(df, protected_attr='race', privileged_value='White')
    eod, tpr_details = equal_opportunity_difference(df, protected_attr='race', privileged_value='White')
    dpd, rates = demographic_parity_difference(df, protected_attr='race', privileged_value='White')
    print(f"\nDisparate Impact Ratio (unprivileged / privileged): {di:.3f}")
    print(f"Equal Opportunity Difference (TPR_unpriv - TPR_priv): {eod:.3f}")
    print(f"Demographic Parity Difference (rate_unpriv - rate_priv): {dpd:.3f}")
    print("\nTPR by group:", tpr_details)
    print("Prediction rates by group:", rates)
    try:
        import matplotlib.pyplot as plt
        groups = list(rates.keys())
        vals = [rates[g] for g in groups]
        plt.bar(groups, vals)
        plt.title("Prediction Rate by Race Group")
        plt.ylabel("P(pred=1)")
        plt.savefig("prediction_rate_by_group.png")
        print("Saved visualization: prediction_rate_by_group.png")
    except Exception as e:
        print("Matplotlib visualization failed:", e)

if __name__ == '__main__':
    run_audit()
