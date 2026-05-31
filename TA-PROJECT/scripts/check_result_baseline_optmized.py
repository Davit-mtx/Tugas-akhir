
import pandas as pd
import numpy as np

df_base = pd.read_csv("summary_all_results.csv")
df_opt = pd.read_csv("summary_optimized_results.csv")

print("Baseline rows:", len(df_base))
print("Optimized rows:", len(df_opt))

print("\nBaseline category count:")
print(df_base["Category"].value_counts())

print("\nOptimized category count:")
print(df_opt["Category"].value_counts())

merged = pd.merge(df_base, df_opt, on=["File Name", "Category"], suffixes=("_base", "_opt"))
print("\nMerged rows:", len(merged))

print("\nLossless baseline:")
print(df_base["Lossless"].value_counts())

print("\nLossless optimized:")
print(df_opt["Lossless"].value_counts())

rgb_cols = [
    "Entropy_R", "Entropy_G", "Entropy_B", "Entropy",
    "NPCR_R (%)", "NPCR_G (%)", "NPCR_B (%)", "NPCR (%)",
    "UACI_R (%)", "UACI_G (%)", "UACI_B (%)", "UACI (%)"
]

print("\nMissing RGB columns baseline:")
print([c for c in rgb_cols if c not in df_base.columns])

print("\nMissing RGB columns optimized:")
print([c for c in rgb_cols if c not in df_opt.columns])