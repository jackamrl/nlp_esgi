"""
Script pour diviser le dataset en train (80%) et test (20%)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
input_file = "data/raw/names_train - names_train.csv"
df = pd.read_csv(input_file)

print(f"Nombre total de lignes: {len(df)}")
print(f"Distribution de is_comic:\n{df['is_comic'].value_counts()}")

# Diviser en train (80%) et test (20%)
# stratify=y pour maintenir la même proportion de classes dans train et test
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,  
    stratify=df['is_comic']  
)

# Sauvegarder les fichiers
train_file = "data/raw/train.csv"
test_file = "data/raw/test.csv"

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"\n Fichiers créés:")
print(f"  - Train: {train_file} ({len(train_df)} lignes)")
print(f"  - Test: {test_file} ({len(test_df)} lignes)")
print(f"\nDistribution dans train:\n{train_df['is_comic'].value_counts()}")
print(f"\nDistribution dans test:\n{test_df['is_comic'].value_counts()}")

