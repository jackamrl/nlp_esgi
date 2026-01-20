import click
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from data import make_dataset
from feature import make_features
from models import make_model

@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y, vectorizer = make_features(df)

    model = make_model()
    model.fit(X, y)

    # Sauvegarder le modèle ET le vectorizer ensemble
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    
    return joblib.dump(model_data, model_dump_filename)


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def predict(input_filename, model_dump_filename, output_filename):
    # Charger le modèle et le vectorizer
    model_data = joblib.load(model_dump_filename)
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    
    # Charger les données à prédire
    df = make_dataset(input_filename)
    
    # Transformer les données avec le MÊME vectorizer qu'en train
    X, _, _ = make_features(df, vectorizer=vectorizer)
    
    # Faire les prédictions
    predictions = model.predict(X)
    
    # Créer un DataFrame avec les résultats
    results_df = pd.DataFrame({
        'video_name': df['video_name'],
        'predicted_is_comic': predictions
    })
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    # Sauvegarder les prédictions
    results_df.to_csv(output_filename, index=False)
    
    print(f"Prédictions sauvegardées dans {output_filename}")
    print(f"   Nombre de prédictions: {len(predictions)}")
    print(f"   Distribution: {pd.Series(predictions).value_counts().to_dict()}")
    
    return results_df


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y, _ = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y, cv=5):
    """
    Évalue le modèle avec k-fold cross-validation.
    
    Args:
        model: Le modèle à évaluer
        X: Features
        y: Labels
        cv: Nombre de folds pour la cross-validation (défaut: 5)
    """
    print("=" * 60)
    print("ÉVALUATION DU MODÈLE - K-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    # Convertir X en array dense si c'est une matrice sparse
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Stratified K-Fold pour maintenir la proportion de classes
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Scores de cross-validation
    print(f"\n Cross-Validation ({cv} folds):")
    print("-" * 60)
    
    # Accuracy
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    print(f"Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std() * 2:.4f})")
    print(f"  Détails par fold: {[f'{s:.4f}' for s in accuracy_scores]}")
    
    # Precision
    precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
    print(f"\nPrecision: {precision_scores.mean():.4f} (+/- {precision_scores.std() * 2:.4f})")
    
    # Recall
    recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
    print(f"Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std() * 2:.4f})")
    
    # F1-score
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    print(f"F1-score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    
    print("\n" + "=" * 60)
    
    # Entraîner sur toutes les données et afficher un rapport détaillé
    print("\nRapport de classification (entraînement sur toutes les données):")
    print("-" * 60)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    print(classification_report(y, y_pred, target_names=['Non-comic', 'Comic']))
    
    print("=" * 60)
    
    return {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'precision_mean': precision_scores.mean(),
        'recall_mean': recall_scores.mean(),
        'f1_mean': f1_scores.mean()
    }


cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
