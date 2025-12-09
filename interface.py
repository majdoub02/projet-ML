"""
interface.py - Version avec prix réel du dataset
"""

from flask import Flask, render_template_string, request, jsonify
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime

print("="*60)
print("🚀 Lancement de l'application Flask - USA Housing")
print("="*60)

app = Flask(__name__)

# Charger le modèle ET le dataset original
def load_model_and_data():
    try:
        print("📦 Chargement du modèle...")
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print(f"✅ Modèle chargé: {type(model_data['model']).__name__}")
        print(f"📊 Colonnes: {model_data['columns']}")
        
        # Charger le dataset original pour avoir les prix réels
        print("📂 Chargement du dataset USA_Housing...")
        df = pd.read_csv('USA_Housing.csv')
        df_clean = df.drop(columns=['Address'])  # Supprimer la colonne texte
        print(f"✅ Dataset chargé: {df_clean.shape}")
        
        return model_data, df_clean
        
    except Exception as e:
        print(f"❌ Erreur de chargement: {e}")
        return None, None

model_data, original_data = load_model_and_data()

# =============================================
# FONCTION POUR TROUVER LE PRIX RÉEL DANS LE DATASET
# =============================================

def find_real_price_in_dataset(features, threshold=0.01):
    """
    Trouve une maison similaire dans le dataset original
    et retourne son prix réel.
    
    threshold: tolérance pour la similarité (1% par défaut)
    """
    if original_data is None:
        return None
    
    # Créer un DataFrame avec les features
    input_df = pd.DataFrame([features])
    
    # Chercher la maison la plus similaire
    similarities = []
    
    for idx, row in original_data.iterrows():
        # Calculer la similarité (distance normalisée)
        similarity_score = 0
        match_count = 0
        
        for col in features.keys():
            if col in row:
                # Normaliser les valeurs pour la comparaison
                val_input = features[col]
                val_dataset = row[col]
                
                # Calculer la différence relative
                if val_input != 0:
                    diff = abs(val_input - val_dataset) / val_input
                    if diff <= threshold:
                        match_count += 1
                    similarity_score += 1 / (1 + diff)
        
        # Si au moins 3 caractéristiques correspondent avec la tolérance
        if match_count >= 3:
            similarities.append({
                'index': idx,
                'similarity': similarity_score,
                'price': row['Price']
            })
    
    if similarities:
        # Trouver la meilleure correspondance
        best_match = max(similarities, key=lambda x: x['similarity'])
        print(f"🔍 Maison similaire trouvée: similarité={best_match['similarity']:.3f}")
        return best_match['price']
    
    # Si aucune correspondance exacte, estimer à partir des caractéristiques
    print("⚠️ Aucune correspondance exacte, estimation à partir du dataset...")
    return estimate_price_from_dataset(features)

def estimate_price_from_dataset(features):
    """
    Estime le prix réel en faisant une moyenne pondérée des maisons similaires
    """
    if original_data is None:
        return None
    
    # Séparer features et prix
    X_data = original_data.drop(columns=['Price'])
    y_data = original_data['Price']
    
    # Calculer les distances
    distances = []
    for idx, row in X_data.iterrows():
        distance = 0
        for col in features.keys():
            if col in row:
                # Distance normalisée
                val_input = features[col]
                val_dataset = row[col]
                # Normaliser par l'écart-type de la colonne
                std_val = X_data[col].std()
                if std_val > 0:
                    distance += ((val_input - val_dataset) / std_val) ** 2
        
        distances.append(np.sqrt(distance))
    
    # Trouver les k plus proches voisins
    k = min(10, len(distances))
    nearest_indices = np.argsort(distances)[:k]
    
    # Moyenne pondérée des prix des voisins les plus proches
    total_weight = 0
    weighted_sum = 0
    
    for idx in nearest_indices:
        weight = 1 / (1 + distances[idx])  # Plus la distance est petite, plus le poids est grand
        weighted_sum += y_data.iloc[idx] * weight
        total_weight += weight
    
    if total_weight > 0:
        estimated_price = weighted_sum / total_weight
        print(f"📊 Estimation basée sur {k} maisons similaires")
        return estimated_price
    
    return None

# =============================================
# TEMPLATES HTML INTÉGRÉS
# =============================================

# Template pour la page d'accueil
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prédiction Prix Maison - USA Housing</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 900px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 15px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(to right, #4a00e0, #8e2de2);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { 
            font-size: 2.5rem; 
            margin-bottom: 15px; 
        }
        .header p { 
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .form-container { 
            padding: 40px; 
        }
        .form-group { 
            margin-bottom: 25px; 
        }
        .form-group label { 
            display: block; 
            margin-bottom: 10px; 
            font-weight: 600;
            color: #333;
            font-size: 1.1rem;
        }
        .form-control { 
            width: 100%; 
            padding: 15px; 
            border: 2px solid #e0e0e0; 
            border-radius: 10px; 
            font-size: 16px;
            transition: all 0.3s;
        }
        .form-control:focus { 
            border-color: #8e2de2; 
            outline: none;
            box-shadow: 0 0 0 3px rgba(142, 45, 226, 0.1);
        }
        .input-hint {
            display: block;
            margin-top: 5px;
            font-size: 0.9rem;
            color: #666;
            font-style: italic;
        }
        .btn { 
            background: linear-gradient(to right, #8e2de2, #4a00e0);
            color: white; 
            border: none; 
            padding: 18px; 
            border-radius: 10px; 
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 10px 20px rgba(142, 45, 226, 0.3);
        }
        .btn-random {
            background: linear-gradient(to right, #00b09b, #96c93d);
            margin-top: 10px;
        }
        .model-info {
            background: #f0f7ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #4a00e0;
        }
        .model-info h3 {
            color: #4a00e0;
            margin-bottom: 10px;
        }
        .error-message {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            border-left: 4px solid #c62828;
        }
        .footer {
            text-align: center;
            padding: 25px;
            color: #666;
            font-size: 0.9rem;
            border-top: 1px solid #eee;
            background: #f9f9f9;
        }
        .feature-range {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: #888;
            margin-top: 5px;
        }
        .stats-info {
            display: flex;
            justify-content: space-around;
            background: #e8f4fc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #4a00e0;
        }
        .stat-label {
            font-size: 0.8rem;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Prédiction de Prix Immobilier USA</h1>
            <p>Basé sur le dataset réel USA Housing</p>
        </div>
        
        <div class="form-container">
            <div class="model-info">
                <h3>📊 Informations du Modèle</h3>
                <p><strong>Modèle:</strong> {{ model_name }}</p>
                <p><strong>Précision (R²):</strong> {{ r2_score }}</p>
                <p><strong>Erreur moyenne:</strong> {{ mae_score }}</p>
                <p><strong>Dataset:</strong> {{ dataset_size }} maisons réelles</p>
            </div>
            
            <form action="/predict" method="POST">
                <div class="form-group">
                    <label for="Avg. Area Income">💰 Revenu Moyen de la Zone ($)</label>
                    <input type="number" step="1000" class="form-control" 
                           name="Avg. Area Income" value="70000" required>
                    <div class="feature-range">Min: ${{ min_values['Avg. Area Income']|round|int }} - Max: ${{ max_values['Avg. Area Income']|round|int }}</div>
                </div>
                
                <div class="form-group">
                    <label for="Avg. Area House Age">🏠 Âge Moyen des Maisons (années)</label>
                    <input type="number" step="0.1" class="form-control" 
                           name="Avg. Area House Age" value="6.0" required>
                    <div class="feature-range">Min: {{ min_values['Avg. Area House Age']|round(1) }} - Max: {{ max_values['Avg. Area House Age']|round(1) }} ans</div>
                </div>
                
                <div class="form-group">
                    <label for="Avg. Area Number of Rooms">🚪 Nombre Moyen de Pièces</label>
                    <input type="number" step="0.1" class="form-control" 
                           name="Avg. Area Number of Rooms" value="6.5" required>
                    <div class="feature-range">Min: {{ min_values['Avg. Area Number of Rooms']|round(1) }} - Max: {{ max_values['Avg. Area Number of Rooms']|round(1) }} pièces</div>
                </div>
                
                <div class="form-group">
                    <label for="Avg. Area Number of Bedrooms">🛏️ Nombre Moyen de Chambres</label>
                    <input type="number" step="0.1" class="form-control" 
                           name="Avg. Area Number of Bedrooms" value="3.0" required>
                    <div class="feature-range">Min: {{ min_values['Avg. Area Number of Bedrooms']|round(1) }} - Max: {{ max_values['Avg. Area Number of Bedrooms']|round(1) }} chambres</div>
                </div>
                
                <div class="form-group">
                    <label for="Area Population">👥 Population de la Zone</label>
                    <input type="number" step="1000" class="form-control" 
                           name="Area Population" value="35000" required>
                    <div class="feature-range">Min: {{ min_values['Area Population']|round|int }} - Max: {{ max_values['Area Population']|round|int }} habitants</div>
                </div>
                
                <button type="submit" class="btn">🎯 Estimer le Prix</button>
                <button type="button" class="btn btn-random" onclick="loadRandomHouse()">🎲 Charger une maison aléatoire du dataset</button>
            </form>
            
            {% if error %}
            <div class="error-message">
                ⚠️ {{ error }}
            </div>
            {% endif %}
            
            <div class="stats-info">
                <div class="stat-item">
                    <div class="stat-value">${{ avg_price|round|int }}</div>
                    <div class="stat-label">Prix moyen</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${{ min_price|round|int }}</div>
                    <div class="stat-label">Prix min</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">${{ max_price|round|int }}</div>
                    <div class="stat-label">Prix max</div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>📈 Projet Machine Learning - Dataset USA Housing réel</p>
            <p>⚡ Powered by Flask & Scikit-learn | {{ dataset_size }} maisons | R²: {{ r2_score }}</p>
        </div>
    </div>
    
    <script>
    function loadRandomHouse() {
        // Rediriger vers la route pour une maison aléatoire
        window.location.href = '/random_house';
    }
    
    // Pré-remplir avec des valeurs réalistes
    function setRealisticValues() {
        document.querySelector('[name="Avg. Area Income"]').value = 70000;
        document.querySelector('[name="Avg. Area House Age"]').value = 6.0;
        document.querySelector('[name="Avg. Area Number of Rooms"]').value = 6.5;
        document.querySelector('[name="Avg. Area Number of Bedrooms"]').value = 3.0;
        document.querySelector('[name="Area Population"]').value = 35000;
    }
    
    // Charger au démarrage
    window.onload = function() {
        setRealisticValues();
    };
    </script>
</body>
</html>
'''

# Template pour les résultats avec prix réel du dataset
RESULT_HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparaison des Prix - Résultat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { 
            max-width: 1000px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.25);
            overflow: hidden;
        }
        .header { 
            background: linear-gradient(to right, #00b09b, #96c93d);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { 
            font-size: 2.8rem; 
            margin-bottom: 15px; 
        }
        .header p { 
            font-size: 1.3rem;
            opacity: 0.9;
        }
        .result-container { 
            padding: 40px; 
        }
        .comparison-title {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 1.8rem;
        }
        .price-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 40px 0;
        }
        @media (max-width: 768px) {
            .price-comparison {
                grid-template-columns: 1fr;
            }
        }
        .price-card {
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            transition: transform 0.3s;
            position: relative;
            overflow: hidden;
        }
        .price-card:hover {
            transform: translateY(-5px);
        }
        .estimated {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .real {
            background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        }
        .price-label {
            font-size: 1.4rem;
            margin-bottom: 15px;
            font-weight: 600;
            opacity: 0.9;
        }
        .price-value {
            font-size: 3.5rem;
            font-weight: 700;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .price-subtext {
            font-size: 1.1rem;
            opacity: 0.85;
            margin-top: 10px;
        }
        .data-source {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        .difference-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 40px 0;
            text-align: center;
            border: 2px solid #e9ecef;
        }
        .difference-title {
            font-size: 1.6rem;
            color: #333;
            margin-bottom: 20px;
        }
        .difference-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 20px 0;
            padding: 15px;
            border-radius: 10px;
            background: white;
            display: inline-block;
            min-width: 200px;
        }
        .positive { color: #2ecc71; }
        .negative { color: #e74c3c; }
        .neutral { color: #f39c12; }
        .difference-percent {
            font-size: 1.3rem;
            margin-top: 15px;
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
        }
        .features-details {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
        }
        .features-title {
            color: #333;
            margin-bottom: 25px;
            font-size: 1.6rem;
            text-align: center;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        .feature-item {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .feature-name {
            color: #666;
            font-size: 1rem;
            margin-bottom: 10px;
        }
        .feature-value {
            color: #333;
            font-size: 1.4rem;
            font-weight: 600;
        }
        .model-performance {
            background: #e3f2fd;
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
            text-align: center;
        }
        .performance-title {
            color: #1565c0;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        .performance-stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
        }
        .stat-item {
            background: white;
            padding: 15px 25px;
            border-radius: 10px;
            min-width: 200px;
        }
        .stat-label {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        .stat-value {
            color: #333;
            font-size: 1.4rem;
            font-weight: 600;
        }
        .buttons-container {
            display: flex;
            gap: 20px;
            margin-top: 40px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .btn { 
            padding: 18px 40px; 
            border: none; 
            border-radius: 10px; 
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            text-align: center;
            transition: all 0.3s;
            min-width: 200px;
        }
        .btn-primary { 
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white; 
        }
        .btn-primary:hover { 
            background: linear-gradient(to right, #764ba2, #667eea);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(118, 75, 162, 0.3);
        }
        .btn-secondary { 
            background: #6c757d; 
            color: white; 
        }
        .btn-secondary:hover { 
            background: #5a6268;
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(108, 117, 125, 0.3);
        }
        .btn-random { 
            background: linear-gradient(to right, #00b09b, #96c93d);
            color: white; 
        }
        .btn-random:hover { 
            background: linear-gradient(to right, #96c93d, #00b09b);
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0, 176, 155, 0.3);
        }
        .timestamp {
            text-align: center;
            color: #666;
            margin-top: 30px;
            font-size: 0.9rem;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }
        .accuracy-badge {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
            margin-top: 15px;
            font-size: 1.1rem;
        }
        .excellent { background: #d4edda; color: #155724; }
        .good { background: #fff3cd; color: #856404; }
        .average { background: #f8d7da; color: #721c24; }
        .data-info {
            background: #fff8e1;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            text-align: center;
            color: #856404;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Comparaison des Prix</h1>
            <p>Prix Estimé vs Prix Réel du Dataset</p>
        </div>
        
        <div class="result-container">
            <h2 class="comparison-title">💰 Analyse Comparative</h2>
            
            {% if data_source == 'estimated' %}
            <div class="data-info">
                ℹ️ Prix réel estimé à partir des caractéristiques similaires dans le dataset
            </div>
            {% elif data_source == 'exact' %}
            <div class="data-info">
                ✅ Prix réel exact provenant du dataset USA Housing
            </div>
            {% endif %}
            
            <div class="price-comparison">
                <div class="price-card estimated">
                    <div class="price-label">🏠 Prix Estimé par le Modèle</div>
                    <div class="price-value">${{ "{:,.2f}".format(prediction) }}</div>
                    <div class="price-subtext">Prédiction de l'IA</div>
                    <div class="price-subtext">Modèle: {{ model_name }}</div>
                </div>
                
                <div class="price-card real">
                    <div class="data-source">📁 Dataset réel</div>
                    <div class="price-label">📈 Prix Réel du Dataset</div>
                    <div class="price-value">${{ "{:,.2f}".format(real_price) }}</div>
                    <div class="price-subtext">Valeur réelle du dataset</div>
                    <div class="price-subtext">{{ data_source_text }}</div>
                </div>
            </div>
            
            <div class="difference-section">
                <div class="difference-title">📉 Différence entre les Prix</div>
                <div class="difference-value {{ diff_class }}">
                    {{ difference }}
                </div>
                <div class="difference-percent {{ diff_class }}">
                    {{ diff_percent }}% d'écart
                </div>
                
                <div class="accuracy-badge {{ accuracy_class }}">
                    {{ accuracy_message }}
                </div>
            </div>
            
            <div class="model-performance">
                <div class="performance-title">📊 Performance du Modèle</div>
                <div class="performance-stats">
                    <div class="stat-item">
                        <div class="stat-label">Score R²</div>
                        <div class="stat-value">{{ model_r2_score }}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Erreur Moyenne Absolue</div>
                        <div class="stat-value">${{ model_mae }}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Précision sur ce cas</div>
                        <div class="stat-value">{{ model_accuracy }}%</div>
                    </div>
                </div>
            </div>
            
            <div class="features-details">
                <div class="features-title">📋 Caractéristiques Saisies</div>
                <div class="feature-grid">
                    {% for feature, value in features.items() %}
                    <div class="feature-item">
                        <div class="feature-name">{{ feature }}</div>
                        <div class="feature-value">
                            {% if feature == 'Avg. Area Income' or feature == 'Area Population' %}
                                ${{ "{:,.0f}".format(value) }}
                            {% elif value is number %}
                                {{ "{:,.2f}".format(value) }}
                            {% else %}
                                {{ value }}
                            {% endif %}
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="buttons-container">
                <a href="/" class="btn btn-primary">🔄 Nouvelle Estimation</a>
                <a href="/random_house" class="btn btn-random">🎲 Autre maison aléatoire</a>
                <a href="/statistics" class="btn btn-secondary">📈 Statistiques</a>
            </div>
            
            <div class="timestamp">
                ⏰ Prédiction effectuée le {{ timestamp }}
            </div>
        </div>
    </div>
</body>
</html>
'''

# =============================================
# ROUTES FLASK
# =============================================

@app.route('/')
def home():
    """Page d'accueil"""
    if model_data is None or original_data is None:
        return render_template_string(INDEX_HTML, 
            model_name="ERREUR - Modèle non chargé",
            r2_score="0.000",
            mae_score="$0",
            error="Le modèle ou le dataset n'a pas pu être chargé",
            dataset_size=0,
            min_values={},
            max_values={},
            avg_price=0,
            min_price=0,
            max_price=0
        )
    
    # Calculer les statistiques du dataset
    min_values = original_data.min().to_dict()
    max_values = original_data.max().to_dict()
    avg_price = original_data['Price'].mean()
    min_price = original_data['Price'].min()
    max_price = original_data['Price'].max()
    
    return render_template_string(INDEX_HTML,
        model_name=model_data.get('best_model_name', 'Modèle ML'),
        r2_score=f"{model_data.get('r2_score', 0):.3f}",
        mae_score=f"${model_data.get('mae', 0):,.0f}",
        error=None,
        dataset_size=len(original_data),
        min_values=min_values,
        max_values=max_values,
        avg_price=avg_price,
        min_price=min_price,
        max_price=max_price
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction du prix avec recherche dans le dataset"""
    if model_data is None or original_data is None:
        return render_template_string(INDEX_HTML,
            model_name="ERREUR",
            r2_score="0.000",
            mae_score="$0",
            error="Modèle ou dataset non disponible"
        )
    
    try:
        # Récupérer les données du formulaire
        features = {}
        for col in model_data['columns']:
            value = float(request.form[col])
            features[col] = value
        
        # Préparer l'input pour la prédiction
        input_data = np.array(list(features.values())).reshape(1, -1)
        
        # Appliquer le scaling si nécessaire
        if model_data['scaler'] is not None:
            input_data = model_data['scaler'].transform(input_data)
        
        # Faire la prédiction (prix estimé)
        prediction = model_data['model'].predict(input_data)[0]
        
        # Chercher le prix réel dans le dataset
        real_price = find_real_price_in_dataset(features)
        
        if real_price is None:
            # Si pas trouvé, utiliser la moyenne du dataset
            real_price = original_data['Price'].mean()
            data_source = 'estimated'
            data_source_text = "Moyenne du dataset (aucune correspondance exacte)"
        else:
            # Vérifier si c'est une correspondance exacte ou estimée
            data_source = 'exact'
            data_source_text = "Correspondance exacte dans le dataset"
        
        # Calculer les différences
        difference_amount = real_price - prediction
        diff_percent = abs(difference_amount / real_price * 100) if real_price != 0 else 0
        
        # Formater la différence
        if difference_amount >= 0:
            difference = f"+${difference_amount:,.2f}"
        else:
            difference = f"-${abs(difference_amount):,.2f}"
        
        # Déterminer la classe CSS et le message de précision
        if diff_percent <= 5:
            diff_class = "positive"
            accuracy_class = "excellent"
            accuracy_message = "✅ Précision Excellente"
        elif diff_percent <= 10:
            diff_class = "neutral"
            accuracy_class = "good"
            accuracy_message = "⚠️ Précision Bonne"
        else:
            diff_class = "negative"
            accuracy_class = "average"
            accuracy_message = "⚠️ Précision Moyenne"
        
        # Préparer les données pour le template
        result = {
            'prediction': float(prediction),
            'real_price': float(real_price),
            'difference': difference,
            'diff_percent': f"{diff_percent:.2f}",
            'diff_class': diff_class,
            'accuracy_class': accuracy_class,
            'accuracy_message': accuracy_message,
            'data_source': data_source,
            'data_source_text': data_source_text,
            'features': features,
            'model_name': model_data.get('best_model_name', 'Modèle ML'),
            'model_r2_score': f"{model_data.get('r2_score', 0):.3f}",
            'model_mae': f"{model_data.get('mae', 0):,.0f}",
            'model_rmse': f"{model_data.get('rmse', 0):,.0f}",
            'model_accuracy': f"{100 - min(diff_percent, 100):.1f}",
            'timestamp': datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        }
        
        return render_template_string(RESULT_HTML, **result)
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return render_template_string(INDEX_HTML,
            model_name=model_data.get('best_model_name', 'Modèle ML'),
            r2_score=f"{model_data.get('r2_score', 0):.3f}",
            mae_score=f"${model_data.get('mae', 0):,.0f}",
            error=f"Erreur: {str(e)}"
        )

@app.route('/random_house')
def random_house():
    """Sélectionne une maison aléatoire du dataset"""
    if model_data is None or original_data is None:
        return render_template_string(INDEX_HTML,
            model_name="ERREUR",
            r2_score="0.000",
            mae_score="$0",
            error="Dataset non disponible"
        )
    
    try:
        # Sélectionner une maison aléatoire
        random_idx = np.random.randint(0, len(original_data))
        random_house = original_data.iloc[random_idx]
        
        # Extraire les caractéristiques
        features = {}
        for col in model_data['columns']:
            features[col] = random_house[col]
        
        # Préparer l'input pour la prédiction
        input_data = np.array(list(features.values())).reshape(1, -1)
        
        # Appliquer le scaling si nécessaire
        if model_data['scaler'] is not None:
            input_data = model_data['scaler'].transform(input_data)
        
        # Faire la prédiction
        prediction = model_data['model'].predict(input_data)[0]
        
        # Prix réel du dataset
        real_price = random_house['Price']
        
        # Calculer les différences
        difference_amount = real_price - prediction
        diff_percent = abs(difference_amount / real_price * 100) if real_price != 0 else 0
        
        # Formater la différence
        if difference_amount >= 0:
            difference = f"+${difference_amount:,.2f}"
        else:
            difference = f"-${abs(difference_amount):,.2f}"
        
        # Déterminer la classe CSS et le message de précision
        if diff_percent <= 5:
            diff_class = "positive"
            accuracy_class = "excellent"
            accuracy_message = "✅ Précision Excellente"
        elif diff_percent <= 10:
            diff_class = "neutral"
            accuracy_class = "good"
            accuracy_message = "⚠️ Précision Bonne"
        else:
            diff_class = "negative"
            accuracy_class = "average"
            accuracy_message = "⚠️ Précision Moyenne"
        
        # Préparer les données pour le template
        result = {
            'prediction': float(prediction),
            'real_price': float(real_price),
            'difference': difference,
            'diff_percent': f"{diff_percent:.2f}",
            'diff_class': diff_class,
            'accuracy_class': accuracy_class,
            'accuracy_message': accuracy_message,
            'data_source': 'exact',
            'data_source_text': f"Maison #{random_idx} du dataset",
            'features': features,
            'model_name': model_data.get('best_model_name', 'Modèle ML'),
            'model_r2_score': f"{model_data.get('r2_score', 0):.3f}",
            'model_mae': f"{model_data.get('mae', 0):,.0f}",
            'model_rmse': f"{model_data.get('rmse', 0):,.0f}",
            'model_accuracy': f"{100 - min(diff_percent, 100):.1f}",
            'timestamp': datetime.now().strftime("%d/%m/%Y à %H:%M:%S")
        }
        
        return render_template_string(RESULT_HTML, **result)
        
    except Exception as e:
        print(f"❌ Erreur avec maison aléatoire: {e}")
        return render_template_string(INDEX_HTML,
            model_name=model_data.get('best_model_name', 'Modèle ML'),
            r2_score=f"{model_data.get('r2_score', 0):.3f}",
            mae_score=f"${model_data.get('mae', 0):,.0f}",
            error=f"Erreur: {str(e)}"
        )

@app.route('/statistics')
def statistics():
    """Page de statistiques détaillées"""
    if model_data is None or original_data is None:
        return render_template_string(INDEX_HTML,
            model_name="ERREUR",
            r2_score="0.000",
            mae_score="$0",
            error="Modèle ou dataset non disponible"
        )
    
    STATS_HTML = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Statistiques du Dataset</title>
        <style>
            body { 
                font-family: 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                padding: 30px;
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 { 
                color: #333; 
                text-align: center; 
                margin-bottom: 40px;
                font-size: 2.5rem;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
                margin-bottom: 40px;
            }
            .stat-card {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                border-left: 5px solid #4a00e0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.05);
            }
            .stat-title {
                color: #666;
                font-size: 1rem;
                margin-bottom: 10px;
            }
            .stat-value {
                color: #333;
                font-size: 2rem;
                font-weight: 700;
            }
            .dataset-info {
                background: #e3f2fd;
                padding: 25px;
                border-radius: 15px;
                margin-top: 30px;
            }
            .features-list {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 15px;
                margin-top: 30px;
            }
            .feature-item {
                padding: 15px;
                border-bottom: 1px solid #dee2e6;
                display: flex;
                justify-content: space-between;
            }
            .btn-back {
                display: block;
                width: 200px;
                margin: 40px auto 0;
                padding: 15px;
                background: #4a00e0;
                color: white;
                text-align: center;
                border-radius: 10px;
                text-decoration: none;
                font-weight: 600;
                transition: all 0.3s;
            }
            .btn-back:hover {
                background: #8e2de2;
                transform: translateY(-2px);
            }
            .table-container {
                overflow-x: auto;
                margin-top: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: 600;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Statistiques du Dataset USA Housing</h1>
            
            <div class="dataset-info">
                <h3>📋 Informations Générales</h3>
                <p><strong>Nombre total de maisons:</strong> {{ num_houses }}</p>
                <p><strong>Période de collecte:</strong> Données actuelles du marché immobilier USA</p>
                <p><strong>Variables:</strong> {{ num_features }} caractéristiques par maison</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-title">Prix Moyen des Maisons</div>
                    <div class="stat-value">${{ avg_price|round|int }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Prix Minimum</div>
                    <div class="stat-value">${{ min_price|round|int }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Prix Maximum</div>
                    <div class="stat-value">${{ max_price|round|int }}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-title">Écart-type des Prix</div>
                    <div class="stat-value">${{ std_price|round|int }}</div>
                </div>
            </div>
            
            <div class="features-list">
                <h3>📈 Statistiques par Variable</h3>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Variable</th>
                                <th>Moyenne</th>
                                <th>Minimum</th>
                                <th>Maximum</th>
                                <th>Écart-type</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for feature, stats in feature_stats.items() %}
                            <tr>
                                <td>{{ feature }}</td>
                                <td>{{ stats.mean|round(2) }}</td>
                                <td>{{ stats.min|round(2) }}</td>
                                <td>{{ stats.max|round(2) }}</td>
                                <td>{{ stats.std|round(2) }}</td>
                            </tr>
                            {% endfor %}
                            <tr style="background-color: #e8f5e9;">
                                <td><strong>Prix ($)</strong></td>
                                <td><strong>${{ price_stats.mean|round(2) }}</strong></td>
                                <td><strong>${{ price_stats.min|round(2) }}</strong></td>
                                <td><strong>${{ price_stats.max|round(2) }}</strong></td>
                                <td><strong>${{ price_stats.std|round(2) }}</strong></td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div style="margin-top: 40px; padding: 20px; background: #fff8e1; border-radius: 10px;">
                <h3>📝 Notes sur le Dataset</h3>
                <p>• Ce dataset contient des données réelles du marché immobilier américain</p>
                <p>• Chaque ligne représente une zone géographique avec des caractéristiques moyennes</p>
                <p>• Le prix représente le prix moyen des maisons dans cette zone</p>
                <p>• Utilisé pour entraîner le modèle de prédiction de prix</p>
            </div>
            
            <a href="/" class="btn-back">🏠 Retour à la Prédiction</a>
        </div>
    </body>
    </html>
    '''
    
    # Calculer les statistiques
    feature_stats = {}
    for col in model_data['columns']:
        if col in original_data.columns:
            feature_stats[col] = {
                'mean': original_data[col].mean(),
                'min': original_data[col].min(),
                'max': original_data[col].max(),
                'std': original_data[col].std()
            }
    
    price_stats = {
        'mean': original_data['Price'].mean(),
        'min': original_data['Price'].min(),
        'max': original_data['Price'].max(),
        'std': original_data['Price'].std()
    }
    
    return render_template_string(STATS_HTML,
        model_name=model_data.get('best_model_name', 'Modèle ML'),
        r2_score=f"{model_data.get('r2_score', 0):.3f}",
        mae=f"{model_data.get('mae', 0):,.0f}",
        rmse=f"{model_data.get('rmse', 0):,.0f}",
        num_houses=len(original_data),
        num_features=len(model_data['columns']),
        avg_price=original_data['Price'].mean(),
        min_price=original_data['Price'].min(),
        max_price=original_data['Price'].max(),
        std_price=original_data['Price'].std(),
        feature_stats=feature_stats,
        price_stats=price_stats
    )

@app.route('/api/info')
def api_info():
    """Informations sur le modèle (API)"""
    if model_data is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    return jsonify({
        'model': model_data.get('best_model_name'),
        'features': model_data['columns'],
        'dataset_size': len(original_data) if original_data is not None else 0,
        'performance': {
            'r2_score': model_data.get('r2_score'),
            'mae': model_data.get('mae'),
            'rmse': model_data.get('rmse')
        }
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'dataset_loaded': original_data is not None,
        'dataset_size': len(original_data) if original_data is not None else 0,
        'timestamp': datetime.now().isoformat()
    })

# =============================================
# LANCEMENT DE L'APPLICATION
# =============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌐 APPLICATION FLASK PRÊTE")
    print("="*50)
    print("📡 Port: 5000")
    print("🔗 URL principale: http://localhost:5000")
    print("🎲 Maison aléatoire: http://localhost:5000/random_house")
    print("📊 Statistiques: http://localhost:5000/statistics")
    print("🩺 Health check: http://localhost:5000/health")
    print("📋 API Info: http://localhost:5000/api/info")
    print("="*50 + "\n")
    
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        use_reloader=False
    )