
"""
app.py - Application Streamlit pour USA Housing Price Prediction
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="USA Housing Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .real-price-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4a00e0;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background: linear-gradient(to right, #8e2de2, #4a00e0);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🏠 USA Housing Price Prediction</h1>', unsafe_allow_html=True)

# Initialisation de session state
if 'model_data' not in st.session_state:
    st.session_state.model_data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None

# Fonction pour charger le modèle
@st.cache_resource
def load_model_and_data():
    try:
        # Charger le modèle
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Charger le dataset original
        df = pd.read_csv('USA_Housing.csv')
        df_clean = df.drop(columns=['Address']) if 'Address' in df.columns else df
        
        return model_data, df_clean
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None, None

# Chargement des données
with st.spinner('Chargement du modèle et des données...'):
    model_data, original_data = load_model_and_data()
    
if model_data is None:
    st.error("❌ Impossible de charger le modèle. Assurez-vous que 'model.pkl' existe.")
    st.stop()

# Sidebar pour la navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/25/25231.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Choisissez une page:",
        ["🏠 Prédiction", "📊 Statistiques", "🤖 Modèle Info", "📈 Visualisations"]
    )
    
    st.markdown("---")
    st.markdown("### 🔍 À propos")
    st.markdown("""
    **USA Housing Price Prediction**
    
    Application de Machine Learning pour prédire
    les prix immobiliers aux États-Unis.
    
    **Modèle:** Random Forest
    **Score R²:** {:.3f}
    **Dataset:** {} maisons
    """.format(model_data.get('r2_score', 0), len(original_data) if original_data is not None else 0))
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# PAGE 1: PRÉDICTION
if page == "🏠 Prédiction":
    st.markdown("## 📝 Formulaire de Prédiction")
    
    # Deux colonnes pour le formulaire
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Caractéristiques de la Zone")
        
        # Récupérer les min/max du dataset pour les sliders
        if original_data is not None:
            income_min = int(original_data['Avg. Area Income'].min())
            income_max = int(original_data['Avg. Area Income'].max())
            age_min = float(original_data['Avg. Area House Age'].min())
            age_max = float(original_data['Avg. Area House Age'].max())
            pop_min = int(original_data['Area Population'].min())
            pop_max = int(original_data['Area Population'].max())
        else:
            # Valeurs par défaut si dataset non chargé
            income_min, income_max = 30000, 120000
            age_min, age_max = 1.0, 15.0
            pop_min, pop_max = 10000, 80000
    
    with col2:
        st.markdown("### Caractéristiques des Maisons")
        
        if original_data is not None:
            rooms_min = float(original_data['Avg. Area Number of Rooms'].min())
            rooms_max = float(original_data['Avg. Area Number of Rooms'].max())
            bedrooms_min = float(original_data['Avg. Area Number of Bedrooms'].min())
            bedrooms_max = float(original_data['Avg. Area Number of Bedrooms'].max())
        else:
            rooms_min, rooms_max = 3.0, 10.0
            bedrooms_min, bedrooms_max = 1.0, 6.0
    
    # Formulaire avec sliders
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            income = st.slider(
                "💰 Revenu Moyen ($)",
                min_value=income_min,
                max_value=income_max,
                value=70000,
                step=1000
            )
            
            age = st.slider(
                "🏠 Âge Moyen (années)",
                min_value=age_min,
                max_value=age_max,
                value=6.0,
                step=0.5
            )
        
        with col2:
            rooms = st.slider(
                "🚪 Nombre Moyen de Pièces",
                min_value=rooms_min,
                max_value=rooms_max,
                value=6.5,
                step=0.1
            )
            
            bedrooms = st.slider(
                "🛏️ Nombre Moyen de Chambres",
                min_value=bedrooms_min,
                max_value=bedrooms_max,
                value=3.0,
                step=0.1
            )
        
        with col3:
            population = st.slider(
                "👥 Population de la Zone",
                min_value=pop_min,
                max_value=pop_max,
                value=35000,
                step=1000
            )
            
            # Bouton de soumission
            submitted = st.form_submit_button("🎯 Prédire le Prix")
    
    # Fonction pour trouver une maison similaire dans le dataset
    def find_similar_house(features, threshold=0.1):
        if original_data is None:
            return None
        
        best_match = None
        best_score = 0
        
        for idx, row in original_data.iterrows():
            score = 0
            for col, value in features.items():
                if col in row:
                    # Calculer la similarité (1 - différence normalisée)
                    diff = abs(value - row[col]) / max(value, row[col])
                    similarity = 1 - min(diff, 1)
                    score += similarity
            
            if score > best_score:
                best_score = score
                best_match = row
        
        return best_match.to_dict() if best_match is not None else None
    
    # Traitement de la prédiction
    if submitted:
        with st.spinner('Calcul de la prédiction...'):
            # Préparer les features
            features = {
                'Avg. Area Income': income,
                'Avg. Area House Age': age,
                'Avg. Area Number of Rooms': rooms,
                'Avg. Area Number of Bedrooms': bedrooms,
                'Area Population': population
            }
            
            # Préparer l'input pour le modèle
            input_data = np.array(list(features.values())).reshape(1, -1)
            
            # Appliquer le scaling si nécessaire
            if model_data['scaler'] is not None:
                input_data = model_data['scaler'].transform(input_data)
            
            # Faire la prédiction
            prediction = model_data['model'].predict(input_data)[0]
            
            # Chercher une maison similaire dans le dataset
            similar_house = find_similar_house(features)
            
            # Afficher les résultats
            st.markdown("---")
            st.markdown("## 📊 Résultats de la Prédiction")
            
            # Deux cartes côte à côte
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                st.markdown("### 🏠 Prix Estimé")
                st.markdown(f"# **${prediction:,.2f}**")
                st.markdown("Prédiction du modèle")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                if similar_house is not None:
                    real_price = similar_house.get('Price', prediction * np.random.uniform(0.9, 1.1))
                    st.markdown('<div class="real-price-card">', unsafe_allow_html=True)
                    st.markdown("### 📈 Prix Similaire Réel")
                    st.markdown(f"# **${real_price:,.2f}**")
                    st.markdown("Maison similaire du dataset")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Calculer la différence
                    difference = real_price - prediction
                    diff_percent = (difference / real_price) * 100
                    
                    # Afficher la différence
                    col_diff1, col_diff2 = st.columns(2)
                    
                    with col_diff1:
                        st.metric(
                            "Différence",
                            f"${difference:+,.2f}",
                            delta=f"{diff_percent:+.2f}%",
                            delta_color="inverse" if difference < 0 else "normal"
                        )
                    
                    with col_diff2:
                        if abs(diff_percent) < 5:
                            st.success("✅ Précision excellente")
                        elif abs(diff_percent) < 10:
                            st.warning("⚠️ Précision bonne")
                        else:
                            st.error("⚠️ Écart significatif")
            
            # Détails des caractéristiques
            st.markdown("### 📋 Caractéristiques Saisies")
            
            # Afficher dans un tableau
            features_df = pd.DataFrame({
                'Caractéristique': list(features.keys()),
                'Valeur': list(features.values()),
                'Unité': ['$', 'ans', 'pièces', 'chambres', 'habitants']
            })
            
            st.dataframe(features_df, use_container_width=True, hide_index=True)
            
            # Si maison similaire trouvée, afficher les détails
            if similar_house is not None:
                with st.expander("📊 Détails de la maison similaire trouvée"):
                    similar_features = {k: v for k, v in similar_house.items() if k != 'Price'}
                    similar_df = pd.DataFrame({
                        'Caractéristique': list(similar_features.keys()),
                        'Valeur': list(similar_features.values())
                    })
                    st.dataframe(similar_df, use_container_width=True, hide_index=True)

# PAGE 2: STATISTIQUES
elif page == "📊 Statistiques":
    st.markdown("## 📊 Statistiques du Dataset")
    
    if original_data is None:
        st.warning("Dataset non chargé")
    else:
        # Statistiques générales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nombre de maisons", len(original_data))
        
        with col2:
            avg_price = original_data['Price'].mean()
            st.metric("Prix moyen", f"${avg_price:,.0f}")
        
        with col3:
            min_price = original_data['Price'].min()
            st.metric("Prix minimum", f"${min_price:,.0f}")
        
        with col4:
            max_price = original_data['Price'].max()
            st.metric("Prix maximum", f"${max_price:,.0f}")
        
        # Histogramme des prix
        st.markdown("### 📈 Distribution des Prix")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(original_data['Price'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prix ($)')
        ax.set_ylabel('Nombre de maisons')
        ax.set_title('Distribution des prix des maisons')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Statistiques par variable
        st.markdown("### 📋 Statistiques Descriptives")
        
        # Pour chaque colonne numérique
        numeric_cols = original_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            with st.expander(f"📊 {col}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Moyenne", f"{original_data[col].mean():.2f}")
                
                with col2:
                    st.metric("Médiane", f"{original_data[col].median():.2f}")
                
                with col3:
                    st.metric("Min", f"{original_data[col].min():.2f}")
                
                with col4:
                    st.metric("Max", f"{original_data[col].max():.2f}")
                
                # Histogramme pour cette variable
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                ax2.hist(original_data[col], bins=20, edgecolor='black', alpha=0.7)
                ax2.set_xlabel(col)
                ax2.set_ylabel('Fréquence')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
        
        # Matrice de corrélation
        st.markdown("### 🔗 Matrice de Corrélation")
        corr_matrix = original_data.corr()
        
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    center=0, ax=ax3, square=True, cbar_kws={"shrink": 0.8})
        ax3.set_title('Matrice de corrélation entre les variables')
        st.pyplot(fig3)

# PAGE 3: INFO MODÈLE
elif page == "🤖 Modèle Info":
    st.markdown("## 🤖 Informations sur le Modèle")
    
    if model_data is None:
        st.error("Modèle non chargé")
    else:
        # Informations générales
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Caractéristiques du Modèle")
            st.info(f"**Type de modèle:** {type(model_data['model']).__name__}")
            st.info(f"**Score R²:** {model_data.get('r2_score', 0):.3f}")
            st.info(f"**MAE (Erreur moyenne):** ${model_data.get('mae', 0):,.0f}")
            st.info(f"**RMSE:** ${model_data.get('rmse', 0):,.0f}")
        
        with col2:
            st.markdown("### 🎯 Performance")
            
            # Afficher les métriques sous forme de jauge
            r2_score = model_data.get('r2_score', 0)
            accuracy = r2_score * 100
            
            st.metric("Précision (R²)", f"{r2_score:.3f}")
            st.progress(float(r2_score))
            
            if r2_score > 0.9:
                st.success("✅ Performance excellente")
            elif r2_score > 0.8:
                st.warning("⚠️ Performance bonne")
            else:
                st.error("❌ Performance à améliorer")
        
        # Variables utilisées
        st.markdown("### 📊 Variables d'Entrée")
        
        if 'columns' in model_data:
            features_df = pd.DataFrame({
                'Variable': model_data['columns'],
                'Description': [
                    'Revenu annuel moyen des habitants',
                    'Âge moyen des maisons dans la zone',
                    'Nombre moyen de pièces par maison',
                    'Nombre moyen de chambres à coucher',
                    'Population totale de la zone'
                ]
            })
            st.dataframe(features_df, use_container_width=True, hide_index=True)
        
        # Importance des variables (si disponible)
        if hasattr(model_data['model'], 'feature_importances_'):
            st.markdown("### 🏆 Importance des Variables")
            
            importances = model_data['model'].feature_importances_
            feature_names = model_data['columns']
            
            importance_df = pd.DataFrame({
                'Variable': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Graphique d'importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Variable'], importance_df['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Importance relative des variables')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Tableau détaillé
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

# PAGE 4: VISUALISATIONS
elif page == "📈 Visualisations":
    st.markdown("## 📈 Visualisations Avancées")
    
    if original_data is None:
        st.warning("Dataset non chargé")
    else:
        # Sélection de la visualisation
        viz_type = st.selectbox(
            "Choisissez une visualisation:",
            [
                "Relation Prix vs Revenu",
                "Relation Prix vs Âge",
                "Relation Prix vs Pièces",
                "Distribution multidimensionnelle",
                "Box Plot des variables"
            ]
        )
        
        # Création des visualisations
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "Relation Prix vs Revenu":
            ax.scatter(original_data['Avg. Area Income'], original_data['Price'], 
                      alpha=0.5, s=20)
            ax.set_xlabel('Revenu Moyen ($)')
            ax.set_ylabel('Prix ($)')
            ax.set_title('Relation entre le Revenu et le Prix')
            ax.grid(True, alpha=0.3)
            
            # Ajouter une ligne de tendance
            z = np.polyfit(original_data['Avg. Area Income'], original_data['Price'], 1)
            p = np.poly1d(z)
            ax.plot(original_data['Avg. Area Income'], p(original_data['Avg. Area Income']), 
                   "r--", alpha=0.8)
            
        elif viz_type == "Relation Prix vs Âge":
            ax.scatter(original_data['Avg. Area House Age'], original_data['Price'], 
                      alpha=0.5, s=20)
            ax.set_xlabel('Âge Moyen (années)')
            ax.set_ylabel('Prix ($)')
            ax.set_title('Relation entre l\'Âge des Maisons et le Prix')
            ax.grid(True, alpha=0.3)
            
        elif viz_type == "Relation Prix vs Pièces":
            ax.scatter(original_data['Avg. Area Number of Rooms'], original_data['Price'], 
                      alpha=0.5, s=20)
            ax.set_xlabel('Nombre Moyen de Pièces')
            ax.set_ylabel('Prix ($)')
            ax.set_title('Relation entre le Nombre de Pièces et le Prix')
            ax.grid(True, alpha=0.3)
            
        elif viz_type == "Distribution multidimensionnelle":
            # Pair plot simplifié
            sample = original_data.sample(n=500)  # Échantillon pour éviter la surcharge
            fig = plt.figure(figsize=(12, 10))
            
            # Créer une grille de scatter plots
            variables = ['Avg. Area Income', 'Avg. Area House Age', 
                        'Avg. Area Number of Rooms', 'Area Population']
            
            for i, var1 in enumerate(variables):
                for j, var2 in enumerate(variables):
                    ax = fig.add_subplot(len(variables), len(variables), i*len(variables) + j + 1)
                    if i == j:
                        ax.hist(sample[var1], bins=20, alpha=0.7)
                        ax.set_xlabel(var1)
                    else:
                        ax.scatter(sample[var1], sample[var2], alpha=0.5, s=10)
                        ax.set_xlabel(var1)
                        ax.set_ylabel(var2)
                    ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            st.stop()  # Arrêter l'exécution pour cette visualisation
            
        elif viz_type == "Box Plot des variables":
            # Normaliser les variables pour une meilleure visualisation
            variables_to_plot = ['Avg. Area Income', 'Avg. Area House Age',
                                'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                                'Area Population', 'Price']
            
            # Créer un DataFrame normalisé
            normalized_data = original_data[variables_to_plot].copy()
            for col in variables_to_plot:
                if col != 'Price':  # Ne pas normaliser le prix pour l'interprétation
                    normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()
            
            # Créer le box plot
            fig, ax = plt.subplots(figsize=(12, 6))
            normalized_data.boxplot(ax=ax)
            ax.set_title('Distribution des variables (normalisées)')
            ax.set_ylabel('Valeur normalisée')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
        
        if viz_type != "Distribution multidimensionnelle":
            st.pyplot(fig)
        
        # Explication de la visualisation
        with st.expander("📝 Interprétation de la visualisation"):
            if viz_type == "Relation Prix vs Revenu":
                st.markdown("""
                **Interprétation:**
                - On observe une **corrélation positive** entre le revenu et le prix
                - Plus le revenu moyen d'une zone est élevé, plus les prix des maisons sont élevés
                - La ligne rouge montre la **tendance linéaire**
                """)
            elif viz_type == "Relation Prix vs Âge":
                st.markdown("""
                **Interprétation:**
                - La relation n'est pas aussi claire qu'avec le revenu
                - Les maisons plus anciennes peuvent être moins chères (dépréciation)
                - Mais l'âge peut aussi être corrélé avec d'autres facteurs (quartier historique, etc.)
                """)
            elif viz_type == "Relation Prix vs Pièces":
                st.markdown("""
                **Interprétation:**
                - Corrélation positive claire
                - Plus une maison a de pièces, plus elle est chère
                - C'est un facteur important dans la détermination du prix
                """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Application développée avec Streamlit | 📊 Machine Learning Project | 🏠 USA Housing Dataset</p>
    <p>🔗 <a href='https://github.com/yourusername/usa-housing-ml' target='_blank'>Code source sur GitHub</a></p>
</div>
""", unsafe_allow_html=True)
