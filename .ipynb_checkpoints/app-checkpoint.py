from flask import Flask, render_template_string
import pickle
import os

app = Flask(__name__)

# Vérifier si model.pkl existe
print("🔍 Recherche de model.pkl...")
print("Chemin actuel:", os.getcwd())
print("Fichier model.pkl existe:", os.path.exists("model.pkl"))

if os.path.exists("model.pkl"):
    print("✅ model.pkl trouvé!")
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    print("📦 Modèle chargé:", type(model_data['model']).__name__)
else:
    print("❌ model.pkl non trouvé!")

@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Flask</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 50px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
                max-width: 600px;
                margin: 0 auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Flask est en marche !</h1>
            <p>Si vous voyez ce message, Flask fonctionne correctement.</p>
            <p>Port: 5000</p>
            <p>Modèle chargé: """ + ("✅" if os.path.exists("model.pkl") else "❌") + """</p>
            <a href="/predict" style="color: #ffcc00; font-weight: bold;">Tester la prédiction</a>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/predict')
def test_predict():
    return "Fonction de prédiction active!"

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Lancement de Flask...")
    print("🌐 Accédez à: http://localhost:5000")
    print("📂 Répertoire:", os.getcwd())
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)