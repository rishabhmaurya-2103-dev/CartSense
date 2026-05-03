from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("cart_abandonment_model.pkl")
brand_encoder = joblib.load("brand_encoder.pkl")
category_encoder = joblib.load("category_encoder.pkl")

def safe_encode(encoder, value):
    value = str(value)
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        price = float(data["price"])
        brand = safe_encode(brand_encoder, data["brand"])
        category = safe_encode(category_encoder, data["category"])

        total_events = int(data["total_events"])
        view_count = int(data["view_count"])
        cart_count = int(data["cart_count"])
        unique_products = int(data["unique_products"])
        avg_price = float(data["avg_price"])
        max_price = float(data["max_price"])
        min_price = float(data["min_price"])

        features = np.array([[
            price, brand, category, total_events, view_count,
            cart_count, unique_products, avg_price, max_price, min_price
        ]])

        probability = model.predict_proba(features)[0][1] * 100

        if probability < 40:
            risk = "Low Risk"
            suggestion = "User is likely to complete purchase. Normal checkout flow is enough."
        elif probability < 70:
            risk = "Medium Risk"
            suggestion = "Send reminder, small discount, or free delivery message."
        else:
            risk = "High Risk"
            suggestion = "Show urgent offer, cart reminder, or personalized recommendation."

        return jsonify({
            "risk": risk,
            "probability": round(probability, 2),
            "suggestion": suggestion,
            "analytics": {
                "views": view_count,
                "cart": cart_count,
                "products": unique_products,
                "price": price
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)