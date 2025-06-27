# main.py
from app import create_app  # ✅ From app/__init__.py

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
