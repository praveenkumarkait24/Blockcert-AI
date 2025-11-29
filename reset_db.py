from app import app
from db import db

with app.app_context():
    # This deletes all existing tables and data
    db.drop_all()

    # This creates fresh, empty tables
    db.create_all()

    print("✅ Database has been completely reset!")