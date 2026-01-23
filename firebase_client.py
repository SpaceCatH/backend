import firebase_admin
from firebase_admin import credentials, firestore

_app = None
_db = None

def get_db():
    global _app, _db

    if _app is None:
        # Cloud Run uses ADC automatically, so no key file needed
        _app = firebase_admin.initialize_app()

    if _db is None:
        # Explicitly point to the Firestore project that actually contains your data
        _db = firestore.client(project="detective-app-67ffd")

    return _db