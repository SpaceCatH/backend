import firebase_admin
from firebase_admin import credentials, firestore

_app = None
_db = None

def get_db():
    global _app, _db

    if _app is None:
        # Force Firebase Admin SDK to use the Firestore project that actually contains your data
        _app = firebase_admin.initialize_app(options={
            "projectId": "detective-app-67ffd"
        })

    if _db is None:
        _db = firestore.client()   # No project arg here — project is set in initialize_app()

    return _db