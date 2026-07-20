"""
utils/db.py
-----------
MongoDB Database Manager for IntelliMoE chat history persistence.

Provides connection handling, server pinging, upsert, query, and deletion operations
with graceful exception handling and low connection timeouts.
"""

import logging
from datetime import datetime
from typing import Any, Optional

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    PYMONGO_INSTALLED = True
except ImportError:
    pymongo = None
    MongoClient = None
    PYMONGO_INSTALLED = False

import os

try:
    from config.settings import MONGODB_URI, MONGODB_DB_NAME, MONGODB_COLLECTION
except (ImportError, AttributeError):
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://annamneedisuresh003_db_user:Suresh123@cluster0.3q6zmri.mongodb.net/?appName=Cluster0")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "intellimoe_db")
    MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "chat_histories")

logger = logging.getLogger(__name__)

# Global cached MongoClient instance
_mongo_client: Optional[Any] = None


def get_mongo_client(timeout_ms: int = 2000) -> Optional[Any]:
    """
    Get or initialize a cached PyMongo MongoClient.

    Parameters
    ----------
    timeout_ms : int
        Server selection timeout in milliseconds to prevent UI blocking.

    Returns
    -------
    MongoClient or None
    """
    global _mongo_client

    if not PYMONGO_INSTALLED:
        logger.debug("PyMongo package is not installed.")
        return None

    if not MONGODB_URI:
        logger.debug("MONGODB_URI is not set in environment or settings.")
        return None

    if _mongo_client is not None:
        return _mongo_client

    try:
        _mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
        )
        # Test connection with a fast admin ping
        _mongo_client.admin.command("ping")
        logger.info("MongoDB client connected successfully to database '%s'.", MONGODB_DB_NAME)
        return _mongo_client
    except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
        logger.warning("MongoDB connection check failed (%s). Falling back to local storage.", e)
        _mongo_client = None
        return None


def is_mongodb_available() -> bool:
    """Check if MongoDB server is active and reachable."""
    client = get_mongo_client(timeout_ms=1500)
    return client is not None


def save_chats_to_mongodb(email: str, serialized_chats: dict) -> bool:
    """
    Save or update user conversation threads in MongoDB collection.

    Parameters
    ----------
    email : str
        User identifier/email.
    serialized_chats : dict
        Dictionary of serialized chat states.

    Returns
    -------
    bool
        True if successfully saved, False otherwise.
    """
    client = get_mongo_client()
    if not client:
        return False

    try:
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]

        safe_email = "".join([c if c.isalnum() else "_" for c in email.lower().strip()])
        doc = {
            "_id": safe_email,
            "email": email.lower().strip(),
            "chats": serialized_chats,
            "updated_at": datetime.now(),
        }

        collection.replace_one({"_id": safe_email}, doc, upsert=True)
        logger.info("Chat history saved to MongoDB for user '%s' (%d threads).", email, len(serialized_chats))
        return True
    except Exception as e:
        logger.error("Failed to save chat history to MongoDB for user '%s': %s", email, e)
        return False


def load_chats_from_mongodb(email: str) -> Optional[dict]:
    """
    Fetch user conversation threads from MongoDB.

    Parameters
    ----------
    email : str
        User identifier/email.

    Returns
    -------
    dict | None
        Serialized chat dictionary if found, None if unconfigured or error.
    """
    client = get_mongo_client()
    if not client:
        return None

    try:
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]

        safe_email = "".join([c if c.isalnum() else "_" for c in email.lower().strip()])
        doc = collection.find_one({"_id": safe_email})

        if doc and "chats" in doc:
            logger.info("Loaded chat history from MongoDB for user '%s' (%d threads).", email, len(doc["chats"]))
            return doc["chats"]
        
        logger.debug("No MongoDB document found for user '%s'.", email)
        return None
    except Exception as e:
        logger.error("Failed to load chat history from MongoDB for user '%s': %s", email, e)
        return None


def delete_chat_from_mongodb(email: str, chat_id: str) -> bool:
    """
    Remove a single conversation thread from MongoDB for user.

    Parameters
    ----------
    email : str
        User identifier/email.
    chat_id : str
        ID of thread to remove.

    Returns
    -------
    bool
        True if deleted/updated in MongoDB, False otherwise.
    """
    client = get_mongo_client()
    if not client:
        return False

    try:
        db = client[MONGODB_DB_NAME]
        collection = db[MONGODB_COLLECTION]

        safe_email = "".join([c if c.isalnum() else "_" for c in email.lower().strip()])
        collection.update_one(
            {"_id": safe_email},
            {"$unset": {f"chats.{chat_id}": ""}, "$set": {"updated_at": datetime.now()}},
        )
        logger.info("Deleted chat '%s' from MongoDB for user '%s'.", chat_id, email)
        return True
    except Exception as e:
        logger.error("Failed to delete chat '%s' from MongoDB for user '%s': %s", chat_id, email, e)
        return False
