# hive_memory_adapter.py
# Verbindung von The Hive zu Pinecone Vektor-Gedächtnis (aktualisiert für Pinecone SDK v3)

"""Hive Memory Adapter.

This module links The Hive with Pinecone vector memory and OpenAI. It provides
helpers for creating and querying a Pinecone index, persisting an optional
Assistant and exposing a minimal HTTP API for remote interaction.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import yaml
from flask import Flask, jsonify, request

# 1. Umgebungsvariablen laden
def load_env(env_path: str = ".envALL.txt") -> None:
    """Load environment variables from the given file."""

    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(".env geladen.")
    else:
        raise FileNotFoundError("Keine .envALL.txt gefunden")

# 2. Pinecone-Verbindung herstellen (neue Syntax)
def connect_pinecone() -> Pinecone:
    """Return an authenticated Pinecone client."""

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY fehlt")
    environment = os.getenv("PINECONE_ENV")
    pc = Pinecone(api_key=api_key)
    print("Pinecone verbunden mit:", environment)
    return pc

# 3. Index vorbereiten
def get_or_create_index(
    pc: Pinecone,
    index_name: str = "hive-core",
    dimension: int = 1536,
    region: Optional[str] = None,
) -> object:
    """Return a usable index, creating it if necessary.

    If the index exists but is broken, it will be deleted and re-created.
    """

    existing = [i.name for i in pc.list_indexes()]
    region = region or os.getenv("PINECONE_REGION", "us-west1")
    if index_name not in existing:
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region=region),
            )
            print("→ Neuer Index erstellt:", index_name)
        except Exception as e:
            if region != "us-west1":
                print("Region fehlerhaft, Fallback auf us-west1:", e)
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="gcp", region="us-west1"),
                )
                print("→ Index mit Fallback erstellt:", index_name)
            else:
                raise
    else:
        print("→ Index gefunden:", index_name)
        try:
            pc.describe_index(index_name)
        except Exception as e:
            print("Index fehlerhaft, erstelle neu:", e)
            pc.delete_index(index_name)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region=region),
            )
            print("→ Index neu erstellt:", index_name)
    return pc.Index(index_name)

# 4. Embedding erzeugen (via OpenAI)
def embed_text(text: str, model: Optional[str] = None) -> list:
    """Create an embedding for text using OpenAI."""

    model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY fehlt")
    client.api_key = api_key
    return client.embeddings.create(model=model, input=text).data[0].embedding

# 5. Abfrage an Pinecone senden
def query_similar(index, text, top_k=3):
    vector = embed_text(text)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results

# 6. Pinecone Assistant erstellen oder laden
def create_hive_assistant(pc: Pinecone, store_path: str = "assistant_id.txt") -> str:
    """Create a Pinecone Assistant if none exists and persist its id."""

    if os.path.exists(store_path):
        with open(store_path, "r", encoding="utf-8") as f:
            assistant_id = f.read().strip()
        if assistant_id:
            print("→ Bestehenden Assistant verwendet:", assistant_id)
            return assistant_id

    assistant = pc.assistant.create_assistant(
        assistant_name="The Hive",
        instructions=(
            "Handle inputs with layered systemic awareness. Reflect semantic depth."
            " Use structured clarity and observe emergent meaning."
        ),
        timeout=30,
    )
    assistant_id = assistant["assistant_id"]
    with open(store_path, "w", encoding="utf-8") as f:
        f.write(assistant_id)
    print("→ Pinecone Assistant erstellt:", assistant_id)
    return assistant_id

# 6. Embedding speichern oder aktualisieren
def upsert_text(index, text: str, metadata: Optional[dict] = None, id: Optional[str] = None) -> str:
    """Store or update an embedding in the index and return its id."""

    vector = embed_text(text)
    if id is None:
        id = str(hash(text))
    index.upsert([
        {
            "id": id,
            "values": vector,
            "metadata": metadata or {},
        }
    ])
    return id


def load_memory_yaml(path):
    """Read memory records from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def system_status(index_name: str = "hive-core") -> None:
    """Print key configuration parameters."""

    print("🧭 Status: Modellkontext und Systemeinstellungen")
    print("🔢 Embedding-Modell:", os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    print("🗂️ Indexname:", index_name)
    print("📡 Pinecone-Region:", os.getenv("PINECONE_REGION", "us-west1"))


def create_app(pc: Pinecone, index_name: str = "hive-core") -> Flask:
    """Return a simple Flask app to access the adapter remotely."""

    app = Flask(__name__)
    index = get_or_create_index(pc, index_name=index_name)

    @app.route("/status")
    def status_route():
        return jsonify({"index": index_name, "region": os.getenv("PINECONE_REGION", "us-west1")})

    @app.route("/embed", methods=["POST"])
    def embed_route():
        text = request.json.get("text", "")
        return jsonify({"embedding": embed_text(text)})

    @app.route("/query", methods=["POST"])
    def query_route():
        text = request.json.get("text", "")
        results = query_similar(index, text).to_dict()
        return jsonify(results)

    return app

# Einstiegspunkt
if __name__ == "__main__":
    load_env()
    pc = connect_pinecone()
    index = get_or_create_index(pc)
    create_hive_assistant(pc)
    system_status()

    if os.getenv("HIVE_SERVER") == "1":
        app = create_app(pc)
        app.run(debug=True)
    else:
        beispieltext = "Systemische Emergenz erzeugt neue Formen."
        print("→ Embedding & Abfrage zu:", beispieltext)
        ergebnis = query_similar(index, beispieltext)
        for match in ergebnis.matches:
            print(f"{match.score:.3f} → {match.metadata}")
