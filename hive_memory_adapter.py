# hive_memory_adapter.py
# Verbindung von The Hive zu Pinecone Vektor-Gedächtnis (aktualisiert für Pinecone SDK v3)

import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import yaml

# 1. Umgebungsvariablen laden
def load_env(env_path=".envALL.txt"):
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(".env geladen.")
    else:
        raise FileNotFoundError("Keine .envALL.txt gefunden")

# 2. Pinecone-Verbindung herstellen (neue Syntax)
def connect_pinecone():
    """Return a Pinecone client using keys from the environment."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError("PINECONE_API_KEY nicht gesetzt")

    environment = os.getenv("PINECONE_ENV", "us-west1-gcp")

    pc = Pinecone(api_key=api_key)
    print("Pinecone verbunden mit:", environment)
    return pc

# 3. Index vorbereiten
def get_or_create_index(pc, index_name="hive-core", dimension=1536, region=None, drop_old=False):
    """Return a Pinecone index, creating it when necessary.

    If ``drop_old`` is ``True`` an existing index with the same name will be
    removed before a new one is created.
    """

    existing = [i.name for i in pc.list_indexes()]
    region = region or os.getenv("PINECONE_REGION", "us-west1")

    if index_name in existing and drop_old:
        pc.delete_index(index_name)
        existing.remove(index_name)
        print("→ Alter Index gelöscht:", index_name)

    if index_name not in existing:
        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region=region)
            )
            print("→ Neuer Index erstellt:", index_name)
        except Exception as e:
            if region != "us-west1":
                print("Region fehlerhaft, Fallback auf us-west1:", e)
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="gcp", region="us-west1")
                )
                print("→ Index mit Fallback erstellt:", index_name)
            else:
                raise
    else:
        print("→ Index gefunden:", index_name)

    return pc.Index(index_name)

# 4. Embedding erzeugen (via OpenAI)
def embed_text(text, model="text-embedding-3-small"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY nicht gesetzt")

    client = OpenAI(api_key=api_key)
    return client.embeddings.create(model=model, input=text).data[0].embedding

# 5. Abfrage an Pinecone senden
def query_similar(index, text, top_k=3):
    vector = embed_text(text)
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return results

# 6. Embedding speichern oder aktualisieren
def upsert_text(index, text, metadata=None, id=None):
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

# Einstiegspunkt
if __name__ == "__main__":
    load_env()
    pc = connect_pinecone()
    index = get_or_create_index(pc)

    beispieltext = "Systemische Emergenz erzeugt neue Formen."
    print("→ Embedding & Abfrage zu:", beispieltext)
    ergebnis = query_similar(index, beispieltext)
    for match in ergebnis.matches:
        print(f"{match.score:.3f} → {match.metadata}")
