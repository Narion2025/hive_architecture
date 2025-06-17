# Hive Architecture

Dieses Repository demonstriert ein kleines Framework zur Orchestrierung von Custom GPTs mit einem Pinecone gestützten Gedächtnis. Hauptkomponenten sind der **Hive Memory Adapter** und ein einfacher **Model Selector**.

## Struktur

```
hive_architecture/
├── hive_memory_adapter.py        # Anbindung an Pinecone und OpenAI
├── Model_select/
│   └── model_selector.py         # Wahl des passenden GPT-Modells
└── README.md
```

## Setup

1. Python Umgebung erstellen

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai pinecone-client python-dotenv pyyaml
```

2. `.envALL.txt` anlegen und API-Schlüssel eintragen

```env
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENV=us-west1-gcp
```

## Rollen im System

- **Hive GPT** – ein Custom GPT, das über den Memory Adapter auf sein Gedächtnis zugreifen kann.
- **Assistant** – von OpenAI bereitgestellt; kann optional mit dem gleichen Speicher verknüpft werden.
- **Pinecone Memory** – persistenter Vektor-Speicher für Embeddings und Metadaten.

Durch die Verschmelzung von Custom GPT und Assistant entsteht ein gemeinsam nutzbares Wissensarchiv. Der Memory Adapter prüft nun, ob alle benötigten Umgebungsvariablen gesetzt sind und kann auf Wunsch einen bestehenden Index löschen und neu erstellen.

## Nutzung

Der Einstiegspunkt ist `hive_memory_adapter.py`. Dieser lädt die Umgebungsvariablen, verbindet sich mit Pinecone, legt bei Bedarf einen Index an und ermöglicht Embeddings sowie Abfragen. Der `ModelSelector` wählt anhand von Zielbeschreibung und Kontext das passende GPT-Modell aus.

---

(c) Narion 2025 – mit freundlicher Unterstützung von The Hive.
