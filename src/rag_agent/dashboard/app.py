"""Streamlit admin dashboard for rag-agent. Run: make dashboard"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import streamlit as st

st.set_page_config(page_title="rag-agent admin", page_icon="🤖", layout="wide")

API_BASE = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
API_KEY = st.sidebar.text_input("API Key", value="dev-key", type="password")
HEADERS = {"X-API-Key": API_KEY}

PAGE = st.sidebar.radio(
    "Navigation",
    ["💬 Chat", "📥 Ingestion", "📊 Métriques", "🔑 API Keys", "🧪 Évaluation"],
)

st.sidebar.divider()
st.sidebar.caption("rag-agent v0.1.0")


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path: str) -> dict[str, object] | None:
    try:
        r = httpx.get(f"{API_BASE}{path}", headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, json_data: dict[str, object]) -> dict[str, object] | None:
    try:
        r = httpx.post(f"{API_BASE}{path}", json=json_data, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Page: Chat ────────────────────────────────────────────────────────────────

if PAGE == "💬 Chat":
    st.title("💬 Chat — Test du pipeline RAG")

    model = st.selectbox(
        "Modèle",
        [
            "(default)",
            "google/gemini-flash-1.5",
            "anthropic/claude-3.5-sonnet",
            "mistralai/mistral-large",
            "openai/gpt-4o-mini",
        ],
    )
    use_agent = st.toggle("Utiliser LangGraph Agent (avec boucle de révision)", value=False)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if query := st.chat_input("Posez votre question…"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours…"):
                payload: dict[str, object] = {"query": query}
                if model != "(default)":
                    payload["model"] = model

                endpoint = "/api/v1/agent" if use_agent else "/api/v1/chat"
                result = api_post(endpoint, payload)

            if result:
                answer = str(result.get("answer", ""))
                st.write(answer)

                col1, col2 = st.columns(2)
                with col1:
                    if result.get("cached"):
                        st.success("⚡ Réponse depuis le cache sémantique")
                    if result.get("hallucination_score") is not None:
                        score = float(result["hallucination_score"])
                        color = "green" if score >= 0.75 else "orange" if score >= 0.5 else "red"
                        st.markdown(f"**Confiance** : :{color}[{score:.2f}]")

                with col2:
                    usage = result.get("usage", {})
                    if usage:
                        st.caption(f"Tokens: {usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion")

                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**[{i+1}]** `{src.get('source', '?')}` — score: `{src.get('score', 0):.3f}`")
                            st.caption(str(src.get("text", ""))[:300])

                st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.button("🗑 Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()


# ── Page: Ingestion ───────────────────────────────────────────────────────────

elif PAGE == "📥 Ingestion":
    st.title("📥 Ingestion de documents")

    tab_file, tab_text = st.tabs(["📄 Fichier", "📝 Texte brut"])

    with tab_file:
        uploaded = st.file_uploader(
            "Choisir un fichier",
            type=["pdf", "docx", "txt", "html"],
            accept_multiple_files=True,
        )
        if uploaded and st.button("Ingérer les fichiers"):
            for f in uploaded:
                try:
                    r = httpx.post(
                        f"{API_BASE}/api/v1/ingest/file",
                        files={"file": (f.name, f.read(), f.type)},
                        headers=HEADERS,
                        timeout=60,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.success(f"✅ {f.name} → job `{data['job_id']}`")
                except Exception as e:
                    st.error(f"❌ {f.name}: {e}")

    with tab_text:
        source = st.text_input("Identifiant source", placeholder="ex: doc-interne-2024")
        text = st.text_area("Texte à ingérer", height=200)
        if st.button("Ingérer le texte") and text and source:
            result = api_post("/api/v1/ingest/text", {"text": text, "source": source})
            if result:
                st.success(f"✅ Job `{result['job_id']}` — statut : {result['status']}")

    st.divider()
    st.subheader("Vérifier un job")
    job_id = st.text_input("Job ID")
    if st.button("Vérifier") and job_id:
        job = api_get(f"/api/v1/jobs/{job_id}")
        if job:
            status_color = {"SUCCESS": "green", "FAILURE": "red", "PENDING": "gray", "STARTED": "blue"}
            s = str(job.get("status", ""))
            st.markdown(f"**Statut** : :{status_color.get(s, 'gray')}[{s}]")
            if job.get("result"):
                st.json(job["result"])
            if job.get("error"):
                st.error(job["error"])


# ── Page: Métriques ───────────────────────────────────────────────────────────

elif PAGE == "📊 Métriques":
    st.title("📊 Métriques en temps réel")
    st.info("Métriques Prometheus détaillées disponibles sur Grafana → http://localhost:3001")

    col1, col2, col3 = st.columns(3)

    with col1:
        health = api_get("/health")
        if health:
            st.metric("Statut", "🟢 OK")
            st.metric("Version", health.get("version", "?"))

    if st.button("🔄 Rafraîchir"):
        st.rerun()

    # Parse /metrics endpoint for key values
    try:
        r = httpx.get(f"{API_BASE}/metrics", timeout=5)
        metrics_text = r.text

        def extract_metric(name: str) -> str:
            for line in metrics_text.splitlines():
                if line.startswith(name + " ") or line.startswith(name + "{"):
                    parts = line.rsplit(" ", 1)
                    if len(parts) == 2:
                        try:
                            return f"{float(parts[1]):.2f}"
                        except ValueError:
                            pass
            return "N/A"

        with col2:
            st.metric("Cache hits", extract_metric("semantic_cache_hits_total"))
            st.metric("Cache misses", extract_metric("semantic_cache_misses_total"))

        with col3:
            st.metric("Guardrail blocks", extract_metric("guardrail_blocked_total"))
            st.metric("RAG queries", extract_metric("rag_queries_total"))

    except Exception:
        st.warning("Impossible de récupérer les métriques Prometheus")


# ── Page: API Keys ────────────────────────────────────────────────────────────

elif PAGE == "🔑 API Keys":
    st.title("🔑 Gestion des API Keys")
    st.info("En développement — intégration avec PostgreSQL à implémenter dans Phase 0c.")

    st.subheader("Créer une clé")
    owner = st.text_input("Propriétaire")
    tier = st.selectbox("Tier", ["free", "pro", "admin"])
    if st.button("Générer") and owner:
        import hashlib
        import secrets
        raw = secrets.token_urlsafe(32)
        st.success(f"Clé générée (sauvegardez-la, elle ne sera plus affichée) :")
        st.code(raw)
        st.caption(f"Hash SHA-256 à stocker en base : `{hashlib.sha256(raw.encode()).hexdigest()}`")


# ── Page: Évaluation ─────────────────────────────────────────────────────────

elif PAGE == "🧪 Évaluation":
    st.title("🧪 Évaluation RAG (Ragas)")

    report_files = list(Path("reports").glob("eval_*.json")) if Path("reports").exists() else []

    if report_files:
        st.subheader("Derniers rapports")
        latest = max(report_files, key=lambda f: f.stat().st_mtime)
        data = json.loads(latest.read_text())

        col1, col2, col3 = st.columns(3)
        col1.metric("Faithfulness", f"{data.get('faithfulness', 0):.3f}", delta_color="normal")
        col2.metric("Answer Relevancy", f"{data.get('answer_relevancy', 0):.3f}")
        col3.metric("Context Recall", f"{data.get('context_recall', 0):.3f}")

        threshold = 0.80
        if data.get("faithfulness", 0) >= threshold:
            st.success(f"✅ Quality gate passed (faithfulness ≥ {threshold})")
        else:
            st.error(f"❌ Quality gate FAILED (faithfulness < {threshold})")

        st.caption(f"Rapport : {latest.name} — {data.get('n_samples', '?')} samples")

    else:
        st.warning("Aucun rapport trouvé dans `reports/`. Lancez `make eval`.")

    if st.button("🚀 Lancer une évaluation"):
        st.info("Lancez `make eval` dans le terminal pour générer un rapport.")
