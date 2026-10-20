"""Agent actus-mediterrane — collecte les actus maritimes corses et les publie sur CORSICAMEDACTU."""

import asyncio
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import httpx
from mistralai import Mistral, workflows
from pydantic import BaseModel, Field

QUERIES = [
    "Corsica Linea ferry Corse",
    "La Méridionale ferry Corse",
    "ports Corse Bastia Ajaccio trafic maritime",
    "météo maritime Corse traversée",
    "Corsica Ferries Corse",
]

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=fr&gl=FR&ceid=FR:fr"

SYSTEM_PROMPT = """Tu es un éditeur du site CORSICAMEDACTU, média dédié à l'actualité maritime en Corse.
On te donne des articles bruts. Sélectionne UNIQUEMENT ceux en rapport avec le maritime corse
(compagnies, ferries, ports, trafic, météo maritime, incidents en mer, réglementation maritime).
Pour chaque article retenu, rédige un résumé factuel en français de 2-3 phrases.
Réponds STRICTEMENT en JSON: {"actus": [{"titre": "...", "contenu": "...", "source_nom": "...", "source_url": "...", "categorie": "trafic|meteo|compagnie|societe", "published_at": "ISO8601"}]}
Si aucun article n'est pertinent, réponds {"actus": []}."""


class ActusInput(BaseModel):
    endpoint: str = Field(description="URL de l'API insert-actus du site")
    api_key: str = Field(description="Clé secrète x-api-key")
    max_items: int = 6


@workflows.activity()
async def fetch_feed(query: str) -> list[dict[str, Any]]:
    """Récupère un flux Google News RSS pour une requête."""
    full_url = GOOGLE_NEWS_RSS.format(query=query.replace(" ", "+"))
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(full_url)
            resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception:
        return []
    items: list[dict[str, Any]] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else "Presse"
        if not title or not link:
            continue
        try:
            published_at = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z").astimezone(timezone.utc).isoformat()
        except ValueError:
            published_at = datetime.now(timezone.utc).isoformat()
        items.append({"title": title, "link": link, "source": source, "published_at": published_at})
    return items


@workflows.activity()
async def curate(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filtre et résume les actus pertinentes avec un modèle Mistral."""
    if not raw_items:
        return []
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for it in raw_items:
        if it["link"] not in seen:
            seen.add(it["link"])
            unique.append(it)
    unique = unique[:25]
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    user_msg = "Articles:\n" + "\n".join(
        f"- {it['title']} | source: {it['source']} | {it['link']} | date: {it['published_at']}"
        for it in unique
    )
    resp = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return []
    actus = data.get("actus", [])
    for a in actus:
        a["categorie"] = a.get("categorie", "societe")
        if a["categorie"] not in ("trafic", "meteo", "compagnie", "societe"):
            a["categorie"] = "societe"
    return actus


@workflows.activity()
async def publish(payload: list[dict[str, Any]], endpoint: str, api_key: str) -> dict[str, Any]:
    """Envoie les actus vers le site CORSICAMEDACTU."""
    if not payload:
        return {"sent": 0, "response": "aucune actu à publier"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            endpoint,
            json=payload,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
        )
    return {"status": resp.status_code, "response": resp.text[:500]}


@workflows.workflow.define(
    name="actus-mediterrane",
    workflow_display_name="Actus Méditerranée",
    workflow_description="Collecte les actus maritimes corses et les publie sur CORSICAMEDACTU.",
)
class ActusWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: ActusInput) -> dict[str, Any]:
        feeds = await asyncio.gather(*(fetch_feed(q) for q in QUERIES))
        raw = [item for feed in feeds for item in feed]
        actus = await curate(raw)
        actus = actus[: input.max_items]
        result = await publish(actus, input.endpoint, input.api_key)
        return {"actus_publiees": len(actus), "resultat_envoi": result}
