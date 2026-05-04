"""
rag_system.py — מערכת RAG (Retrieval-Augmented Generation) מעולה
=================================================================
תומכת ב: PDF, DOCX, TXT, Markdown, URLs
מודל שפה: Claude (Anthropic)
מסד נתוני וקטורים: ChromaDB
Embeddings: sentence-transformers (מקומי, ללא עלות API)
"""

from __future__ import annotations

import os
import re
import hashlib
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

# ── Document Loaders ──────────────────────────────────────────────────────────
import pypdf
from docx import Document as DocxDocument
import requests
from bs4 import BeautifulSoup

# ── Embeddings & Vector DB ────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# ── LLM ──────────────────────────────────────────────────────────────────────
import anthropic

# ── Utilities ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """יחידת טקסט בודדת עם מטאדטה"""
    text: str
    source: str          # שם הקובץ / URL
    source_type: str     # pdf | docx | txt | url
    page: Optional[int] = None
    chunk_index: int = 0
    doc_id: str = ""     # hash ייחודי

    def __post_init__(self):
        if not self.doc_id:
            content = f"{self.source}::{self.chunk_index}::{self.text[:100]}"
            self.doc_id = hashlib.md5(content.encode()).hexdigest()


@dataclass
class SearchResult:
    """תוצאת חיפוש בודדת מ-ChromaDB"""
    text: str
    source: str
    source_type: str
    page: Optional[int]
    score: float         # ציון דמיון (0–1, גבוה יותר = רלוונטי יותר)


@dataclass
class RAGResponse:
    """תשובה מלאה מהמערכת"""
    question: str
    answer: str
    sources: list[SearchResult]
    tokens_used: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Document Loader
# ══════════════════════════════════════════════════════════════════════════════

class DocumentLoader:
    """טוען מסמכים ממקורות שונים ומחזיר רשימת Chunks גולמיים (לפני חלוקה)"""

    # ── PDF ───────────────────────────────────────────────────────────────────
    def load_pdf(self, path: str) -> list[Chunk]:
        chunks = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = self._clean(text)
                if text.strip():
                    chunks.append(Chunk(
                        text=text,
                        source=Path(path).name,
                        source_type="pdf",
                        page=page_num,
                        chunk_index=page_num - 1,
                    ))
        return chunks

    # ── DOCX ──────────────────────────────────────────────────────────────────
    def load_docx(self, path: str) -> list[Chunk]:
        doc = DocxDocument(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)
        full_text = self._clean(full_text)
        return [Chunk(
            text=full_text,
            source=Path(path).name,
            source_type="docx",
            chunk_index=0,
        )]

    # ── TXT / Markdown ────────────────────────────────────────────────────────
    def load_text(self, path: str) -> list[Chunk]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        text = self._clean(text)
        ext = Path(path).suffix.lower()
        stype = "markdown" if ext in (".md", ".markdown") else "txt"
        return [Chunk(
            text=text,
            source=Path(path).name,
            source_type=stype,
            chunk_index=0,
        )]

    # ── URL ───────────────────────────────────────────────────────────────────
    def load_url(self, url: str, timeout: int = 15) -> list[Chunk]:
        headers = {"User-Agent": "Mozilla/5.0 (RAG-Bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # הסר תגיות לא רלוונטיות
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = self._clean(text)
        return [Chunk(
            text=text,
            source=url,
            source_type="url",
            chunk_index=0,
        )]

    # ── Auto-detect ───────────────────────────────────────────────────────────
    def load(self, source: str) -> list[Chunk]:
        """טוען אוטומטית לפי סוג המקור"""
        if source.startswith("http://") or source.startswith("https://"):
            return self.load_url(source)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"הקובץ לא נמצא: {source}")
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.load_pdf(source)
        elif ext in (".docx", ".doc"):
            return self.load_docx(source)
        elif ext in (".txt", ".md", ".markdown", ".rst"):
            return self.load_text(source)
        else:
            raise ValueError(f"סוג קובץ לא נתמך: {ext}")

    @staticmethod
    def _clean(text: str) -> str:
        """ניקוי בסיסי של טקסט"""
        text = re.sub(r"\n{3,}", "\n\n", text)      # הסר שורות ריקות עודפות
        text = re.sub(r"[ \t]{2,}", " ", text)        # הסר רווחים עודפים
        text = text.strip()
        return text


# ══════════════════════════════════════════════════════════════════════════════
# Text Chunker
# ══════════════════════════════════════════════════════════════════════════════

class TextChunker:
    """
    מחלק טקסט ארוך לחתיכות (chunks) עם חפיפה (overlap).
    משתמש בגישה של פיצול לפי משפטים / פסקאות לחתיכות נקיות.
    """

    def __init__(self, chunk_size: int = 600, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, chunk: Chunk) -> list[Chunk]:
        """מפצל Chunk בודד לחתיכות קטנות יותר"""
        text = chunk.text
        if len(text) <= self.chunk_size:
            return [chunk]  # קצר מספיק — החזר כמות שהוא

        # פיצול לפי פסקאות תחילה, אחר כך לפי משפטים
        paragraphs = text.split("\n\n")
        sentences: list[str] = []
        for para in paragraphs:
            para = para.strip()
            if para:
                sents = re.split(r"(?<=[.!?])\s+", para)
                sentences.extend(sents)
                sentences.append("")  # מפריד בין פסקאות

        chunks: list[Chunk] = []
        current: list[str] = []
        current_len = 0
        idx = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                # שמור chunk נוכחי
                chunk_text = " ".join(s for s in current if s).strip()
                if chunk_text:
                    chunks.append(Chunk(
                        text=chunk_text,
                        source=chunk.source,
                        source_type=chunk.source_type,
                        page=chunk.page,
                        chunk_index=idx,
                    ))
                    idx += 1

                # Overlap: שמור את המשפטים האחרונים
                overlap_text = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.overlap:
                        overlap_text.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current = overlap_text
                current_len = overlap_len

            current.append(sent)
            current_len += sent_len

        # שמור שאריות
        if current:
            chunk_text = " ".join(s for s in current if s).strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    source=chunk.source,
                    source_type=chunk.source_type,
                    page=chunk.page,
                    chunk_index=idx,
                ))

        return chunks if chunks else [chunk]

    def split_all(self, raw_chunks: list[Chunk]) -> list[Chunk]:
        result = []
        for c in raw_chunks:
            result.extend(self.split(c))
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Vector Store (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    עוטף את ChromaDB עם sentence-transformers לצורך embeddings מקומיים.
    """
    EMBED_MODEL = "all-MiniLM-L6-v2"   # מהיר, קל, תוצאות מצוינות

    def __init__(self, persist_dir: str = "./rag_db", collection: str = "documents"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # טען מודל embeddings
        print("⏳ טוען מודל embeddings...")
        self.embedder = SentenceTransformer(self.EMBED_MODEL)

        # ChromaDB עם שמירה לדיסק
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.col = self.client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Vector store מוכן ({self.col.count()} מסמכים שמורים)")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        """מוסיף chunks למסד הנתונים. מחזיר כמות שנוספו."""
        added = 0
        for i in tqdm(range(0, len(chunks), batch_size), desc="מוסיף למסד הנתונים"):
            batch = chunks[i:i + batch_size]
            ids = [c.doc_id for c in batch]

            # בדוק אם כבר קיים
            existing = set(self.col.get(ids=ids)["ids"])
            new_batch = [c for c in batch if c.doc_id not in existing]
            if not new_batch:
                continue

            texts = [c.text for c in new_batch]
            embeddings = self._embed(texts)
            metadatas = [
                {
                    "source": c.source,
                    "source_type": c.source_type,
                    "page": c.page if c.page is not None else -1,
                    "chunk_index": c.chunk_index,
                }
                for c in new_batch
            ]

            self.col.add(
                ids=[c.doc_id for c in new_batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(new_batch)

        return added

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """חיפוש סמנטי — מחזיר k תוצאות הכי רלוונטיות"""
        if self.col.count() == 0:
            return []

        query_embedding = self._embed([query])[0]
        results = self.col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.col.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            output.append(SearchResult(
                text=doc,
                source=meta["source"],
                source_type=meta["source_type"],
                page=meta["page"] if meta["page"] != -1 else None,
                score=score,
            ))

        return output

    def list_sources(self) -> list[str]:
        """רשימת כל המקורות השמורים"""
        if self.col.count() == 0:
            return []
        all_meta = self.col.get(include=["metadatas"])["metadatas"]
        return sorted(set(m["source"] for m in all_meta))

    def clear(self):
        """מוחק את כל המסמכים"""
        self.client.delete_collection(self.col.name)
        self.col = self.client.get_or_create_collection(
            name=self.col.name,
            metadata={"hnsw:space": "cosine"}
        )


# ══════════════════════════════════════════════════════════════════════════════
# Claude Generator
# ══════════════════════════════════════════════════════════════════════════════

class ClaudeGenerator:
    """
    מייצר תשובות באמצעות Claude, תוך שימוש בהקשר שנשלף מה-RAG.
    """
    MODEL = "claude-sonnet-4-6"

    SYSTEM_PROMPT = """אתה עוזר AI אינטליגנטי שמשתמש במידע מהמסמכים שסופקו לו.
כאשר אתה עונה:
1. הסתמך **רק** על המידע שסופק בהקשר (Context).
2. אם המידע אינו קיים בהקשר — אמור זאת בכנות.
3. ציין את המקורות הרלוונטיים בסוף תשובתך.
4. ענה בשפה שבה נשאלת השאלה.
5. היה ממצה, מדויק ומועיל."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "לא נמצא ANTHROPIC_API_KEY. "
                "הגדר אותו בקובץ .env או כמשתנה סביבה."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self.history: list[dict] = []   # היסטוריית שיחה

    def generate(
        self,
        question: str,
        context_chunks: list[SearchResult],
        use_history: bool = True,
        max_tokens: int = 1024,
    ) -> tuple[str, int]:
        """
        מייצר תשובה מ-Claude.
        מחזיר (answer_text, tokens_used).
        """
        # בנה קטע הקשר
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            src = chunk.source
            page_info = f", עמוד {chunk.page}" if chunk.page else ""
            context_parts.append(
                f"[מקור {i}: {src}{page_info}]\n{chunk.text}"
            )
        context_str = "\n\n---\n\n".join(context_parts)

        user_message = f"""הקשר מהמסמכים:
{context_str}

שאלה: {question}"""

        # בנה רשימת הודעות עם היסטוריה
        messages = []
        if use_history and self.history:
            messages.extend(self.history[-6:])  # עד 3 סבבי שיחה אחרונים
        messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=messages,
        )

        answer = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens

        # שמור להיסטוריה (בלי ה-context הארוך, רק שאלה ותשובה)
        if use_history:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})

        return answer, tokens

    def clear_history(self):
        self.history.clear()


# ══════════════════════════════════════════════════════════════════════════════
# RAG System — הממשק הראשי
# ══════════════════════════════════════════════════════════════════════════════

class RAGSystem:
    """
    מערכת RAG מלאה — ממשק אחד לכל הפעולות.

    שימוש בסיסי:
        rag = RAGSystem()
        rag.ingest("my_document.pdf")
        response = rag.query("מה כתוב במסמך?")
        print(response.answer)
    """

    def __init__(
        self,
        persist_dir: str = "./rag_db",
        collection: str = "documents",
        chunk_size: int = 600,
        chunk_overlap: int = 80,
        top_k: int = 5,
        api_key: Optional[str] = None,
    ):
        self.top_k = top_k
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.vector_store = VectorStore(persist_dir=persist_dir, collection=collection)
        self.generator = ClaudeGenerator(api_key=api_key)

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(self, source: str) -> int:
        """
        טוען מסמך / URL ומוסיף לבסיס הידע.
        מחזיר מספר ה-chunks שנוספו.
        """
        print(f"\n📥 טוען: {source}")
        raw = self.loader.load(source)
        print(f"   נטענו {len(raw)} קטעי גלם, מחלק לחתיכות...")
        chunks = self.chunker.split_all(raw)
        print(f"   {len(chunks)} חתיכות נוצרו, מוסיף למסד הנתונים...")
        added = self.vector_store.add_chunks(chunks)
        print(f"✅ הושלם! נוספו {added} חתיכות חדשות מ-{source}")
        return added

    def ingest_directory(self, directory: str, extensions: Optional[list[str]] = None) -> int:
        """טוען כל המסמכים בתיקייה"""
        exts = extensions or [".pdf", ".docx", ".txt", ".md"]
        files = [f for f in Path(directory).rglob("*") if f.suffix.lower() in exts]
        total = 0
        for f in files:
            try:
                total += self.ingest(str(f))
            except Exception as e:
                print(f"⚠️  שגיאה בטעינת {f}: {e}")
        return total

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_history: bool = True,
        min_score: float = 0.0,
    ) -> RAGResponse:
        """
        שואל שאלה ומקבל תשובה עם מקורות.

        Args:
            question:    שאלת המשתמש
            top_k:       כמה chunks לשלוף (ברירת מחדל: self.top_k)
            use_history: האם להשתמש בהיסטוריית שיחה
            min_score:   ציון מינימלי לסינון תוצאות (0–1)
        """
        k = top_k or self.top_k
        results = self.vector_store.search(question, k=k)

        # סנן לפי ציון מינימלי
        results = [r for r in results if r.score >= min_score]

        if not results:
            return RAGResponse(
                question=question,
                answer="לא נמצא מידע רלוונטי בבסיס הידע. "
                       "אנא טען מסמכים תחילה באמצעות `rag.ingest(...)`.",
                sources=[],
            )

        answer, tokens = self.generator.generate(
            question=question,
            context_chunks=results,
            use_history=use_history,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=results,
            tokens_used=tokens,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def list_sources(self) -> list[str]:
        """מחזיר רשימת כל המקורות הטעונים"""
        return self.vector_store.list_sources()

    def clear_knowledge_base(self):
        """מוחק את כל המסמכים"""
        self.vector_store.clear()
        print("🗑️  בסיס הידע נוקה.")

    def clear_conversation(self):
        """מוחק היסטוריית שיחה"""
        self.generator.clear_history()
        print("🗑️  היסטוריית השיחה נוקתה.")

    def stats(self) -> dict:
        """סטטיסטיקות על המערכת"""
        return {
            "total_chunks": self.vector_store.col.count(),
            "sources": self.list_sources(),
            "history_turns": len(self.generator.history) // 2,
            "embed_model": self.vector_store.EMBED_MODEL,
            "llm_model": self.generator.MODEL,
        }
