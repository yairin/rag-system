"""
build_db.py — סקריפט חד-פעמי לבניית בסיס הידע מהמסמכים
=========================================================
הרץ פעם אחת לפני העלאה ל-GitHub:
    python build_db.py

הסקריפט:
  1. מחפש את כל קבצי ה-PDF בתיקיית המסמכים
  2. מחלק אותם לחתיכות טקסט
  3. מייצר embeddings (וקטורים) לכל חתיכה
  4. שומר את הכל ב-ChromaDB בתיקיית rag_db/

לאחר הרצה — העלה את תיקיית rag_db/ ל-GitHub יחד עם שאר הקבצים.
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── הגדרות ────────────────────────────────────────────────────────────────────

# תיקיית המסמכים שלך — שנה אם נדרש
DOCS_DIR = r"C:\Users\yair\Claude\קבצי מידע ל rag"

# תיקיית הפלט (ChromaDB) — תיכנס ל-GitHub
OUTPUT_DIR = "./rag_db"

# הגדרות chunking
CHUNK_SIZE  = 600   # תווים לחתיכה
CHUNK_OVERLAP = 80  # תווים חפיפה בין חתיכות


# ══════════════════════════════════════════════════════════════════════════════

def print_header():
    print("\n" + "═" * 60)
    print("   🔨  בניית בסיס הידע — מערכת RAG")
    print("═" * 60)

def print_step(step: int, total: int, msg: str):
    print(f"\n[{step}/{total}] {msg}")

def print_success(msg: str):
    print(f"  ✅ {msg}")

def print_info(msg: str):
    print(f"  ℹ️  {msg}")

def print_error(msg: str):
    print(f"  ❌ {msg}")


def main():
    print_header()
    t0 = time.time()

    # ── שלב 1: בדיקת תיקיית מסמכים ─────────────────────────────────────────
    print_step(1, 5, "בדיקת תיקיית המסמכים...")
    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        print_error(f"התיקייה לא נמצאה: {DOCS_DIR}")
        print_info("ערוך את DOCS_DIR בתוך build_db.py")
        sys.exit(1)

    # מצא קבצים נתמכים
    supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}
    files = [f for f in docs_path.rglob("*") if f.suffix.lower() in supported]

    if not files:
        print_error("לא נמצאו קבצים נתמכים בתיקייה.")
        sys.exit(1)

    print_success(f"נמצאו {len(files)} קבצים:")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        print(f"     📄 {f.name}  ({size_kb:.0f} KB)")

    # ── שלב 2: ייבוא מסגרת ה-RAG ────────────────────────────────────────────
    print_step(2, 5, "טוען מסגרת ה-RAG...")
    try:
        from rag_system import DocumentLoader, TextChunker, VectorStore
        print_success("מסגרת RAG נטענה בהצלחה")
    except ImportError as e:
        print_error(f"שגיאת ייבוא: {e}")
        print_info("ודא שהרצת: pip install -r requirements.txt")
        sys.exit(1)

    # ── שלב 3: טעינת וחלוקת מסמכים ──────────────────────────────────────────
    print_step(3, 5, "טוען ומחלק מסמכים לחתיכות...")
    loader  = DocumentLoader()
    chunker = TextChunker(chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    all_chunks = []
    failed = []

    for f in sorted(files):
        print(f"  ⏳ מעבד: {f.name}", end="", flush=True)
        try:
            raw = loader.load(str(f))
            chunks = chunker.split_all(raw)
            all_chunks.extend(chunks)
            print(f" → {len(chunks)} חתיכות ✓")
        except Exception as e:
            failed.append((f.name, str(e)))
            print(f" → ❌ שגיאה: {e}")

    if not all_chunks:
        print_error("לא נוצרו חתיכות. בדוק את קבצי הקלט.")
        sys.exit(1)

    print_success(f"סה\"כ {len(all_chunks):,} חתיכות נוצרו מ-{len(files) - len(failed)} קבצים")

    if failed:
        print(f"\n  ⚠️  {len(failed)} קבצים נכשלו:")
        for name, err in failed:
            print(f"      {name}: {err}")

    # ── שלב 4: בניית ChromaDB ────────────────────────────────────────────────
    print_step(4, 5, "בונה מסד נתוני וקטורים (ChromaDB)...")
    print_info("טוען מודל embeddings... (עשוי לקחת כמה דקות בפעם הראשונה)")

    try:
        store = VectorStore(persist_dir=OUTPUT_DIR)
        added = store.add_chunks(all_chunks)
        print_success(f"נוספו {added:,} חתיכות חדשות למסד הנתונים")
        print_success(f"סה\"כ במסד: {store.col.count():,} חתיכות")
    except Exception as e:
        print_error(f"שגיאה בבניית ChromaDB: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── שלב 5: סיכום ─────────────────────────────────────────────────────────
    print_step(5, 5, "סיכום")
    elapsed = time.time() - t0
    db_size = sum(f.stat().st_size for f in Path(OUTPUT_DIR).rglob("*") if f.is_file())

    print(f"""
  ✅ הבנייה הושלמה בהצלחה בתוך {elapsed:.1f} שניות!

  📊 סטטיסטיקות:
     • קבצים שעובדו:  {len(files) - len(failed)}/{len(files)}
     • חתיכות טקסט:  {store.col.count():,}
     • גודל rag_db/:  {db_size / 1024:.0f} KB

  🚀 השלבים הבאים:
     1. העלה את כל הקבצים ל-GitHub (כולל תיקיית rag_db/)
     2. חבר את ה-repo ל-Streamlit Cloud
     3. הגדר ANTHROPIC_API_KEY ב-Secrets של Streamlit Cloud
     4. לחץ Deploy!

  📁 הקבצים ל-GitHub:
     streamlit_app.py  ✓
     rag_system.py     ✓
     requirements.txt  ✓
     .streamlit/       ✓
     rag_db/           ✓  (נוצר עכשיו)
    """)

    if failed:
        print(f"  ⚠️  {len(failed)} קבצים לא עובדו — בדוק שגיאות מעל.")


if __name__ == "__main__":
    main()
