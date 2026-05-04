"""
push_to_github.py — יוצר repository ב-GitHub ומעלה את כל הקוד
==============================================================
שימוש:
    python push_to_github.py                        # ישאל לtoken
    python push_to_github.py --token ghp_XXX        # token מוגדר מראש
"""

import os
import sys
import subprocess
import getpass
import requests
from pathlib import Path


# ─── הגדרות ────────────────────────────────────────────────────────────────
REPO_NAME        = "rag-system"
REPO_DESCRIPTION = "מערכת RAG חכמה לשאלות ותשובות על מסמכי משאבי אנוש — Claude AI + ChromaDB"
REPO_PRIVATE     = True

# קבצים ותיקיות להעלאה
FILES_TO_COMMIT = [
    "streamlit_app.py",
    "rag_system.py",
    "build_db.py",
    "push_to_github.py",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    ".python-version",
    ".streamlit/config.toml",
    "מדריך-פרסום.md",
]
DIRS_TO_COMMIT = ["docs"]   # PDFs — רק ב-private repo


# ─── עזר ───────────────────────────────────────────────────────────────────
def ok(m):   print(f"  ✅ {m}")
def err(m):  print(f"  ❌ {m}")
def info(m): print(f"  ℹ️  {m}")
def step(n, t, m): print(f"\n[{n}/{t}] {m}")


def get_token():
    """קרא token מארגומנט --token, משתנה סביבה, או ישאל."""
    for i, arg in enumerate(sys.argv):
        if arg == "--token" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    t = os.getenv("GH_TOKEN", "").strip()
    if t:
        return t
    print("\n🔑 GitHub Personal Access Token נדרש.")
    print("   צור ב: https://github.com/settings/tokens/new (הרשאה: repo)")
    return getpass.getpass("   הדבק token: ").strip()


def gh_headers(token):
    return {"Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"}


def get_user(token):
    r = requests.get("https://api.github.com/user", headers=gh_headers(token), timeout=15)
    if r.status_code == 401:
        err("Token לא תקין.")
        sys.exit(1)
    r.raise_for_status()
    return r.json()


def ensure_repo(token, username):
    """מחזיר נתוני repo קיים או יוצר חדש."""
    h = gh_headers(token)
    chk = requests.get(f"https://api.github.com/repos/{username}/{REPO_NAME}",
                       headers=h, timeout=15)
    if chk.status_code == 200:
        info(f"Repository '{REPO_NAME}' כבר קיים.")
        return chk.json()
    r = requests.post("https://api.github.com/user/repos", headers=h, timeout=15,
                      json={"name": REPO_NAME, "description": REPO_DESCRIPTION,
                            "private": REPO_PRIVATE, "auto_init": False})
    if r.status_code not in (200, 201):
        err(f"יצירת repo נכשלה ({r.status_code}): {r.text[:200]}")
        sys.exit(1)
    return r.json()


def run(args, cwd="."):
    res = subprocess.run(["git"] + args, cwd=cwd,
                         capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0 and res.stderr.strip():
        bad = ["warning:", "hint:", "already exists", "nothing to commit"]
        if not any(w in res.stderr.lower() for w in bad):
            print(f"    git {' '.join(args)}: {res.stderr.strip()[:120]}")
    return res.returncode == 0


def push(token, username):
    cwd = str(Path(".").resolve())
    remote = f"https://{token}@github.com/{username}/{REPO_NAME}.git"

    run(["init", "-b", "main"], cwd)
    run(["config", "user.email", f"{username}@users.noreply.github.com"], cwd)
    run(["config", "user.name", username], cwd)
    run(["remote", "remove", "origin"], cwd)
    run(["remote", "add", "origin", remote], cwd)

    added = 0
    for f in FILES_TO_COMMIT:
        if Path(f).exists():
            run(["add", f], cwd)
            print(f"    + {f}")
            added += 1
        else:
            print(f"    ⚠️  {f} — לא נמצא, ידולג")

    for d in DIRS_TO_COMMIT:
        if Path(d).exists():
            run(["add", d + "/"], cwd)
            n = sum(1 for _ in Path(d).rglob("*") if _.is_file())
            print(f"    + {d}/  ({n} קבצים)")
            added += n
        else:
            print(f"    ⚠️  {d}/ — לא נמצא")

    if added == 0:
        err("אין קבצים להעלות.")
        sys.exit(1)

    run(["commit", "-m", "Initial commit — RAG system with HR documents"], cwd)
    ok_push = run(["push", "-u", "origin", "main", "--force"], cwd)
    if not ok_push:
        err("Push נכשל. בדוק חיבור אינטרנט ו-token.")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 56)
    print("   🚀  העלאה ל-GitHub — מערכת RAG")
    print("═" * 56)

    step(1, 4, "אימות ב-GitHub")
    token    = get_token()
    user     = get_user(token)
    username = user["login"]
    ok(f"מחובר כ: {username}  ({user.get('name','')})")

    step(2, 4, f"יצירת repository '{REPO_NAME}'")
    repo = ensure_repo(token, username)
    vis  = "פרטי 🔒" if repo.get("private") else "ציבורי 🌍"
    ok(f"{repo['html_url']}  [{vis}]")

    step(3, 4, "מעלה קבצים ל-GitHub")
    push(token, username)
    ok("כל הקבצים הועלו!")

    step(4, 4, "קישורים לפרסום")

    # קרא API key מ-.env
    api_hint = "sk-ant-..."
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                v = line.split("=", 1)[1].strip()
                api_hint = v[:16] + "..." if len(v) > 16 else v

    print(f"""
  ════════════════════════════════════════════════════
   הקוד ב-GitHub! עכשיו לפרסם ב-Streamlit Cloud:
  ════════════════════════════════════════════════════

   1. כנס ל:  https://share.streamlit.io
   2. לחץ "New app"
   3. Repository:  {username}/{REPO_NAME}
   4. Main file:   streamlit_app.py
   5. Advanced settings → Secrets הכנס:

      ANTHROPIC_API_KEY = "{api_hint}"

   6. לחץ "Deploy!" ✨

   כתובת האתר:
   https://{username}-{REPO_NAME}-streamlit-app-XXXXX.streamlit.app
  ════════════════════════════════════════════════════
    """)


if __name__ == 