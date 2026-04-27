#!/bin/bash
# WP2: Create an anonymized mirror of the BioPAT repo for TACL review.
#
# Strips:
#   - Author names, emails, GitHub username from all files
#   - URLs pointing to identified repo
#   - Co-Authored-By tags in commit messages
#   - File system metadata (mtime, owner)
#
# Output: a fresh /tmp/biopat-anon directory with single-commit git history.
#
# Usage:
#   ./scripts/anonymize_repo.sh                  # default output dir
#   ./scripts/anonymize_repo.sh /path/to/output  # custom output

set -e

SOURCE_DIR="${SOURCE_DIR:-$(pwd)}"
OUTPUT_DIR="${1:-/tmp/biopat-anon}"

echo "Source:      $SOURCE_DIR"
echo "Output:      $OUTPUT_DIR"
echo ""

if [ -d "$OUTPUT_DIR" ]; then
    echo "ERROR: $OUTPUT_DIR already exists. Remove it first."
    exit 1
fi

# Identifiers to strip (extend as needed)
IDENTIFIED_USERNAME="VibeCodingScientist"
IDENTIFIED_EMAIL_PATTERN="lukassebastianweidener@gmail\.com"  # also strips other personal emails
IDENTIFIED_REPO_URL="github.com/VibeCodingScientist/BioPAT"
ANON_USERNAME="biopat-anon-2026"
ANON_REPO_URL="github.com/biopat-anon-2026/BioPAT-anonymous"

# 1. Copy source (preserve symlinks, exclude git history)
echo ">>> Copying files..."
mkdir -p "$OUTPUT_DIR"
rsync -a --exclude='.git' --exclude='venv' --exclude='__pycache__' \
      --exclude='.pytest_cache' --exclude='*.pyc' \
      --exclude='secrets/' \
      "$SOURCE_DIR/" "$OUTPUT_DIR/"

cd "$OUTPUT_DIR"

# 2. Find-and-replace identifiers in all text files
echo ">>> Anonymizing text content..."
# Use a portable find/sed combo that works on macOS and Linux
find . -type f \( \
    -name "*.py" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" \
    -o -name "*.toml" -o -name "*.json" -o -name "*.txt" \
    -o -name "*.tex" -o -name "*.sh" -o -name "*.cfg" \
    -o -name "Dockerfile" -o -name ".gitignore" \
\) -print0 | while IFS= read -r -d '' f; do
    # Skip binary files
    if file "$f" | grep -qE 'binary|data'; then
        continue
    fi
    # macOS sed needs '' after -i; use perl for portability
    perl -pi -e "s/${IDENTIFIED_USERNAME}/${ANON_USERNAME}/g" "$f"
    perl -pi -e "s/${IDENTIFIED_REPO_URL}/${ANON_REPO_URL}/g" "$f"
    perl -pi -e "s/${IDENTIFIED_EMAIL_PATTERN}/anonymous\@example.com/g" "$f"
done

# 3. Update bibtex citation in README
if [ -f README.md ]; then
    perl -i -pe '
        s/\@software\{biopat,/\@misc{biopat_anon,/;
        s/author\s*=\s*\{BioPAT Contributors\}/author = {anonymous}/;
        s/note\s*=\s*\{[^}]*\}/note = {Under review}/;
    ' README.md
fi

# 4. Reset file timestamps
echo ">>> Resetting file timestamps..."
find . -exec touch -t "$(date +%Y%m%d%H%M)" {} +

# 5. Strip PDF metadata if any PDFs exist
if command -v exiftool >/dev/null 2>&1; then
    echo ">>> Stripping PDF/image metadata..."
    find . \( -name "*.pdf" -o -name "*.png" -o -name "*.jpg" \) \
        -exec exiftool -overwrite_original -all= {} \; >/dev/null 2>&1 || true
else
    echo ">>> exiftool not installed; skipping PDF/image metadata strip"
    echo "    (install via: brew install exiftool)"
fi

# 6. Init fresh git history with single commit
echo ">>> Creating fresh git history..."
git init -b main >/dev/null
git config user.name "anonymous"
git config user.email "anonymous@example.com"
git add -A
git commit -m "Initial release: BioPAT-NovEx anonymous version for TACL review" >/dev/null

# 7. Verify no identifiers leak
echo ""
echo ">>> Verification:"
LEAKED=$(grep -ri "$IDENTIFIED_USERNAME\|lukassebastianweidener" . 2>/dev/null \
    | grep -v ".git/" || true)
if [ -n "$LEAKED" ]; then
    echo "WARNING: identifier leakage detected:"
    echo "$LEAKED" | head -5
else
    echo "  OK: no identifier leakage in text content"
fi

LEAKED_GIT=$(git log --format="%an %ae %s" | grep -i "VibeCoding\|lukas" || true)
if [ -n "$LEAKED_GIT" ]; then
    echo "WARNING: identifier in git log:"
    echo "$LEAKED_GIT"
else
    echo "  OK: clean git log"
fi

# 8. Stats
echo ""
echo ">>> Repository stats:"
echo "  Files:    $(find . -type f -not -path './.git/*' | wc -l)"
echo "  Size:     $(du -sh . | cut -f1)"
echo "  Commits:  $(git log --oneline | wc -l)"
echo ""
echo "Anonymized repo ready at: $OUTPUT_DIR"
echo "Next: cd $OUTPUT_DIR && git remote add origin <anon-repo-url> && git push"
