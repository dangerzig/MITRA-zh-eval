#!/bin/bash
# Full MITRA-zh-eval replication pipeline
# Faithful reproduction of Nehrdich et al. 2025 methodology
set -e
cd ~/MITRA-zh-eval
source .venv/bin/activate

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set" >&2
    exit 1
fi
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "Error: ANTHROPIC_API_KEY not set" >&2
    exit 1
fi

TARGET=2168
LOG=~/MITRA-zh-eval/eval_run.log

echo "$(date): MITRA-zh-eval faithful replication" | tee $LOG
echo "  Judge model: gemini-flash-lite-latest" | tee -a $LOG
echo "  Translation prompt: Nehrdich Appendix A (non-RAG)" | tee -a $LOG
echo "  Scoring prompt: GEMBA-DA (Kocmi & Federmann 2023)" | tee -a $LOG

# --- Wait for all translations ---
echo "$(date): Waiting for translations..." | tee -a $LOG
while true; do
    S=$(wc -l < translations/sonnet.jsonl 2>/dev/null || echo 0)
    O=$(wc -l < translations/opus.jsonl 2>/dev/null || echo 0)
    H=$(wc -l < translations/haiku.jsonl 2>/dev/null || echo 0)
    echo "  Sonnet: $S  Opus: $O  Haiku: $H  (target: $TARGET each)"

    PROCS=$(ps aux | grep "translate.py" | grep -v grep | wc -l | tr -d ' ')
    if [ "$PROCS" -eq 0 ]; then
        echo "$(date): Translation processes exited (S=$S O=$O H=$H)" | tee -a $LOG
        # Validate: at least 90% complete for each model
        MIN=$(( TARGET * 9 / 10 ))
        if [ "$S" -lt "$MIN" ] || [ "$O" -lt "$MIN" ] || [ "$H" -lt "$MIN" ]; then
            echo "ERROR: Translations incomplete! S=$S O=$O H=$H (min=$MIN)" | tee -a $LOG
            echo "Relaunch translate.py processes and rerun this script." | tee -a $LOG
            exit 1
        fi
        break
    fi
    sleep 30
done

# --- Deduplicate ---
echo "$(date): Deduplicating..." | tee -a $LOG
for f in translations/sonnet.jsonl translations/opus.jsonl translations/haiku.jsonl; do
    [ -f "$f" ] || continue
    python3 -c "
import json
seen, lines = set(), []
with open('$f') as fh:
    for line in fh:
        obj = json.loads(line)
        if obj['id'] not in seen:
            seen.add(obj['id'])
            lines.append(line)
with open('$f', 'w') as fh:
    fh.writelines(lines)
print(f'  $f: {len(lines)} unique')
"
done

# --- GEMBA* (reference-free) for all three models ---
for MODEL in sonnet opus haiku; do
    echo "$(date): GEMBA* scoring $MODEL..." | tee -a $LOG
    python3 scripts/score_gemba.py \
        --translations translations/${MODEL}.jsonl \
        --reference-free \
        --output results/${MODEL}_gemba_star.jsonl \
        2>&1 | tee -a $LOG
done

# --- GEMBA (reference-based) for BDK subset ---
for MODEL in sonnet opus haiku; do
    echo "$(date): GEMBA ref scoring $MODEL..." | tee -a $LOG
    python3 scripts/score_gemba.py \
        --translations translations/${MODEL}.jsonl \
        --references data/reference_translations.jsonl \
        --output results/${MODEL}_gemba_ref.jsonl \
        2>&1 | tee -a $LOG
done

# --- Generate reports ---
echo "$(date): Generating reports..." | tee -a $LOG

python3 scripts/compare.py \
    --scores results/haiku_gemba_star.jsonl results/sonnet_gemba_star.jsonl results/opus_gemba_star.jsonl \
    --labels "Claude Haiku 4.5" "Claude Sonnet 4" "Claude Opus 4" \
    --output results/comparison_gemba_star.md

python3 scripts/compare.py \
    --scores results/haiku_gemba_ref.jsonl results/sonnet_gemba_ref.jsonl results/opus_gemba_ref.jsonl \
    --labels "Claude Haiku 4.5 (ref)" "Claude Sonnet 4 (ref)" "Claude Opus 4 (ref)" \
    --output results/comparison_gemba_ref.md

# --- Final summary ---
echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "FINAL RESULTS" | tee -a $LOG
echo "============================================" | tee -a $LOG
python3 -c "
import json, statistics, os

print('=== GEMBA* (reference-free, all sentences) ===')
for label, path in [
    ('Haiku 4.5', 'results/haiku_gemba_star.jsonl'),
    ('Sonnet 4',  'results/sonnet_gemba_star.jsonl'),
    ('Opus 4',    'results/opus_gemba_star.jsonl'),
]:
    if not os.path.exists(path): continue
    scores = [json.loads(l)['score'] for l in open(path)]
    if scores:
        print(f'  {label}: avg={statistics.mean(scores):.2f}, med={statistics.median(scores):.1f}, std={statistics.stdev(scores):.1f}, n={len(scores)}')

print()
print('=== GEMBA (reference-based, BDK subset) ===')
for label, path in [
    ('Haiku 4.5', 'results/haiku_gemba_ref.jsonl'),
    ('Sonnet 4',  'results/sonnet_gemba_ref.jsonl'),
    ('Opus 4',    'results/opus_gemba_ref.jsonl'),
]:
    if not os.path.exists(path): continue
    scores = [json.loads(l)['score'] for l in open(path)]
    if scores:
        print(f'  {label}: avg={statistics.mean(scores):.2f}, med={statistics.median(scores):.1f}, std={statistics.stdev(scores):.1f}, n={len(scores)}')

print()
print('Nehrdich et al. 2025 baselines (Gemini 2.0 Flash judge):')
print('  Claude 3.5 Sonnet RAG: 85.75')
print('  Claude 3.5 Sonnet:     84.65')
print('  Claude 3.5 Haiku:      81.09')
print('  GPT-4 Turbo:           82.05')
print('  Gemini 1.5 Pro:        82.68')
print()
print('Note: Our judge is gemini-flash-lite-latest (closest available')
print('calibration to the retired Gemini 2.0 Flash used by Nehrdich).')
" 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "$(date): Pipeline complete." | tee -a $LOG
echo "Reports: ~/MITRA-zh-eval/results/comparison_*.md" | tee -a $LOG
echo "Full log: ~/MITRA-zh-eval/eval_run.log" | tee -a $LOG
