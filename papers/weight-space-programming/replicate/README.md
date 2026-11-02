# Replication

*AI-sourced: written by an AI agent.*

Step 1 (probe): rebuild Qwen3.5-0.8B with the authors' released edits (`edits/`, from RikiyaT/lm_understand-logs, CC BY 4.0; `vendor/` scripts MIT) and score samples of MATH-500 and GSM8K with the paper's thinking-on protocol. Paper scores for the full sets are in `edits/paper_*.json`.

    ./run.sh <name> <problems per set> <samples per problem> [unedited,p50c,...]

Runs on a local RTX 3090 through a one-job GPU queue. Results land in `results/<name>/`.
