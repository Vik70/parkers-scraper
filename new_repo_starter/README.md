Project starter: variant descriptions + embeddings + Supabase load

Overview
- Build per‑variant user‑focused texts, embed three fields per row, and load to Supabase with pgvector for cosine ANN search.

Artifacts in this starter
- scripts
  - export_for_supabase.py: convert an embedded Excel to a CSV with vector_*_text columns ready for DB import
  - run_llm.ps1: resume‑aware wrapper to run build_essence_texts.py reliably with logging
- sql
  - 01_schema.sql: create final table (vectors) and a text‑only staging table
  - 02_stage_to_final.sql: move rows from stage → final, coerce JSON, cast vectors
  - 03_indexes.sql: HNSW indexes for cosine
  - 04_sample_search.sql: example query snippets for intent/rivals search

Pipeline (end‑to‑end)
1) Generate texts per row
   - Use build_essence_texts.py (llm_variant mode) to produce:
     - essence_main (80–150 words; includes size, character, practicality, highlights, buyer fit)
     - specs_summary_compact (short facts)
     - rivals_json_family ([] where unknown)
   - The script logs "[base]" for the rules phase and "[llm]" for the refinement.
   - Use run_llm.ps1 to run with resume/checkpoints on Windows.

2) Embed three texts per row
   - Use embed_text_vectors.py to produce:
     - vector_main ← essence_main
     - vector_specs ← specs_summary_compact
     - vector_rivals ← compacted rivals string (built from rivals_json_family)
   - Vectors are L2‑normalized and stored as JSON arrays in Excel.

3) Export for database import
   - Run export_for_supabase.py to output car_embeddings_upload_ordered.csv with:
     - make, model, series_years, essence_main, specs_summary_compact,
       rivals_json_family, vector_main_text, vector_specs_text, vector_rivals_text

4) Load to Supabase (via DBeaver or psql)
   - Run 01_schema.sql to create tables and enable pgvector.
   - Import CSV into public.car_embeddings_stage (TEXT table).
   - Run 02_stage_to_final.sql to move to public.car_embeddings and cast vectors.
   - Run 03_indexes.sql to create HNSW indexes (cosine).

Search recipe (high‑level)
- Stage 1 (family/intent): order by 0.75 * cosine(vector_main, q_intent) + 0.25 * cosine(vector_rivals, q_rivals)
- Stage 2 (within top families): re‑rank variants using γ * cosine(vector_specs or vector_main, q_intent) + numeric filters.

Notes
- rivals_json_family may be [] for many rows; you can backfill later and re‑embed rivals only.
- Query embeddings should be L2‑normalized to match stored vectors.

Quick commands
- Export CSV:
  ```powershell
  python new_repo_starter/scripts/export_for_supabase.py --input tmp_llm_snapshot_embedded.xlsx --output car_embeddings_upload_ordered.csv --series-col "Series (production years start-end)"
  ```
- Load (DBeaver): import CSV → public.car_embeddings_stage, then run 02_stage_to_final.sql and 03_indexes.sql


