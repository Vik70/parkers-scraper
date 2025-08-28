-- Enable pgvector
create extension if not exists vector;

-- Final table with vectors
create table if not exists public.car_embeddings (
  id bigserial primary key,
  make text,
  model text,
  series_years text,
  essence_main text,
  specs_summary_compact text,
  rivals_json_family jsonb,
  -- optional structured fields for filters/analytics
  engine_type text,
  transmission_type text,
  bhp_min integer,
  bhp_max integer,
  insurance_group_min integer,
  insurance_group_max integer,
  mpg_min integer,
  mpg_max integer,
  price_min integer,
  price_max integer,
  vector_main_text  text,
  vector_specs_text text,
  vector_rivals_text text,
  vector_main  vector(1536),
  vector_specs vector(1536),
  vector_rivals vector(1536)
);

-- Text-only staging table for reliable CSV imports
create table if not exists public.car_embeddings_stage (
  make text,
  model text,
  series_years text,
  essence_main text,
  specs_summary_compact text,
  rivals_json_family text,
  engine_type text,
  transmission_type text,
  bhp_min text,
  bhp_max text,
  insurance_group_min text,
  insurance_group_max text,
  mpg_min text,
  mpg_max text,
  price_min text,
  price_max text,
  vector_main_text  text,
  vector_specs_text text,
  vector_rivals_text text
);


