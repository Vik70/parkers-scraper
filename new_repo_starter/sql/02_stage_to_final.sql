-- Move rows from stage → final and cast types

-- Optional: clean load
truncate public.car_embeddings;

-- If you need to coerce Python-style single-quoted JSON to JSONB, replace
-- nullif(rivals_json_family,'')::jsonb with a safer function.

insert into public.car_embeddings
  (make, model, series_years, essence_main, specs_summary_compact,
   rivals_json_family,
   engine_type, transmission_type,
   bhp_min, bhp_max,
   insurance_group_min, insurance_group_max,
   mpg_min, mpg_max,
   price_min, price_max,
   vector_main_text, vector_specs_text, vector_rivals_text)
select
  make,
  model,
  series_years,
  essence_main,
  specs_summary_compact,
  nullif(rivals_json_family,'')::jsonb,
  engine_type,
  transmission_type,
  nullif(bhp_min,'')::integer,
  nullif(bhp_max,'')::integer,
  nullif(insurance_group_min,'')::integer,
  nullif(insurance_group_max,'')::integer,
  nullif(mpg_min,'')::integer,
  nullif(mpg_max,'')::integer,
  nullif(price_min,'')::integer,
  nullif(price_max,'')::integer,
  vector_main_text,
  vector_specs_text,
  vector_rivals_text
from public.car_embeddings_stage;

-- Cast text arrays to vectors
update public.car_embeddings
set
  vector_main  = vector_main_text::vector,
  vector_specs = vector_specs_text::vector,
  vector_rivals= vector_rivals_text::vector;


