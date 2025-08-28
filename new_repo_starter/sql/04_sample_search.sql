-- Example: embed a query offline, then pass its vector here as :q_main and :q_rivals
-- This file shows the shape of a two-stage search. Replace placeholders in your app.

-- Stage 1: top N by family/intent (here we just score all variants)
-- select id, make, model, series_years,
--        0.75 * (1 - (vector_main <=> :q_main)) + 0.25 * (1 - (vector_rivals <=> :q_rivals)) as score
-- from public.car_embeddings
-- order by score desc
-- limit 200;

-- Stage 2: re-rank within families (requires a family_id column if you add one)
-- with top_families as (
--   select make, model, series_years
--   from public.car_embeddings
--   order by 0.75 * (1 - (vector_main <=> :q_main)) + 0.25 * (1 - (vector_rivals <=> :q_rivals)) desc
--   limit 150
-- )
-- select e.*,
--        (1 - (e.vector_specs <=> :q_main)) as vscore
-- from public.car_embeddings e
-- join top_families f using (make, model, series_years)
-- order by vscore desc
-- limit 50;


