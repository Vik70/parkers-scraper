-- HNSW indexes for cosine on all three vectors
create index if not exists car_embeddings_vec_main_idx
  on public.car_embeddings using hnsw (vector_main vector_cosine_ops);

create index if not exists car_embeddings_vec_specs_idx
  on public.car_embeddings using hnsw (vector_specs vector_cosine_ops);

create index if not exists car_embeddings_vec_rivals_idx
  on public.car_embeddings using hnsw (vector_rivals vector_cosine_ops);


