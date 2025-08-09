MODEL (
  name raw.items,
  kind SEED (
    path '../../seeds/items.csv'
  ),
  columns (
    id INT,
    name TEXT,
    price DOUBLE,
    ds TEXT
  ),
  grain "id, ds"
);