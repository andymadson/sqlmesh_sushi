-- File: models/raw/order_items.sql
MODEL (
  name raw.order_items,
  kind SEED (
    path '../../seeds/order_items.csv'
  ),
  columns (
    id INT,
    order_id INT,
    item_id INT,
    quantity INT,
    ds TEXT
  ),
  grain "id, ds"
);
