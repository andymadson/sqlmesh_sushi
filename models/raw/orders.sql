MODEL (
  name raw.orders,
  kind SEED (
    path '../../seeds/orders.csv'
  ),
  columns (
    id INT,
    customer_id INT,
    waiter_id INT,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP,
    ds TEXT
  ),
  grain "id, ds"
);
