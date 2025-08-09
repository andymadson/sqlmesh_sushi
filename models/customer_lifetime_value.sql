MODEL (
  name sushimoderate.customer_lifetime_value,
  kind FULL,
  owner analytics,
  grain customer_id,
  audits (
    unique_values(columns := customer_id),
    not_null(columns := (customer_id, lifetime_value))
  )
);

WITH per_customer AS (
  SELECT
    crl.customer_id::INT AS customer_id,
    MIN(CAST(crl.ds AS DATE)) AS first_order_date,
    MAX(CAST(crl.ds AS DATE)) AS last_order_date,
    COUNT(DISTINCT DATE_TRUNC('month', CAST(crl.ds AS DATE))) AS active_months,
    -- cumulative series: take the final cumulative revenue for each customer
    MAX(crl.revenue)::DOUBLE AS historical_revenue
  FROM sushimoderate.customer_revenue_lifetime AS crl
  GROUP BY crl.customer_id
)
SELECT
  pc.customer_id,
  pc.first_order_date,
  pc.active_months,
  pc.historical_revenue,
  CASE
    WHEN pc.active_months >= 6 THEN pc.historical_revenue * 2.5
    WHEN pc.active_months >= 3 THEN pc.historical_revenue * 2.0
    ELSE pc.historical_revenue * 1.5
  END AS lifetime_value
FROM per_customer AS pc;