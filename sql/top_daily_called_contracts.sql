WITH filtered_transactions AS (
  SELECT
    to_address,
    COUNT(*) AS call_count,
    DATE(block_timestamp) AS transaction_date
  FROM
    `bigquery-public-data.crypto_ethereum.transactions`
  WHERE
    block_timestamp >= '2024-01-01' 
    -- block_timestamp < '2024-01-01'
  GROUP BY
    to_address, transaction_date
),
daily_average AS (
  SELECT
    to_address,
    SUM(call_count) AS total_calls,
    COUNT(DISTINCT transaction_date) AS active_days, -- Count the number of unique active days
    SUM(call_count) / COUNT(DISTINCT transaction_date) AS daily_avg_calls
  FROM
    filtered_transactions
  GROUP BY
    to_address
)
SELECT
  to_address,
  total_calls,
  daily_avg_calls
FROM
  daily_average
ORDER BY
  total_calls DESC
LIMIT
  50;
