DECLARE victim_address STRING;
DECLARE starting_scope DATE;
DECLARE victim_create_date DATE;
DECLARE ending_scope TIMESTAMP;
DECLARE attack_datetime TIMESTAMP;

SET victim_address = '0x67B66C99D3Eb37Fa76Aa3Ed1ff33E8e39F0b9c7A'; #EDIT
SET attack_datetime = "2021-05-08T14:48:06"; #EDIT
SET victim_create_date = "2020-10-07"; #EDIT
SET ending_scope = attack_datetime;
SET starting_scope = victim_create_date;

WITH transaction_involved AS(
  SELECT transaction_hash,COUNT(*) as trace_involved_amt, LOWER(victim_address) as victim
  FROM `bigquery-public-data.crypto_ethereum.traces` 
  WHERE
    to_address = LOWER(victim_address)
    AND block_timestamp <= ending_scope
    AND DATE(block_timestamp) >= starting_scope
  GROUP BY transaction_hash
),

transaction_info AS(
  SELECT
    tx_db.hash as transaction_hash
    ,trace_involved_amt
    ,from_address
    ,to_address
    ,block_timestamp
    ,gas, gas_price
    ,receipt_cumulative_gas_used
    ,receipt_gas_used, value, nonce, transaction_type
    ,ROW_NUMBER() OVER (ORDER BY block_timestamp DESC) AS rn
    ,TIMESTAMP_SUB(DATE(tx_db.block_timestamp), INTERVAL 1 YEAR) AS lb_block_timestamp
  FROM `bigquery-public-data.crypto_ethereum.transactions` as tx_db
  JOIN transaction_involved ON transaction_involved.transaction_hash = tx_db.hash 
  WHERE
    block_timestamp <= ending_scope
    AND DATE(block_timestamp) >= starting_scope
  ORDER BY block_timestamp DESC
  LIMIT 1000
),

-- timestamp_stat AS (
--   SELECT MIN(block_timestamp) as min_timestamp
--   FROM transaction_info
-- ),

-- transaction_info AS(
--   SELECT *
--   FROM transaction_info_
--   WHERE rn BETWEEN 1 AND 500
-- ),

contract_info AS ( 
  SELECT
    transaction_hash as contract_tx_hash
    ,transaction_info.to_address as contract
    ,COUNT(DISTINCT block_number) as contract_block_involved
    ,COUNT(*) as contract_tx_count
    ,COUNT(DISTINCT block_number)/COUNT(*) as contract_block_per_tx
    ,COUNT(DISTINCT DATE(tx_db.block_timestamp)) as contract_main_active_days
  FROM transaction_info
  LEFT JOIN `bigquery-public-data.crypto_ethereum.transactions` as tx_db
  ON ((transaction_info.to_address = tx_db.to_address) OR (transaction_info.to_address = tx_db.from_address))
  WHERE
    tx_db.block_timestamp <= ending_scope
    AND DATE(tx_db.block_timestamp) >= TIMESTAMP_SUB(starting_scope, INTERVAL 1 YEAR)
    AND tx_db.block_timestamp <= transaction_info.block_timestamp
    AND DATE(tx_db.block_timestamp) >= transaction_info.lb_block_timestamp
  GROUP BY transaction_hash, transaction_info.to_address
),

sender_info AS (
  SELECT
    transaction_hash as sender_tx_hash
    ,transaction_info.from_address as sender
    ,COUNT(DISTINCT block_number) as sender_block_involved
    ,COUNT(*) as sender_tx_count
    ,COUNT(DISTINCT block_number)/COUNT(*) as sender_block_per_tx
    ,COUNT(DISTINCT DATE(tx_db.block_timestamp)) as sender_main_active_days
  FROM transaction_info
  LEFT JOIN `bigquery-public-data.crypto_ethereum.transactions` as tx_db
  ON ((transaction_info.from_address = tx_db.from_address) OR (transaction_info.from_address = tx_db.to_address))
  WHERE
    tx_db.block_timestamp <= ending_scope
    AND DATE(tx_db.block_timestamp) >= TIMESTAMP_SUB(starting_scope, INTERVAL 1 YEAR)
    AND tx_db.block_timestamp <= transaction_info.block_timestamp
    AND DATE(tx_db.block_timestamp) >= transaction_info.lb_block_timestamp
  GROUP BY transaction_hash, transaction_info.from_address
),
-- !
###! TEST ###
interactions AS (
  SELECT
    transaction_hash as interact_hash,
    ROW_NUMBER() OVER (PARTITION BY to_address ORDER BY block_timestamp) AS contract_interact,
  FROM transaction_info
),

################ trace_amt #################
traces AS (
  select
    tc.transaction_hash,
    COUNT(*) as trace_amt
  from `bigquery-public-data.crypto_ethereum.traces` as tc
  where
    tc.transaction_hash in (
      select transaction_hash
      from transaction_involved
    )
    AND block_timestamp <= ending_scope
    AND DATE(block_timestamp) >= starting_scope
  group by transaction_hash
),
################ distinct_sender_in_contract #################
callers_in_contract AS (
  select
    transaction_info.transaction_hash,
    transaction_info.from_address as sender,
    tx.to_address,
    tx.from_address,
    tx.block_timestamp,
    tx.block_number
  from transaction_info
  join `bigquery-public-data.crypto_ethereum.transactions` as tx
  on transaction_info.to_address = tx.to_address
  WHERE
    tx.block_timestamp <= ending_scope
    AND DATE(tx.block_timestamp) >= TIMESTAMP_SUB(starting_scope, INTERVAL 1 YEAR)
    AND tx.block_timestamp <= transaction_info.block_timestamp
    AND DATE(tx.block_timestamp) >= transaction_info.lb_block_timestamp
),
contract_total_callers AS (
  select
    transaction_hash,
    to_address,
    COUNT(distinct from_address) as distinct_sender_in_contract,
    ABS(date_diff(DATE(max(block_timestamp)), DATE(min(block_timestamp)),DAY)) as contract_lifetime_days,
    ABS(max(block_number)-min(block_number)) as contract_lifetime_block
  from callers_in_contract
  group by transaction_hash, to_address
),
################ distinct_contract_s_called #################
contracts_in_caller AS (
  select
    transaction_info.transaction_hash,
    transaction_info.from_address,
    transaction_info.to_address as contract_called,
    tx.to_address,
    tx.block_timestamp,
    tx.block_number
  from transaction_info
  join `bigquery-public-data.crypto_ethereum.transactions` as tx
  on transaction_info.from_address = tx.from_address
  WHERE
    tx.block_timestamp <= ending_scope
    AND DATE(tx.block_timestamp) >= TIMESTAMP_SUB(starting_scope, INTERVAL 1 YEAR)
    AND tx.block_timestamp <= transaction_info.block_timestamp
    AND DATE(tx.block_timestamp) >= transaction_info.lb_block_timestamp
),
caller_total_contracts AS (
  select
    transaction_hash,
    from_address,
    COUNT(distinct to_address) as distinct_contract_sender_called, 
    ABS(date_diff(DATE(max(block_timestamp)), DATE(min(block_timestamp)),DAY)) as sender_lifetime_days,
    ABS(max(block_number)-min(block_number)) as sender_lifetime_block
  from contracts_in_caller
  group by transaction_hash, from_address
),

################ s_tx_count_call_c, s_days_call_c #################
caller_stats_in_contract AS (
  select
    transaction_hash,
    from_address,
    to_address,
    COUNT(*) as sender_tx_count_call_contract,
    COUNT(distinct date((block_timestamp))) as sender_days_call_contract
  from callers_in_contract
  where sender = from_address
  group by
    transaction_hash,
    from_address,
    to_address
),

distinct_from as (
  SELECT transaction_info.transaction_hash, COUNT(DISTINCT IF(tc.to_address = transaction_info.to_address,tc.from_address,NULL)) as distinct_contract_called
  FROM `bigquery-public-data.crypto_ethereum.traces` as tc
  JOIN transaction_info ON transaction_info.transaction_hash = tc.transaction_hash
  WHERE
    tc.block_timestamp <= transaction_info.block_timestamp
    AND tc.block_timestamp <= ending_scope
    AND DATE(tc.block_timestamp) >= starting_scope
  GROUP BY transaction_info.transaction_hash
)
, distinct_to as (
  SELECT transaction_info.transaction_hash, COUNT(DISTINCT IF(tc.from_address = transaction_info.to_address,tc.to_address,NULL)) as distinct_called_by 
  FROM `bigquery-public-data.crypto_ethereum.traces` as tc
  JOIN transaction_info ON transaction_info.transaction_hash = tc.transaction_hash
  WHERE
    tc.block_timestamp <= transaction_info.block_timestamp
    AND tc.block_timestamp <= ending_scope
    AND DATE(tc.block_timestamp) >= starting_scope
  GROUP BY transaction_info.transaction_hash
)
, distinct_contract as (
  SELECT transaction_hash,from_address as address FROM `bigquery-public-data.crypto_ethereum.traces`
  WHERE transaction_hash in (select transaction_hash from transaction_info)
    AND block_timestamp <= ending_scope
    AND DATE(block_timestamp) >= starting_scope
  UNION DISTINCT
  SELECT transaction_hash,to_address as address FROM `bigquery-public-data.crypto_ethereum.traces`
  WHERE transaction_hash in (select transaction_hash from transaction_info)
    AND block_timestamp <= ending_scope
    AND DATE(block_timestamp) >= starting_scope
)
, no_d_contract as (
  SELECT transaction_hash,COUNT(DISTINCT address) as contract_involved_amt
  FROM distinct_contract
  GROUP BY transaction_hash
)
, depth_and_breadth AS (
  SELECT 
    transaction_info.transaction_hash, 
    MAX((SELECT MAX(SAFE_CAST(value AS INT64)) FROM UNNEST(SPLIT(trace_address, ',')) AS value)) AS max_breadth, 
    MAX(ARRAY_LENGTH(SPLIT(trace_address,','))) AS depth
  FROM 
    `bigquery-public-data.crypto_ethereum.traces` AS traces
  JOIN 
    transaction_info
  ON 
    traces.transaction_hash = transaction_info.transaction_hash
  WHERE
    traces.block_timestamp <= ending_scope
    AND DATE(traces.block_timestamp) >= starting_scope
  GROUP BY 
    transaction_info.transaction_hash
    
)
, result as (
  SELECT no_d_contract.transaction_hash,contract_involved_amt,max_breadth,depth,distinct_contract_called, distinct_called_by
  FROM no_d_contract
  JOIN depth_and_breadth ON depth_and_breadth.transaction_hash = no_d_contract.transaction_hash
  JOIN distinct_from ON distinct_from.transaction_hash = no_d_contract.transaction_hash
  JOIN distinct_to ON distinct_to.transaction_hash = no_d_contract.transaction_hash
),

additional_features AS (
    SELECT
        transaction_info.transaction_hash as add_feat_hash,
        sender_tx_count_call_contract,
        sender_days_call_contract, 
        trace_amt,
        distinct_sender_in_contract,
        contract_lifetime_days,
        contract_lifetime_block,
        distinct_contract_sender_called,
        sender_lifetime_days,
        sender_lifetime_block,
        contract_involved_amt,
        max_breadth,depth,
        distinct_contract_called as distinct_was_called_in_sample,
        distinct_called_by as distinct_sender_call_in_sample,
    FROM transaction_info
    JOIN traces ON traces.transaction_hash = transaction_info.transaction_hash
    JOIN contract_total_callers ON contract_total_callers.transaction_hash = transaction_info.transaction_hash
    JOIN caller_total_contracts ON caller_total_contracts.transaction_hash = transaction_info.transaction_hash
    JOIN caller_stats_in_contract ON caller_stats_in_contract.transaction_hash = transaction_info.transaction_hash
    JOIN result ON result.transaction_hash = transaction_info.transaction_hash
)

SELECT * EXCEPT(sender,contract,contract_tx_hash,sender_tx_hash,interact_hash)
FROM transaction_info
JOIN contract_info ON contract_tx_hash = transaction_hash
JOIN sender_info ON sender_tx_hash = transaction_hash
JOIN interactions ON interact_hash = transaction_hash
JOIN additional_features ON add_feat_hash = transaction_hash