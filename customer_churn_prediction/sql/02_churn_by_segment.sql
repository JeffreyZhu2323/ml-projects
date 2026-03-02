SELECT
  Contract,
  `Internet Service` AS internet_service,
  COUNT(*) AS n,
  AVG(CASE WHEN `Churn Label` = TRUE THEN 1 ELSE 0 END) AS churn_rate
FROM `customerchurn-488906.telcocustomerchurn.churn_raw`
GROUP BY Contract, `Internet Service`
HAVING n >= 50
ORDER BY churn_rate DESC;