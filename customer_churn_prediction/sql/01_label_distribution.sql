SELECT
  `Churn Value` AS churn_value,
  COUNT(*) AS n,
  ROUND(COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS pct
FROM `customerchurn-488906.telcocustomerchurn.churn_raw`
GROUP BY `Churn Value`
ORDER BY `Churn Value` DESC;