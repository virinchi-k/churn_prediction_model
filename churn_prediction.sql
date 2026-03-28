-- -- Create a table in master database (not recommended for production, use user database instead)
-- USE master;
-- GO

-- DROP TABLE IF EXISTS ChurnData;
-- GO

-- CREATE TABLE ChurnData (
--     RowNumber INT,
--     CustomerId BIGINT,
--     Surname VARCHAR(50),
--     CreditScore INT,
--     Geography VARCHAR(20),
--     Gender VARCHAR(10),
--     Age INT,
--     Tenure INT,
--     Balance DECIMAL(18,2),
--     NumOfProducts INT,
--     HasCrCard INT,
--     IsActiveMember INT,
--     EstimatedSalary DECIMAL(18,2),
--     Exited INT
-- );
-- GO

-- -- Load CSV file
-- BULK INSERT ChurnData
-- FROM '/var/opt/mssql/data/Churn_Modelling.csv'
-- WITH (
--     FIELDTERMINATOR = ',',
--     ROWTERMINATOR = '\n',
--     FIRSTROW = 2  -- Skip header row
-- );
-- GO

-- Verify data
-- SELECT TOP 10 * FROM ChurnData;

/* High-Value Customer Retention Analysis: Question: For each geography, identify the top 3 customers with the highest Balance who are still active members 
(IsActiveMember = 1) and have not exited.*/

-- WITH CTE as (SELECT 
-- 	*,
--     RANK() OVER(partition by geography ORDER BY Balance DESC) as ranks
-- FROM ChurnData
-- WHERE IsActiveMember = 1 AND
-- 	Exited = 0
-- )

-- SELECT * from CTE
-- WHERE ranks <=3
-- ORDER BY geography, ranks;

/*Churn Rate by Credit Score Tier: Question: Define three credit tiers: 'Poor' (0-580), 'Fair' (581-669), and 'Good+' (670+). Calculate the churn rate (Exited) for each 
tier, ordered by the highest churn rate first. The output should include the tier name, total customers in that tier, and the percentage of churned customers.*/

-- with cte as (SELECT 
-- *, case when   CreditScore between 0 AND 580 THEN 'Poor'
--         when	CreditScore between 581 AND 669 THEN 'Fair'
--         when	CreditScore >= 670 THEN 'Good+' end as credit_tiers
--  FROM ChurnData)
 
-- SELECT 
--     credit_tiers,
--     count(*) as total_cust,
--     count(case when exited = 1 then 1 end) as churned_cust,
--     round(avg(exited * 100.00), 2) as churned_percentage    
-- from cte
-- group by credit_tiers
-- order by churned_percentage DESC;

/*3. Salary vs. Balance Correlation
Question: Find the average EstimatedSalary for customers who have a Balance higher than the overall average balance of the entire dataset. 
Additionally, count how many of these customers are considered "Loyal" (defined as having a Tenure of 5 years or more).*/
--select top 10 * from churndata

with cte as (select *, AVG(EstimatedSalary) as avgsal
FROM churndata)

select *, (select estimatedsalary from cte where estimatedsalary > avgsal ) as estsal
from cte


/*4. Product Penetration by Demographics
Question: For each combination of Gender and Geography, calculate the average number of products (NumOfProducts) held by customers. Only include groups where the average EstimatedSalary is greater than 100,000.

Technical Focus: Multi-level Grouping and HAVING clause filters.*/

/*5. Identifying Outlier Financial Behavior
Question: Identify customers whose Balance is more than two standard deviations above the mean balance for their specific Geography. List their CustomerId, Surname, Geography, and Balance.

Technical Focus: Standard Deviation functions (STDDEV or STDEV), Window Functions, and Statistical Filtering.*/

