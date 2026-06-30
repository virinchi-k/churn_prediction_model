-- ===========================================================================
-- CUSTOMER CHURN ANALYSIS - SQL DEEP DIVE
-- Author: Virinchi Kopparam
-- 
-- What this file does: takes a 10,000 row bank customer dataset and answers
-- the questions a retention team would actually ask, not just "who churned?"
-- My goal was to think like the analyst who has to present this to leadership
-- on Monday morning, not just write queries that technically run.
-- ===========================================================================

-- ===========================================================================
-- ASSUMPTIONS (so nobody has to guess where my numbers came from)
-- 1. $243 acquisition cost per customer - this is a commonly cited industry
--    average for retail banking customer acquisition. Used here to turn
--    churn counts into dollars leadership actually cares about.
-- 2. 20% churn reduction target in Q9 - a conservative, achievable target
--    for a first-pass retention campaign. Not tied to the ML model's actual
--    lift yet, that gets validated once the model is in production.
-- 3. Risk score weights in Q3 (30/20/25/15/10) - inactivity gets the highest
--    weight because an inactive member with no recent engagement is the
--    single strongest churn signal in this dataset based on the EDA below.
--    Geography gets the lowest weight since it's a softer, indirect signal
--    compared to actual behavior.
-- ===========================================================================


-- ===========================================================================
-- TABLE SETUP:
-- Heads up: I loaded the data using the MySQL Workbench Import Wizard
-- instead of a manual BULK INSERT, so the schema below is jsut for reference,
-- not something you need to run.
-- ===========================================================================

-- DROP TABLE IF EXISTS churndata;

-- CREATE TABLE ChurnData (
--     RowNumber       INT,
--     CustomerId      INT,
--     Surname         VARCHAR(25),
--     CreditScore     INT,
--     Geography       VARCHAR(10),
--     Gender          VARCHAR(10),
--     Age             INT,
--     Tenure          INT,
--     Balance         DOUBLE,
--     NumOfProducts   INT,
--     HasCrCard       INT,
--     IsActiveMember  INT,
--     EstimatedSalary DOUBLE,
--     Exited          INT
-- );


-- ===========================================================================
-- DATA VALIDATION (trust nothing until you've checked it yourself)
-- ===========================================================================

-- EDA 1: Row count and churn split
SELECT
    COUNT(*)                        AS TotalRecords,
    SUM(Exited)                     AS TotalChurned,
    COUNT(*) - SUM(Exited)          AS TotalRetained,
    ROUND(AVG(Exited * 100.0), 2)   AS OverallChurnRate_Pct
FROM ChurnData;

-- FINDING: TotalChurned = 2037;	TotalRetained = 7963;	OverallChurnRate_Pct = 20.37 
-- Run this first. If your churn rate isn't somewhere around 20%,
-- something went wrong in the import and the rest of this file is irrelevant.


-- EDA 2: Null check across every column
SELECT
    SUM(CASE WHEN CustomerId      IS NULL THEN 1 ELSE 0 END) AS Null_CustomerId,
    SUM(CASE WHEN CreditScore     IS NULL THEN 1 ELSE 0 END) AS Null_CreditScore,
    SUM(CASE WHEN Geography       IS NULL THEN 1 ELSE 0 END) AS Null_Geography,
    SUM(CASE WHEN Gender          IS NULL THEN 1 ELSE 0 END) AS Null_Gender,
    SUM(CASE WHEN Age             IS NULL THEN 1 ELSE 0 END) AS Null_Age,
    SUM(CASE WHEN Tenure          IS NULL THEN 1 ELSE 0 END) AS Null_Tenure,
    SUM(CASE WHEN Balance         IS NULL THEN 1 ELSE 0 END) AS Null_Balance,
    SUM(CASE WHEN NumOfProducts   IS NULL THEN 1 ELSE 0 END) AS Null_NumOfProducts,
    SUM(CASE WHEN EstimatedSalary IS NULL THEN 1 ELSE 0 END) AS Null_EstimatedSalary,
    SUM(CASE WHEN Exited          IS NULL THEN 1 ELSE 0 END) AS Null_Exited
FROM ChurnData;
-- FINDING: Crystal clean dataset, zero nulls across the board. 
-- Kaggle datasets are rarely this polite, so I'm not complaining


-- EDA 3: Duplicate check on CustomerId
SELECT
    CustomerId,
    COUNT(*) AS Occurrences
FROM ChurnData
GROUP BY CustomerId
HAVING COUNT(*) > 1;
-- FINDING: No rows returned means no duplicate customers. Good, because
-- duplicate customers would have quietly inflated every churn rate below.


-- EDA 4: Quick profile of the average customer
SELECT
    ROUND(AVG(Age), 1)                 AS AvgAge,
    ROUND(AVG(CreditScore), 1)         AS AvgCreditScore,
    ROUND(AVG(Balance), 2)             AS AvgBalance,
    ROUND(AVG(EstimatedSalary), 2)     AS AvgSalary,
    ROUND(AVG(Tenure), 1)              AS AvgTenure,
    ROUND(AVG(NumOfProducts), 2)       AS AvgProducts,
    ROUND(AVG(IsActiveMember*100.0),2) AS ActiveMemberRate_Pct,
    COUNT(DISTINCT Geography)          AS GeographiesCount,
    MIN(CreditScore)                   AS MinCreditScore,
    MAX(CreditScore)                   AS MaxCreditScore
FROM ChurnData;
-- FINDING: This is your baseline. Every segment-level number in the
-- queries below should be compared back against these averages.


select * from churndata limit 5;

-- Checking column lengths before modifying anything, learned this the
-- hard way once on a different project after truncating half a column
SELECT MAX(LENGTH(surname)), MAX(LENGTH(geography)), MAX(LENGTH(gender)) from churndata;
ALTER TABLE churndata
MODIFY Surname VARCHAR (25),
MODIFY  Geography VARCHAR (10),
MODIFY  Gender VARCHAR (10);


-- ===========================================================================
-- NOW THE FUN PART: BUSINESS QUESTIONS
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Q1: WHO CHURNS MORE, GOOD CREDIT OR BAD CREDIT CUSTOMERS?
-- Why I'm asking: if churn tracks credit risk, that changes who gets
-- prioritized for retention outreach versus who gets prioritized for
-- collections. Two very different teams, two very different budgets.
-- ---------------------------------------------------------------------------

WITH CreditTiers AS (
    SELECT
        CustomerId,
        Exited,
        CASE
            WHEN CreditScore BETWEEN 0   AND 580 THEN '1_Poor (0-580)'
            WHEN CreditScore BETWEEN 581 AND 669 THEN '2_Fair (581-669)'
            WHEN CreditScore BETWEEN 670 AND 739 THEN '3_Good (670-739)'
            WHEN CreditScore >= 740              THEN '4_Very Good (740+)'
        END AS CreditTier
    FROM ChurnData
)
SELECT
    CreditTier,
    COUNT(*)                                    AS TotalCustomers,
    SUM(Exited)                                 AS ChurnedCustomers,
    COUNT(*) - SUM(Exited)                      AS RetainedCustomers,
    ROUND(AVG(Exited * 100.0), 2)               AS ChurnRate_Pct
FROM CreditTiers
GROUP BY CreditTier
ORDER BY CreditTier;
-- FINDING: [fill in once you run it] e.g. "Churn rate barely moves across
-- credit tiers, so credit score on its own is a weak churn signal. Good
-- news for collections, not very useful for retention targeting."


-- ---------------------------------------------------------------------------
-- Q2: WHICH YEAR OF THE CUSTOMER RELATIONSHIP IS THE DANGER ZONE?
-- Why I'm asking: throwing retention budget at every tenure year equally
-- is wasteful. I want to know exactly which year we're losing people so
-- intervention timing actually lines up with the risk.
-- ---------------------------------------------------------------------------

WITH TenureCohorts AS (
    SELECT
        Tenure,
        COUNT(*)                        AS TotalCustomers,
        SUM(Exited)                     AS ChurnedCustomers,
        ROUND(AVG(Exited * 100.0), 2)   AS ChurnRate_Pct
    FROM ChurnData
    GROUP BY Tenure
),
WithRunning AS (
    SELECT
        Tenure,
        TotalCustomers,
        ChurnedCustomers,
        ChurnRate_Pct,
        SUM(ChurnedCustomers) OVER (ORDER BY Tenure ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS CumulativeChurned,
        SUM(TotalCustomers)   OVER (ORDER BY Tenure ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            AS CumulativeTotal
    FROM TenureCohorts
)
SELECT
    Tenure,
    TotalCustomers,
    ChurnedCustomers,
    ChurnRate_Pct,
    CumulativeChurned,
    ROUND(CumulativeChurned * 100.0 / CumulativeTotal, 2) AS CumulativeChurnRate_Pct
FROM WithRunning
ORDER BY Tenure;
-- FINDING: [fill in] e.g. "Churn rate doesn't spike at year 1 like I
-- expected, it's flat across the whole tenure curve. Tells me this isn't
-- an onboarding problem, it's a persistent satisfaction issue."


-- ---------------------------------------------------------------------------
-- Q3: BUILDING A "WHO'S ABOUT TO LEAVE" SCORE BY HAND
-- Why I'm asking: before the ML model exists, an analyst still needs to
-- hand the retention team a ranked list on Monday. This rules-based score
-- is exactly that, a transparent, explainable stand-in the model formalizes
-- later. Five behavioral flags, weighted by how strong each signal is.
-- ---------------------------------------------------------------------------

WITH RiskFactors AS (
    SELECT
        CustomerId,
        Surname,
        Geography,
        Age,
        Balance,
        NumOfProducts,
        IsActiveMember,
        Exited,
        CASE WHEN IsActiveMember = 0    THEN 30 ELSE 0 END AS InactiveScore,
        CASE WHEN NumOfProducts  = 1    THEN 20 ELSE 0 END AS SingleProductScore,
        CASE WHEN Age BETWEEN 45 AND 60 THEN 25 ELSE 0 END AS AgeBandScore,
        CASE WHEN Balance = 0           THEN 15 ELSE 0 END AS ZeroBalanceScore,
        CASE WHEN Geography = 'Germany' THEN 10 ELSE 0 END AS GeoRiskScore
    FROM ChurnData
),
Scored AS (
    SELECT
        *,
        InactiveScore + SingleProductScore + AgeBandScore + ZeroBalanceScore + GeoRiskScore
            AS CompositeRiskScore
    FROM RiskFactors
),
Ranked AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY CompositeRiskScore DESC) AS RiskQuartile
    FROM Scored
)
SELECT
    CustomerId,
    Surname,
    Geography,
    Age,
    Balance,
    CompositeRiskScore,
    CASE RiskQuartile
        WHEN 1 THEN 'Critical'
        WHEN 2 THEN 'High'
        WHEN 3 THEN 'Medium'
        WHEN 4 THEN 'Low'
    END AS RiskBand,
    Exited
FROM Ranked
ORDER BY CompositeRiskScore DESC;
-- FINDING: [fill in] This is the list. Whoever's in the "Critical" band
-- at the top is who the retention team calls first.


-- ---------------------------------------------------------------------------
-- Q4: OKAY BUT DOES MY SCORE FROM Q3 ACTUALLY WORK?
-- Why I'm asking: building a score is easy, anyone can assign random
-- weights and call it done. Proving the score actually separates real
-- churners from non-churners is the part most people skip. I'm not skipping it.
-- ---------------------------------------------------------------------------

WITH RiskFactors AS (
    SELECT
        CustomerId,
        Exited,
        CASE WHEN IsActiveMember = 0    THEN 30 ELSE 0 END +
        CASE WHEN NumOfProducts  = 1    THEN 20 ELSE 0 END +
        CASE WHEN Age BETWEEN 45 AND 60 THEN 25 ELSE 0 END +
        CASE WHEN Balance = 0           THEN 15 ELSE 0 END +
        CASE WHEN Geography = 'Germany' THEN 10 ELSE 0 END
            AS CompositeRiskScore
    FROM ChurnData
),
Banded AS (
    SELECT
        Exited,
        NTILE(4) OVER (ORDER BY CompositeRiskScore DESC) AS RiskQuartile
    FROM RiskFactors
),
Baseline AS (
    SELECT ROUND(AVG(Exited * 100.0), 2) AS OverallChurnRate FROM ChurnData
)
SELECT
    CASE RiskQuartile
        WHEN 1 THEN 'Critical'
        WHEN 2 THEN 'High'
        WHEN 3 THEN 'Medium'
        WHEN 4 THEN 'Low'
    END                                                 AS RiskBand,
    COUNT(*)                                            AS Customers,
    SUM(Exited)                                         AS Churned,
    ROUND(AVG(Exited * 100.0), 2)                       AS ChurnRate_Pct,
    b.OverallChurnRate,
    ROUND(AVG(Exited * 100.0) - b.OverallChurnRate, 2)  AS LiftVsBaseline_Pct
FROM Banded
CROSS JOIN Baseline b
GROUP BY RiskQuartile, b.OverallChurnRate
ORDER BY RiskQuartile;
-- FINDING: [fill in] If the "Critical" band's churn rate is meaningfully
-- higher than the overall baseline, the score earns its keep. If it's
-- basically flat across all four bands, back to the drawing board on weights.


-- ---------------------------------------------------------------------------
-- Q5: THE CUSTOMERS WE REALLY CANNOT AFFORD TO LOSE
-- Why I'm asking: not all churn costs the same. Losing a customer with
-- $200K sitting in their account hurts a lot more than losing one with
-- $20K. This finds the high-balance, disengaged customers before they walk.
-- ---------------------------------------------------------------------------

WITH BalancePercentiles AS (
    SELECT
        CustomerId,
        Surname,
        Geography,
        Gender,
        Age,
        Balance,
        NumOfProducts,
        IsActiveMember,
        Exited,
        NTILE(4) OVER (ORDER BY Balance DESC) AS BalanceQuartile,
        ROUND(AVG(Balance) OVER (PARTITION BY Geography), 2) AS AvgBalanceByGeo
    FROM ChurnData
    WHERE Balance > 0
)
SELECT
    CustomerId,
    Surname,
    Geography,
    Age,
    Balance,
    AvgBalanceByGeo,
    ROUND((Balance - AvgBalanceByGeo) / AvgBalanceByGeo * 100, 1) AS PctAboveGeoAvg,
    NumOfProducts,
    IsActiveMember,
    Exited
FROM BalancePercentiles
WHERE BalanceQuartile = 1
  AND IsActiveMember  = 0
ORDER BY Balance DESC;
-- FINDING: [fill in] This list should go straight to a relationship
-- manager, not a generic retention email blast.


-- ---------------------------------------------------------------------------
-- Q6: WHO LOOKS LIKE A MULTI-PRODUCT CUSTOMER BUT ISN'T ONE YET?
-- Why I'm asking: cross-sell and retention aren't separate problems here.
-- A single-product customer who matches the profile of a multi-product
-- customer in every other way is exactly who a cross-sell campaign
-- should target first, and it happens to reduce their churn risk too.
-- ---------------------------------------------------------------------------

WITH ProductGroups AS (
    SELECT
        NumOfProducts,
        Geography,
        Gender,
        CASE
            WHEN Age < 30             THEN 'Under 30'
            WHEN Age BETWEEN 30 AND 44 THEN '30-44'
            WHEN Age BETWEEN 45 AND 60 THEN '45-60'
            ELSE '60+'
        END                                     AS AgeGroup,
        COUNT(*)                                AS Customers,
        ROUND(AVG(Balance), 2)                  AS AvgBalance,
        ROUND(AVG(EstimatedSalary), 2)          AS AvgSalary,
        ROUND(AVG(Exited * 100.0), 2)           AS ChurnRate_Pct,
        ROUND(AVG(IsActiveMember * 100.0), 2)   AS ActiveRate_Pct
    FROM ChurnData
    GROUP BY NumOfProducts, Geography, Gender,
        CASE
            WHEN Age < 30             THEN 'Under 30'
            WHEN Age BETWEEN 30 AND 44 THEN '30-44'
            WHEN Age BETWEEN 45 AND 60 THEN '45-60'
            ELSE '60+'
        END
)
SELECT
    NumOfProducts,
    Geography,
    Gender,
    AgeGroup,
    Customers,
    AvgBalance,
    AvgSalary,
    ChurnRate_Pct,
    ActiveRate_Pct
FROM ProductGroups
WHERE AvgSalary > 100000
ORDER BY NumOfProducts, ChurnRate_Pct DESC;
-- FINDING: [fill in] Look for single-product segments with high churn
-- and high salary. That combo is the cross-sell sweet spot.


-- ---------------------------------------------------------------------------
-- Q7: IS THIS A REGION PROBLEM, A GENDER PROBLEM, OR BOTH?
-- Why I'm asking: it matters a lot whether churn is a regional issue
-- (maybe a service gap in one market) or a demographic one. Mixing the
-- two up means fixing the wrong thing.
-- ---------------------------------------------------------------------------

WITH GeoDemographics AS (
    SELECT
        Geography,
        Gender,
        COUNT(*)                                AS TotalCustomers,
        SUM(Exited)                             AS Churned,
        ROUND(AVG(Exited * 100.0), 2)           AS ChurnRate_Pct,
        ROUND(AVG(Age), 1)                      AS AvgAge,
        ROUND(AVG(Balance), 2)                  AS AvgBalance,
        ROUND(AVG(CreditScore * 1.0), 1)        AS AvgCreditScore
    FROM ChurnData
    GROUP BY Geography, Gender
),
WithRank AS (
    SELECT
        *,
        RANK() OVER (ORDER BY ChurnRate_Pct DESC) AS ChurnRank
    FROM GeoDemographics
)
SELECT
    ChurnRank,
    Geography,
    Gender,
    TotalCustomers,
    Churned,
    ChurnRate_Pct,
    AvgAge,
    AvgBalance,
    AvgCreditScore
FROM WithRank
ORDER BY ChurnRank;
-- FINDING: [fill in] e.g. "Germany shows up at the top regardless of
-- gender, so this looks like a market issue, not a demographic one."


-- ---------------------------------------------------------------------------
-- Q8: FINDING THE CUSTOMERS WHO DON'T FIT THE PATTERN
-- Why I'm asking: every geography has its "normal" balance range. The
-- customers sitting way outside that range, in either direction, are
-- worth a second look. Could be a VIP about to leave, could be a data error.
-- ---------------------------------------------------------------------------

WITH GeoStats AS (
    SELECT
        CustomerId,
        Surname,
        Geography,
        Balance,
        Exited,
        IsActiveMember,
        AVG(Balance)    OVER (PARTITION BY Geography) AS GeoAvgBalance,
        STDDEV(Balance) OVER (PARTITION BY Geography) AS GeoStdDevBalance
    FROM ChurnData
)
SELECT
    CustomerId,
    Surname,
    Geography,
    Balance,
    ROUND(GeoAvgBalance, 2)                                     AS GeoAvgBalance,
    ROUND(GeoStdDevBalance, 2)                                  AS GeoStdDev,
    ROUND((Balance - GeoAvgBalance) / GeoStdDevBalance, 2)      AS ZScore,
    IsActiveMember,
    Exited
FROM GeoStats
WHERE Balance > GeoAvgBalance + (2 * GeoStdDevBalance)
ORDER BY ZScore DESC;
-- FINDING: [fill in] These customers are the outliers worth a manual
-- review before they show up in next quarter's churn report.


-- ---------------------------------------------------------------------------
-- Q9: TURNING ALL OF THIS INTO AN ACTUAL DOLLAR FIGURE
-- Why I'm asking: nobody in a leadership meeting cares about a churn
-- rate percentage on its own. They care what it costs and what fixing
-- it is worth. This is the slide that gets budget approved.
-- ---------------------------------------------------------------------------

WITH ChurnMetrics AS (
    SELECT
        Geography,
        COUNT(*)                                      AS TotalCustomers,
        SUM(Exited)                                   AS ChurnedCustomers,
        ROUND(AVG(Exited * 100.0), 2)                 AS ChurnRate_Pct,
        ROUND(AVG(Balance), 2)                        AS AvgBalance,
        ROUND(SUM(Exited * Balance), 2)               AS BalanceAtRisk
    FROM ChurnData
    GROUP BY Geography
)
SELECT
    Geography,
    TotalCustomers,
    ChurnedCustomers,
    ChurnRate_Pct,
    AvgBalance,
    BalanceAtRisk,
    ChurnedCustomers * 243                              AS AcquisitionCostAtRisk_USD,
    ROUND(ChurnedCustomers * 0.20, 0)                   AS CustomersRetainedWith20PctReduction,
    ROUND(ChurnedCustomers * 0.20 * 243, 2)             AS SavingsFromReduction_USD
FROM ChurnMetrics
ORDER BY ChurnRate_Pct DESC;
