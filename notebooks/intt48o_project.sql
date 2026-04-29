CREATE DATABASE IF NOT EXISTS draft;
USE draft;

-- we add all tables from csv files, than we add constraints for each table

-- USERS
ALTER TABLE users
MODIFY user_id INT PRIMARY KEY,
MODIFY full_name VARCHAR(255),
MODIFY gender VARCHAR(20),
MODIFY age INT,
MODIFY city VARCHAR(255),
MODIFY platform VARCHAR(20),
MODIFY signup_date DATE,
MODIFY last_active_date DATE;

-- PRODUCTS
ALTER TABLE products
MODIFY product_id INT PRIMARY KEY,
MODIFY product_name VARCHAR(255),
MODIFY product_category VARCHAR(255),
MODIFY product_price DECIMAL(10,2),
MODIFY product_stock INT;

-- ORDERS
ALTER TABLE orders
MODIFY order_id INT PRIMARY KEY,
MODIFY user_id INT,
MODIFY payment_method VARCHAR(100),
MODIFY order_date DATE,
MODIFY total_payment DECIMAL(10,2),
ADD FOREIGN KEY (user_id) REFERENCES users(user_id);

-- ORDERPRODUCT
ALTER TABLE orderproduct
MODIFY order_id INT,
MODIFY product_id INT,
MODIFY product_name VARCHAR(255),
MODIFY product_category VARCHAR(255),
MODIFY product_price DECIMAL(12,2),
MODIFY quantity INT,
MODIFY total_price DECIMAL(12,2),
ADD FOREIGN KEY (order_id) REFERENCES orders(order_id),
ADD FOREIGN KEY (product_id) REFERENCES products(product_id);

-- CAMPAIGNS
ALTER TABLE campaigns
MODIFY campaign_id INT PRIMARY KEY,
MODIFY campaign_name VARCHAR(255),
MODIFY objective VARCHAR(255),
MODIFY account_name VARCHAR(255),
MODIFY start_date DATE,
MODIFY end_date DATE,
MODIFY status VARCHAR(100),
MODIFY campaing_budget DECIMAL(10,2),
MODIFY campaing_revenue DECIMAL(10,2),
MODIFY total_click INT,
MODIFY total_lead INT,
MODIFY total_reach INT;

-- ADS
ALTER TABLE ads
MODIFY ad_id INT PRIMARY KEY,
MODIFY ad_name VARCHAR(255),
MODIFY ad_format VARCHAR(255),
MODIFY placement VARCHAR(255),
MODIFY platform VARCHAR(100),
MODIFY clicks INT,
MODIFY reach INT,
MODIFY purchase INT,
MODIFY created_at DATE,
MODIFY is_active BOOLEAN,
MODIFY campaign_id INT,
ADD FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id);

-- ADREACH
ALTER TABLE adreach
MODIFY ad_id INT,
MODIFY user_id INT,
MODIFY reach_time DATETIME,
ADD FOREIGN KEY (ad_id) REFERENCES ads(ad_id),
ADD FOREIGN KEY (user_id) REFERENCES users(user_id);


-- ADCLICK
ALTER TABLE adclick
MODIFY ad_id INT,
MODIFY user_id INT,
MODIFY click_time VARCHAR(50), -- We used VARCHAR here due to issues with datetime parsing during data import. Since this column is not used in any procedures or views, we ignored the formatting error.
ADD FOREIGN KEY (ad_id) REFERENCES ads(ad_id),
ADD FOREIGN KEY (user_id) REFERENCES users(user_id);

-- ADPURCHASE
ALTER TABLE adpurchase
MODIFY ad_id INT,
MODIFY user_id INT,
MODIFY purchase_time VARCHAR(50), -- We used VARCHAR here due to issues with datetime parsing during data import. Since this column is not used in any procedures or views, we ignored the formatting error.
ADD FOREIGN KEY (ad_id) REFERENCES ads(ad_id),
ADD FOREIGN KEY (user_id) REFERENCES users(user_id);

-- PRODUCTSADS
ALTER TABLE productsads
MODIFY ad_id INT,
MODIFY product_id INT,
MODIFY product_name VARCHAR(255),
MODIFY product_category VARCHAR(255),
MODIFY product_price DECIMAL(10,2),
ADD FOREIGN KEY (ad_id) REFERENCES ads(ad_id),
ADD FOREIGN KEY (product_id) REFERENCES products(product_id);

-- SESSIONS
ALTER TABLE sessions
MODIFY session_id INT PRIMARY KEY,
MODIFY user_id INT,
MODIFY session_start DATETIME,
MODIFY session_end DATETIME,
MODIFY source VARCHAR(255),
ADD FOREIGN KEY (user_id) REFERENCES users(user_id);

-- USERADINT
ALTER TABLE useradint
MODIFY interaction_id INT PRIMARY KEY,
MODIFY ad_id INT,
MODIFY user_id INT,
MODIFY ad_reach TINYINT,
MODIFY ad_click TINYINT,
MODIFY ad_purchase TINYINT,
MODIFY order_id INT,
MODIFY product_id INT,
ADD FOREIGN KEY (ad_id) REFERENCES ads(ad_id),
ADD FOREIGN KEY (user_id) REFERENCES users(user_id),
ADD FOREIGN KEY (order_id) REFERENCES orders(order_id),
ADD FOREIGN KEY (product_id) REFERENCES products(product_id);

ALTER TABLE ad_prices
MODIFY placement VARCHAR(255),
MODIFY ad_price DECIMAL(10,2),
ADD PRIMARY KEY (placement);


SELECT 
    TABLE_NAME, COLUMN_NAME, CONSTRAINT_NAME, 
    REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
FROM 
    information_schema.KEY_COLUMN_USAGE
WHERE 
    TABLE_SCHEMA = 'draft'
    AND REFERENCED_TABLE_NAME IS NOT NULL;

-- STORED PROCEDURES
-- Orders total payment

DELIMITER $$

CREATE PROCEDURE update_orders_total_payment()
BEGIN
    UPDATE Orders o
    JOIN (
        SELECT 
            order_id, 
            SUM(total_price) AS total_payment_sum
        FROM 
            OrderProduct
        GROUP BY 
            order_id
    ) op ON o.order_id = op.order_id
    SET o.total_payment = op.total_payment_sum;
END$$

DELIMITER ;

CALL update_orders_total_payment();
SELECT * FROM Orders;

-- Ads calculate ad budget

DELIMITER $$
CREATE PROCEDURE update_ads_budget()
BEGIN
    UPDATE Ads a
    JOIN Ad_prices ap ON a.placement = ap.placement
    SET a.ad_budget = a.reach * ap.ad_price;
END$$

CALL update_ads_budget();
SELECT * FROM Ads;

-- Campaigns calculate campaign budget

DELIMITER $$

CREATE PROCEDURE update_campaign_budgets()
BEGIN
    UPDATE Campaigns c
    JOIN (
        SELECT 
            campaign_id,
            SUM(ad_budget) AS total_ad_budget
        FROM Ads
        GROUP BY campaign_id
    ) a ON c.campaign_id = a.campaign_id
    SET c.campaign_budget = a.total_ad_budget;
END$$

DELIMITER ;

CALL update_campaign_budgets();
SELECT * FROM Campaigns;

-- Campaings total reach

DELIMITER $$

CREATE PROCEDURE update_campaign_total_reach()
BEGIN
    UPDATE Campaigns c
    JOIN (
        SELECT 
            campaign_id,
            SUM(reach) AS total_reach_sum
        FROM Ads
        GROUP BY campaign_id
    ) a ON c.campaign_id = a.campaign_id
    SET c.total_reach = a.total_reach_sum;
END$$

DELIMITER ;

CALL update_campaign_total_reach();

SELECT * FROM Campaigns;

-- Campaigns total click

DELIMITER $$

CREATE PROCEDURE update_campaign_total_click()
BEGIN
    UPDATE Campaigns c
    JOIN (
        SELECT 
            campaign_id,
            SUM(clicks) AS total_click_sum
        FROM Ads
        GROUP BY campaign_id
    ) a ON c.campaign_id = a.campaign_id
    SET c.total_click = a.total_click_sum;
END$$

DELIMITER ;

CALL update_campaign_total_click();
SELECT * FROM Campaigns;

-- Campaigns total lead

DELIMITER $$

CREATE PROCEDURE update_campaign_total_lead()
BEGIN
    UPDATE Campaigns c
    JOIN (
        SELECT 
            campaign_id,
            SUM(purchase) AS total_lead_sum
        FROM Ads
        GROUP BY campaign_id
    ) a ON c.campaign_id = a.campaign_id
    SET c.total_lead = a.total_lead_sum;
END$$

DELIMITER ;

CALL update_campaign_total_lead();
SELECT * FROM Campaigns;

-- Calculating revenue for campaigns
DELIMITER $$

CREATE PROCEDURE update_campaign_revenue()
BEGIN
    UPDATE Campaigns c
    JOIN (
        SELECT 
            a.campaign_id,
            SUM(op.total_price) AS campaign_revenue
        FROM Ads a
        JOIN UserAdInt uai ON a.ad_id = uai.ad_id
        JOIN OrderProduct op 
             ON uai.order_id = op.order_id AND uai.product_id = op.product_id
        WHERE uai.order_id IS NOT NULL AND uai.product_id IS NOT NULL
        GROUP BY a.campaign_id
    ) revenue_data ON c.campaign_id = revenue_data.campaign_id
    SET c.campaign_revenue = revenue_data.campaign_revenue;
END$$

DELIMITER ;

CALL update_campaign_revenue();
SELECT * FROM Campaigns;

-- VIEWS
-- Total orders and spending by user
CREATE VIEW vw_user_order_summary AS
SELECT 
    u.user_id,
    u.full_name,
    COUNT(o.order_id) AS total_orders,
    SUM(o.total_payment) AS total_spending
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.full_name;

SELECT * FROM vw_user_order_summary;

-- Aggregated ad KPIs
CREATE VIEW vw_ad_performance_summary AS
SELECT 
    a.ad_id,
    a.ad_name,
    a.ad_format,
    a.platform,
    a.placement,
    a.created_at,
    a.is_active,
    a.reach,
    a.clicks,
    a.purchase,
    a.ad_budget
FROM ads a;

SELECT * FROM vw_ad_performance_summary;

-- ROI calculation by campaign
CREATE VIEW vw_campaign_roi AS
SELECT 
    c.campaign_id,
    c.campaign_name,
    c.campaing_budget,
    c.campaing_revenue,
    ROUND(c.campaing_revenue / NULLIF(c.campaing_budget, 0), 2) AS roi
FROM campaigns c;

SELECT * FROM vw_campaign_roi;

-- Funnel stage per user
CREATE VIEW FunnelStagePerUser AS
SELECT 
    u.user_id,
    CASE WHEN ar.ad_id IS NOT NULL THEN 1 ELSE 0 END AS saw_ad,
    CASE WHEN ac.ad_id IS NOT NULL THEN 1 ELSE 0 END AS clicked_ad,
    CASE WHEN ap.ad_id IS NOT NULL THEN 1 ELSE 0 END AS purchased_ad
FROM Users u
LEFT JOIN AdReach ar ON u.user_id = ar.user_id
LEFT JOIN AdClick ac ON u.user_id = ac.user_id
LEFT JOIN AdPurchase ap ON u.user_id = ap.user_id
GROUP BY u.user_id;

SELECT * FROM FunnelStagePerUser;

-- Revenue by product category
CREATE VIEW RevenueByCategory AS
SELECT 
    p.product_category,
    SUM(op.total_price) AS total_revenue
FROM OrderProduct op
JOIN Products p ON op.product_id = p.product_id
GROUP BY p.product_category;

SELECT * FROM RevenueByCategory;

-- Conversion % by traffic source
CREATE VIEW ConversionBySource AS
SELECT 
    s.source,
    COUNT(DISTINCT u.user_id) AS total_users,
    COUNT(DISTINCT o.user_id) AS purchasers,
    ROUND(COUNT(DISTINCT o.user_id) / COUNT(DISTINCT u.user_id) * 100, 2) AS conversion_rate
FROM Sessions s
JOIN Users u ON s.user_id = u.user_id
LEFT JOIN Orders o ON u.user_id = o.user_id
GROUP BY s.source;

SELECT * FROM ConversionBySource;

-- Most efficient ads by revenue/budget
CREATE VIEW MostEfficientAds AS
SELECT 
    a.ad_id,
    a.ad_name,
    a.ad_budget,
    c.campaing_revenue,
    ROUND(c.campaing_revenue / a.ad_budget, 2) AS efficiency_ratio
FROM Ads a
JOIN Campaigns c ON a.campaign_id = c.campaign_id
WHERE a.ad_budget > 0
ORDER BY efficiency_ratio DESC;

SELECT * FROM MostEfficientAds;
