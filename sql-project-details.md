# E-Commerce Marketing Analytics Dashboard

This project focuses on designing a full-stack marketing analytics system for a simulated e-commerce platform. We combined synthetic data generation, relational database development, SQL automation, and business intelligence reporting through Power BI.

---

## Project Overview

We built this system to simulate the type of marketing analytics environment typically used by e-commerce companies to monitor performance, customer behavior, and advertising effectiveness. Instead of using pre-existing datasets, we generated a fully synthetic dataset using Python, designed to reflect real-world e-commerce sales structures, marketing funnels, and advertising campaigns.

---

## Dataset Design

A total of **13 interrelated tables** were designed, representing key marketing and sales activities, including:

- Users (demographics, signup dates, activity history)
- Products (catalog, pricing, inventory)
- Orders & Order Details (transactional data)
- Digital Campaigns & Ads (marketing structure)
- Ad Interactions (reach, click, conversion stages)
- Sessions (traffic source and user sessions)
- Ad Pricing (ad placement costs)

We carefully created the dataset from scratch, ensuring realistic patterns such as proportional ad reach, conversion probabilities, purchase behaviors, and budget dynamics.

---

## Database Development

The dataset was imported into **MySQL**, where we fully implemented the relational schema and referential integrity using **foreign key constraints**.

To automate calculations and ensure data consistency, we developed **stored procedures** to compute:

- Total payments per order
- Ad budgets (based on reach and placement pricing)
- Campaign budgets, clicks, leads, revenues
- Funnel interaction metrics

Additionally, we created multiple **views** to simplify reporting:

- User order summaries
- Ad performance metrics
- Campaign ROI calculations
- Funnel stage indicators
- Revenue by product category
- Conversion rates by traffic source
- Ad efficiency rankings

---

## Business Intelligence Dashboard (Power BI)

Using the finalized SQL database, we built an interactive Power BI dashboard consisting of 6 pages:

1. **User Insights**: Demographics, platform usage, city distributions, active/new users.
2. **Products**: Product category distribution, stock levels, pricing, low stock alerts.
3. **Orders**: Order trends, payment methods, revenue by city, top products.
4. **Campaigns**: Performance metrics, conversion rates, profitability, CTR comparisons.
5. **Ad Structure**: Ad formats, platform distribution, budget allocation, active status.
6. **Ad Performance**: Funnel metrics, click-to-lead ratios, ROAS, platform-level CTR.

The dashboard enables real-time analysis of user behavior, sales performance, campaign efficiency, and advertising effectiveness.

---

## Sample ERD Diagram

![ERD Diagram](images/ERD.jpeg)

---

## Sample Dashboard Visualization

![Power BI Screenshot](images/orders.png)

---

## Technical Stack

- **Data Generation**: Python
- **Database Development**: MySQL (DDL, Stored Procedures, Views)
- **Business Intelligence**: Power BI
- **Languages**: SQL, DAX, Python

---

## Key Takeaways

This project demonstrates the full pipeline from dataset creation to database development and business intelligence reporting for a realistic marketing use case. By combining automated SQL calculations and Power BI visualizations, we developed a scalable environment to analyze marketing KPIs and optimize business decisions.

---

[View Full Dataset & SQL Code](link_to_sql_repo_or_folder)  
