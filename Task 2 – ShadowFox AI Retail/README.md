# ShadowFox AI Retail Intelligence Platform

AI-powered sales, profit, forecasting, anomaly detection, and executive reporting platform for retail analytics.

## Highlights

- Executive KPI dashboard with premium dark UI
- Sales, profit, customer, and product intelligence
- AI-style business insight and recommendation engine
- Forecasting with trend and seasonality signals
- Anomaly detection for unusual revenue and profit behavior
- Natural-language business chat for common analysis questions
- Excel and PDF executive report exports
- Role-based demo access: Admin, Analyst, Viewer

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo Login

Use any email/password in the sidebar login form and choose a role. This portfolio build keeps authentication local to Streamlit session state so recruiters can explore quickly.

## Data

The app automatically generates a realistic sample retail dataset. You can also upload a CSV with columns such as:

- `order_date`
- `region`
- `segment`
- `category`
- `subcategory`
- `product`
- `customer`
- `sales`
- `profit`
- `discount`
- `quantity`

Column names are normalized automatically.
