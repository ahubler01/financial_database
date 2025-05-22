-- AAPL 1-minute data table
CREATE TABLE IF NOT EXISTS stock_aapl_1min (
    date TIMESTAMP PRIMARY KEY,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    adj_close DECIMAL(10,2),
    volume BIGINT
);

-- VIX data table
CREATE TABLE IF NOT EXISTS stock_vix (
    date DATE PRIMARY KEY,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2)
); 