-- Reddit posts table
CREATE TABLE IF NOT EXISTS reddit_posts (
    id VARCHAR(255) PRIMARY KEY,
    created_utc TIMESTAMP,
    subreddit VARCHAR(255),
    author VARCHAR(255),
    title TEXT,
    selftext TEXT,
    permalink TEXT,
    url TEXT
);

-- Stock index table
CREATE TABLE IF NOT EXISTS reddit_stock_index (
    id VARCHAR(255),
    stock_symbol VARCHAR(10),
    created_utc TIMESTAMP,
    PRIMARY KEY (id, stock_symbol)
);

-- Subreddit subscribers table
CREATE TABLE IF NOT EXISTS reddit_subreddit_subscribers (
    subreddit VARCHAR(255) PRIMARY KEY,
    subscribers INTEGER
); 