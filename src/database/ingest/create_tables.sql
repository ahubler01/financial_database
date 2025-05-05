-- Drop tables if they exist
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS subreddit_subscribers;

-- Create posts table
CREATE TABLE posts (
    id text,
    created_utc bigint,  -- Unix timestamp format
    subreddit text,
    author text,
    title text,
    selftext text,
    permalink text,
    url text
);

-- Create subreddit_subscribers table
CREATE TABLE subreddit_subscribers (
    subreddit text,
    subscribers bigint,  -- Changed to bigint for large numbers
    active_users bigint  -- Changed to bigint for large numbers
); 