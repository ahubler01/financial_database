-- Composite Business Confidence Index
CREATE TABLE IF NOT EXISTS macro_composite_business_confidence (
    observation_date DATE PRIMARY KEY,
    confidence_index DECIMAL(10,4)  -- BSCICP03USM665S
);

-- Effective Federal Funds Rate
CREATE TABLE IF NOT EXISTS macro_effective_rates (
    observation_date DATE PRIMARY KEY,
    fed_funds_rate DECIMAL(10,4)  -- FEDFUNDS
);

-- Insured Unemployment Rate
CREATE TABLE IF NOT EXISTS macro_insured_unemployment (
    observation_date DATE PRIMARY KEY,
    unemployment_claims DECIMAL(10,4)  -- CCSA
);

-- M1 Money Stock
CREATE TABLE IF NOT EXISTS macro_m1 (
    observation_date DATE PRIMARY KEY,
    m1_stock DECIMAL(20,2)  -- WM1NS
);

-- M2 Money Stock
CREATE TABLE IF NOT EXISTS macro_m2 (
    observation_date DATE PRIMARY KEY,
    m2_stock DECIMAL(20,2)  -- WM2NS
);

-- Net Exports
CREATE TABLE IF NOT EXISTS macro_net_exports (
    observation_date DATE PRIMARY KEY,
    net_exports DECIMAL(20,2)  -- NETEXP
);

-- Sticky CPI
CREATE TABLE IF NOT EXISTS macro_sticky_cpi (
    observation_date DATE PRIMARY KEY,
    core_sticky_cpi DECIMAL(10,6)  -- CORESTICKM159SFRBATL
); 