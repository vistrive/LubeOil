-- Initialize LOBP Database
-- This script sets up basic database configuration

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create basic indexes for performance
-- Additional tables will be created by Alembic migrations

-- Set default timezone
SET timezone = 'UTC';

-- Log successful initialization
SELECT 'Database initialization completed' as status;