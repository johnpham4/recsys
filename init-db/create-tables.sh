#!/bin/bash
set -e

echo "Creating tables for AGENTIC and RECOMMENDER systems..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL

    -- Session Header
    CREATE TABLE IF NOT EXISTS ${AGENTIC_SYSTEM_SCHEMA:-agentic_system}.sessions (
        thread_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_user ON ${AGENTIC_SYSTEM_SCHEMA:-agentic_system}.sessions(user_id);

    -- Session events
    CREATE TABLE IF NOT EXISTS ${AGENTIC_SYSTEM_SCHEMA:-agentic_system}.session_events (
        event_id SERIAL PRIMARY KEY,
        thread_id TEXT NOT NULL,
        turn_id TEXT NOT NULL,
        step_index INT NOT NULL DEFAULT 0,
        event_type VARCHAR(50) NOT NULL, 
        payload JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_events_thread ON ${AGENTIC_SYSTEM_SCHEMA:-agentic_system}.session_events(thread_id, turn_id);

    -- User cart
    CREATE TABLE IF NOT EXISTS ${AGENTIC_SYSTEM_SCHEMA:-agentic_system}.user_cart (
        cart_id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        asin VARCHAR(50) NOT NULL,
        quantity INT NOT NULL DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, asin)
    );

    -- D. Bảng Tinh chỉnh Sở thích (Phục vụ Action: record_user_preference / Critiquing)
    CREATE TABLE IF NOT EXISTS ${AGENTIC_SYSTEM_SCHEMA:-agentic_system}.user_preferences (
        pref_id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        pref_key VARCHAR(100) NOT NULL,   -- e.g., 'brand', 'color', 'ergonomic'
        pref_value TEXT NOT NULL,          -- e.g., 'Razer', 'pink', 'wireless'
        sentiment VARCHAR(20) NOT NULL,    -- 'positive' (thích) hoặc 'negative' (ghét / loại trừ)
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );


    -- Bảng lưu lịch sử gợi ý phục vụ đánh giá A/B Testing hoặc offline analytics
    CREATE TABLE IF NOT EXISTS ${RECOMMENDER_SYSTEM_SCHEMA:-recommender_system}.recommendation_logs (
        log_id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        model_version VARCHAR(50),
        recommended_asins JSONB NOT NULL,
        context JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

EOSQL

echo "Tables created successfully!"
