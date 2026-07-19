
-- RecoMart Feature Warehouse Schema

CREATE TABLE IF NOT EXISTS user_features (
    user_id INTEGER PRIMARY KEY,
    total_ratings INTEGER,
    avg_rating REAL,
    rating_std REAL,
    total_purchases INTEGER,
    total_spend REAL,
    avg_order_value REAL,
    total_clicks INTEGER,
    view_count INTEGER,
    cart_add_count INTEGER,
    purchase_frequency REAL,
    activity_score REAL,
    account_age_days INTEGER,
    is_premium INTEGER,
    gender_encoded INTEGER
);

CREATE TABLE IF NOT EXISTS item_features (
    product_id INTEGER PRIMARY KEY,
    total_ratings INTEGER,
    avg_user_rating REAL,
    rating_variance REAL,
    total_purchases INTEGER,
    total_revenue REAL,
    total_views INTEGER,
    view_to_purchase_ratio REAL,
    popularity_score REAL,
    price REAL,
    price_normalized REAL,
    category_encoded INTEGER,
    in_stock INTEGER
);

CREATE TABLE IF NOT EXISTS user_item_features (
    user_id INTEGER,
    product_id INTEGER,
    rating REAL,
    n_interactions INTEGER,
    has_purchased INTEGER,
    has_viewed INTEGER,
    has_carted INTEGER,
    PRIMARY KEY (user_id, product_id)
);
