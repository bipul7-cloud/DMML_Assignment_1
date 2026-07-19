"""
RecoMart - Synthetic Data Generator
Generates realistic e-commerce data for the recommendation pipeline:
- Users, Products, Clickstream logs, Transactions, Ratings
"""

import os
import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from faker import Faker
from loguru import logger

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
LOG_DIR = BASE_DIR / "logs"


def setup_directories():
    """Create necessary directory structure."""
    dirs = [
        RAW_DIR / "users",
        RAW_DIR / "products",
        RAW_DIR / "clickstream",
        RAW_DIR / "transactions",
        RAW_DIR / "ratings",
        LOG_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    logger.info("Directory structure created.")


def generate_users(n_users=1000):
    """Generate synthetic user data."""
    logger.info(f"Generating {n_users} users...")
    users = []
    for uid in range(1, n_users + 1):
        user = {
            "user_id": uid,
            "username": fake.user_name(),
            "email": fake.email(),
            "age": random.randint(18, 70),
            "gender": random.choice(["M", "F", "Other"]),
            "country": fake.country_code(),
            "signup_date": fake.date_between(
                start_date="-3y", end_date="today"
            ).isoformat(),
            "is_premium": random.choice([True, False]),
        }
        users.append(user)

    # Inject some missing values for realism
    for _ in range(int(n_users * 0.05)):
        idx = random.randint(0, n_users - 1)
        users[idx]["age"] = ""
    for _ in range(int(n_users * 0.03)):
        idx = random.randint(0, n_users - 1)
        users[idx]["gender"] = ""

    ts = datetime.now().strftime("%Y%m%d")
    filepath = RAW_DIR / "users" / f"users_{ts}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=users[0].keys())
        writer.writeheader()
        writer.writerows(users)

    logger.info(f"Users saved to {filepath}")
    return users


def generate_products(n_products=500):
    """Generate synthetic product catalog."""
    logger.info(f"Generating {n_products} products...")
    categories = [
        "Electronics", "Clothing", "Books", "Home & Kitchen",
        "Sports", "Beauty", "Toys", "Automotive", "Grocery", "Health"
    ]
    products = []
    for pid in range(1, n_products + 1):
        category = random.choice(categories)
        product = {
            "product_id": pid,
            "name": fake.catch_phrase(),
            "category": category,
            "subcategory": f"{category}_{random.randint(1, 5)}",
            "price": round(random.uniform(5.0, 500.0), 2),
            "brand": fake.company(),
            "avg_rating": round(random.uniform(1.0, 5.0), 1),
            "n_reviews": random.randint(0, 5000),
            "in_stock": random.choice([True, True, True, False]),
            "description": fake.sentence(nb_words=12),
        }
        products.append(product)

    # Save as CSV
    ts = datetime.now().strftime("%Y%m%d")
    filepath = RAW_DIR / "products" / f"products_{ts}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)

    # Also save as JSON (simulating API response)
    json_path = RAW_DIR / "products" / f"products_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, default=str)

    logger.info(f"Products saved to {filepath} and {json_path}")
    return products


def generate_clickstream(users, products, n_events=50000):
    """Generate synthetic clickstream/interaction logs."""
    logger.info(f"Generating {n_events} clickstream events...")
    event_types = ["view", "click", "add_to_cart", "wishlist", "search"]
    devices = ["desktop", "mobile", "tablet"]
    events = []

    for _ in range(n_events):
        user = random.choice(users)
        product = random.choice(products)
        event = {
            "event_id": fake.uuid4(),
            "user_id": user["user_id"],
            "product_id": product["product_id"],
            "event_type": random.choices(
                event_types, weights=[40, 25, 15, 10, 10], k=1
            )[0],
            "timestamp": fake.date_time_between(
                start_date="-90d", end_date="now"
            ).isoformat(),
            "session_id": fake.uuid4()[:8],
            "device": random.choice(devices),
            "page_url": f"/product/{product['product_id']}",
        }
        events.append(event)

    # Inject duplicates for testing validation
    for _ in range(int(n_events * 0.02)):
        events.append(random.choice(events).copy())

    random.shuffle(events)

    ts = datetime.now().strftime("%Y%m%d")
    filepath = RAW_DIR / "clickstream" / f"clickstream_{ts}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)

    logger.info(f"Clickstream saved to {filepath}")
    return events


def generate_transactions(users, products, n_transactions=10000):
    """Generate synthetic purchase transactions."""
    logger.info(f"Generating {n_transactions} transactions...")
    payment_methods = ["credit_card", "debit_card", "upi", "wallet", "cod"]
    transactions = []

    for tid in range(1, n_transactions + 1):
        user = random.choice(users)
        product = random.choice(products)
        quantity = random.randint(1, 5)
        price = float(product["price"]) if product["price"] else 0
        txn = {
            "transaction_id": tid,
            "user_id": user["user_id"],
            "product_id": product["product_id"],
            "quantity": quantity,
            "unit_price": price,
            "total_amount": round(price * quantity, 2),
            "payment_method": random.choice(payment_methods),
            "transaction_date": fake.date_time_between(
                start_date="-180d", end_date="now"
            ).isoformat(),
            "status": random.choices(
                ["completed", "pending", "cancelled", "refunded"],
                weights=[80, 10, 5, 5], k=1
            )[0],
        }
        transactions.append(txn)

    # Inject missing values
    for _ in range(int(n_transactions * 0.02)):
        idx = random.randint(0, n_transactions - 1)
        transactions[idx]["total_amount"] = ""

    ts = datetime.now().strftime("%Y%m%d")
    filepath = RAW_DIR / "transactions" / f"transactions_{ts}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=transactions[0].keys())
        writer.writeheader()
        writer.writerows(transactions)

    logger.info(f"Transactions saved to {filepath}")
    return transactions


def generate_ratings(users, products, n_ratings=20000):
    """Generate synthetic user-item ratings."""
    logger.info(f"Generating {n_ratings} ratings...")
    ratings = []
    seen = set()

    for _ in range(n_ratings):
        user = random.choice(users)
        product = random.choice(products)
        key = (user["user_id"], product["product_id"])
        if key in seen:
            continue
        seen.add(key)

        rating = {
            "user_id": user["user_id"],
            "product_id": product["product_id"],
            "rating": random.randint(1, 5),
            "review_text": fake.sentence(nb_words=random.randint(5, 20)) if random.random() > 0.3 else "",
            "timestamp": fake.date_time_between(
                start_date="-180d", end_date="now"
            ).isoformat(),
        }
        ratings.append(rating)

    # Inject some invalid ratings (out of range) for validation testing
    for _ in range(int(len(ratings) * 0.01)):
        idx = random.randint(0, len(ratings) - 1)
        ratings[idx]["rating"] = random.choice([0, 6, -1, 10])

    ts = datetime.now().strftime("%Y%m%d")
    filepath = RAW_DIR / "ratings" / f"ratings_{ts}.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ratings[0].keys())
        writer.writeheader()
        writer.writerows(ratings)

    logger.info(f"Ratings saved to {filepath}")
    return ratings


def main():
    log_file = LOG_DIR / "data_generation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), rotation="10 MB", level="INFO")

    logger.info("=" * 60)
    logger.info("Starting RecoMart Synthetic Data Generation")
    logger.info("=" * 60)

    setup_directories()
    users = generate_users(n_users=1000)
    products = generate_products(n_products=500)
    generate_clickstream(users, products, n_events=50000)
    generate_transactions(users, products, n_transactions=10000)
    generate_ratings(users, products, n_ratings=20000)

    logger.info("=" * 60)
    logger.info("Data generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
