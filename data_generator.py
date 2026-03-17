"""
data_generator.py
CargoCredit – XchangeBox Perishable Trade Finance Risk Engine
Generates synthetic Nigerian agricultural shipment data.
Run once: python data_generator.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5_000

# ── Nigerian routes ───────────────────────────────────────────────────────
CITIES = {
    'Kano':          {'lat': 12.00, 'lon': 8.52,  'zone': 'north'},
    'Lagos':         {'lat':  6.52, 'lon': 3.38,  'zone': 'south'},
    'Ibadan':        {'lat':  7.38, 'lon': 3.93,  'zone': 'south'},
    'Abuja':         {'lat':  9.07, 'lon': 7.40,  'zone': 'central'},
    'Port Harcourt': {'lat':  4.82, 'lon': 7.04,  'zone': 'south'},
    'Kaduna':        {'lat': 10.52, 'lon': 7.44,  'zone': 'north'},
    'Onitsha':       {'lat':  6.14, 'lon': 6.79,  'zone': 'south'},
    'Zaria':         {'lat': 11.08, 'lon': 7.71,  'zone': 'north'},
    'Jos':           {'lat':  9.92, 'lon': 8.89,  'zone': 'central'},
    'Enugu':         {'lat':  6.44, 'lon': 7.50,  'zone': 'south'},
}
CITY_NAMES = list(CITIES.keys())

# ── Products with cold-chain requirements ────────────────────────────────
PRODUCTS = {
    'Tomatoes':     {'max_temp': 25, 'shelf_hrs': 72,  'value_per_kg': 350},
    'Leafy Greens': {'max_temp': 10, 'shelf_hrs': 48,  'value_per_kg': 280},
    'Strawberries': {'max_temp':  8, 'shelf_hrs': 36,  'value_per_kg': 800},
    'Fresh Fish':   {'max_temp':  4, 'shelf_hrs': 24,  'value_per_kg': 600},
    'Dairy':        {'max_temp':  6, 'shelf_hrs': 48,  'value_per_kg': 420},
    'Mangoes':      {'max_temp': 20, 'shelf_hrs': 96,  'value_per_kg': 200},
    'Peppers':      {'max_temp': 22, 'shelf_hrs': 96,  'value_per_kg': 180},
}
PRODUCT_NAMES = list(PRODUCTS.keys())

def haversine_km(c1, c2):
    """Straight-line distance between two cities in km."""
    lat1, lon1 = np.radians(CITIES[c1]['lat']), np.radians(CITIES[c1]['lon'])
    lat2, lon2 = np.radians(CITIES[c2]['lat']), np.radians(CITIES[c2]['lon'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

records = []

for i in range(N):
    # Route
    origin = np.random.choice(CITY_NAMES)
    dest   = np.random.choice([c for c in CITY_NAMES if c != origin])
    dist   = haversine_km(origin, dest) * np.random.uniform(1.2, 1.6)  # road factor

    # Product
    product   = np.random.choice(PRODUCT_NAMES)
    prod_info = PRODUCTS[product]

    # Shipment size
    cargo_kg     = np.random.uniform(200, 5000)
    invoice_ngn  = cargo_kg * prod_info['value_per_kg'] * np.random.uniform(0.9, 1.1)

    # Transit conditions
    avg_speed_kmh  = np.random.uniform(35, 65)
    transit_hrs    = dist / avg_speed_kmh
    delay_hrs      = np.random.exponential(1.5)          # random delays
    total_hrs      = transit_hrs + delay_hrs

    # Temperature management
    has_cold_chain   = np.random.choice([True, False], p=[0.45, 0.55])
    ambient_temp     = np.random.uniform(28, 38)          # Nigerian climate
    if has_cold_chain:
        avg_temp = np.random.uniform(prod_info['max_temp'] - 2,
                                     prod_info['max_temp'] + 4)
    else:
        avg_temp = ambient_temp + np.random.normal(0, 3)

    max_temp       = avg_temp + np.random.uniform(2, 8)
    temp_exceedance = max(0, max_temp - prod_info['max_temp'])

    # Humidity and road roughness
    avg_humidity   = np.random.uniform(55, 92)
    road_quality   = np.random.choice(['good', 'fair', 'poor'],
                                       p=[0.25, 0.45, 0.30])
    road_num       = {'good': 0, 'fair': 1, 'poor': 2}[road_quality]
    vibration_score = road_num * np.random.uniform(0.5, 1.5)

    # Borrower profile
    borrower_history = np.random.choice(['excellent', 'good', 'fair', 'poor'],
                                         p=[0.20, 0.40, 0.25, 0.15])
    history_num      = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3}[borrower_history]
    prior_defaults   = np.random.poisson(history_num * 0.4)

    # Time ratio: how long in transit vs shelf life
    time_ratio = total_hrs / prod_info['shelf_hrs']

    # ── Spoilage probability (ground truth) ──────────────────────────────
    spoilage_score = (
        0.30 * min(time_ratio, 1.5)
        + 0.25 * min(temp_exceedance / 10, 1.0)
        + 0.15 * (1 - int(has_cold_chain))
        + 0.10 * road_num / 2
        + 0.10 * max(0, avg_humidity - 75) / 20
        + 0.10 * min(delay_hrs / 6, 1.0)
        + np.random.normal(0, 0.08)
    )
    spoilage_prob = float(np.clip(spoilage_score, 0.02, 0.98))
    spoiled       = int(spoilage_prob > np.random.uniform(0.35, 0.65))

    # ── Credit default probability ────────────────────────────────────────
    # Default risk = spoilage risk + borrower risk
    default_score = (
        0.50 * spoilage_prob
        + 0.30 * history_num / 3
        + 0.20 * min(prior_defaults / 3, 1.0)
        + np.random.normal(0, 0.06)
    )
    default_prob = float(np.clip(default_score, 0.01, 0.97))
    defaulted    = int(default_prob > np.random.uniform(0.40, 0.70))

    # ── Suggested financing rate (%) ──────────────────────────────────────
    # Base rate 2%, +risk premium up to 8%
    financing_rate = round(2.0 + default_prob * 8.0, 2)

    records.append({
        # Identity
        'shipment_id':      f'XB-{2024_0001 + i}',
        'product':          product,
        'origin':           origin,
        'destination':      dest,
        'origin_lat':       CITIES[origin]['lat'],
        'origin_lon':       CITIES[origin]['lon'],
        'dest_lat':         CITIES[dest]['lat'],
        'dest_lon':         CITIES[dest]['lon'],

        # Shipment
        'cargo_kg':         round(cargo_kg, 1),
        'invoice_ngn':      round(invoice_ngn, 0),
        'distance_km':      round(dist, 1),
        'transit_hrs':      round(transit_hrs, 2),
        'delay_hrs':        round(delay_hrs, 2),
        'total_hrs':        round(total_hrs, 2),
        'shelf_hrs':        prod_info['shelf_hrs'],
        'time_ratio':       round(time_ratio, 3),

        # Conditions
        'has_cold_chain':   int(has_cold_chain),
        'avg_temp':         round(avg_temp, 1),
        'max_temp':         round(max_temp, 1),
        'max_temp_allowed': prod_info['max_temp'],
        'temp_exceedance':  round(temp_exceedance, 1),
        'avg_humidity':     round(avg_humidity, 1),
        'road_quality':     road_quality,
        'vibration_score':  round(vibration_score, 3),

        # Borrower
        'borrower_history': borrower_history,
        'prior_defaults':   prior_defaults,

        # Targets
        'spoilage_prob':    round(spoilage_prob, 4),
        'spoiled':          spoiled,
        'default_prob':     round(default_prob, 4),
        'defaulted':        defaulted,
        'financing_rate':   financing_rate,
    })

df = pd.DataFrame(records)
df.to_csv('cargo_data.csv', index=False)

print(f"Generated {N} shipments")
print(f"Spoilage rate:  {df['spoiled'].mean():.1%}")
print(f"Default rate:   {df['defaulted'].mean():.1%}")
print(f"Avg invoice:    ₦{df['invoice_ngn'].mean():,.0f}")
print(f"Avg rate:       {df['financing_rate'].mean():.2f}%")
print(df[['spoilage_prob','default_prob','financing_rate']].describe().round(3))