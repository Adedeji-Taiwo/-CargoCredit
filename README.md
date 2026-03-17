# CargoCredit – Perishable Trade Finance Risk Engine

> A machine learning-powered credit decision tool built for XchangeBox to automate risk assessment on perishable agricultural cargo invoices. Combines IoT sensor simulation, spoilage prediction, and financial risk pricing into a single loan officer decision-support platform.

**Built at XchangeBox Technologies · Advanced Analytics for Agribusiness · MSc Agribusiness & Innovation · UM6P**

---

## Live Demo

[Launch App on Streamlit Cloud](https://your-app-name.streamlit.app) ← *replace after deployment*

---

## Problem Statement

XchangeBox finances perishable agricultural invoices across Nigerian supply chains. The core challenge: when a Farmer Cooperative in Kano ships 3,000 kg of fresh fish to Lagos Market and presents the invoice for financing, how do you price that risk?

The cargo may spoil in transit. The borrower may default. Both risks compound each other — spoiled cargo reduces collateral value, increasing default probability.

Traditional credit scoring ignores the cargo dimension entirely. This tool prices both.

---

## Architecture

### Two-model system

| Model | Task | Algorithm | AUC |
|---|---|---|---|
| Spoilage Model | Will this cargo spoil in transit? | Logistic Regression | ~0.97 |
| Default Model | Will this borrower default? | Gradient Boosting | ~0.97 |

### Decision logic

| Condition | Recommendation |
|---|---|
| Spoilage < 40% AND Default < 30% | ✓ Approve |
| Spoilage 40–65% OR Default 30–55% | ⚑ Manual Review |
| Spoilage > 65% OR Default > 55% | ✗ Decline |

### Pricing formula

```
Discount Rate = Base Rate (by credit grade) + 3.0 × Spoilage Probability
```

Grade A base: 1.5% · Grade B: 2.5% · Grade C: 4.0%

---

## Features

| Page | What it shows |
|---|---|
| New Application | Loan officer form → instant credit decision + gauges + route summary |
| Portfolio Monitor | KPIs, risk matrix scatter, product risk bar, route heatmap, grade exposure |
| Model Intelligence | Model comparison table, feature importance charts, full methodology |

---

## Project Structure

```
cargocredit/
├── data_generator.py    # generates cargo_data.csv (run once)
├── train_model.py       # trains both models, saves pkl files (run once)
├── app.py               # Streamlit dashboard
├── requirements.txt
├── packages.txt
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/cargocredit
cd cargocredit
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Step 1 — generate training data
python data_generator.py

# Step 2 — train and save models
python train_model.py

# Step 3 — launch dashboard
streamlit run app.py
```

---

## Deployment

Commit all `.pkl` and `.json` files alongside the code — Streamlit Cloud loads them directly without retraining.

```bash
git add .
git commit -m "initial commit with trained models"
git push
```

Deploy at [share.streamlit.io](https://share.streamlit.io).

---

## Data

5,000 simulated Nigerian agricultural shipments across 15 routes, 7 perishable product types, 200 borrower profiles. Spoilage model based on perishability kinetics literature (temperature exceedance × transit time / shelf life). Default model incorporates credit grade, platform tenure, and cargo risk signal.

---

## References

Arah, P. B., et al. (2015). Factors affecting postharvest quality and losses in fruits and vegetables. *Food and Nutrition Sciences*, 6, 1207–1216.

Ntsafack, B., et al. (2021). Cold chain management in developing countries. *Journal of Food Quality*.

XchangeBox Technologies (2025). *Trade Finance Infrastructure for African Agricultural Supply Chains*. Internal.

---

*Advanced Analytics for Agribusiness · MSc Agribusiness & Innovation · UM6P*