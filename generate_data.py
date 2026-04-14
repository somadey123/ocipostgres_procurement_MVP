import json
import random
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "synthetic"
POLICY_DIR = DATA_DIR / "policies"

random.seed(42)

ADDITIONAL_INVENTORY_ROWS = 150
ADDITIONAL_VENDOR_ROWS = 150
ADDITIONAL_POLICY_DOCS = 150

REGIONS = ["APAC", "EMEA", "NA"]

CATEGORY_CATALOG = {
    "laptop": [
        ("Dell Latitude 5450", "14 inch business laptop for engineering and office use", 1250),
        ("Lenovo ThinkPad T14", "durable business laptop with long battery life", 1180),
        ("HP EliteBook 840", "secure business ultrabook for hybrid teams", 1320),
        ("MacBook Pro 14", "high-performance laptop for developers and design", 2100),
    ],
    "monitor": [
        ("Dell UltraSharp 27", "27 inch QHD office monitor", 360),
        ("LG Ergo 32", "32 inch monitor with ergonomic arm", 520),
        ("Samsung ViewFinity 27", "27 inch 4K productivity monitor", 480),
    ],
    "accessory": [
        ("HP USB-C Dock", "universal docking station for laptops", 180),
        ("Logitech MX Keys", "wireless keyboard for office productivity", 120),
        ("Logitech MX Master 3", "ergonomic wireless mouse", 95),
        ("Jabra Evolve2 Headset", "noise-canceling headset for meetings", 220),
    ],
    "network": [
        ("Cisco Catalyst 9200", "managed access switch for branch office", 3400),
        ("Ubiquiti WiFi 6 AP", "enterprise wireless access point", 290),
        ("Fortinet Firewall 100F", "mid-size office security appliance", 4200),
    ],
    "furniture": [
        ("Ergonomic Office Chair", "adjustable ergonomic chair for office workstations", 320),
        ("Sit-Stand Desk 140cm", "height-adjustable desk for hybrid office", 610),
        ("Storage Cabinet", "lockable storage cabinet for office supplies", 260),
    ],
    "printer": [
        ("HP LaserJet Pro", "monochrome laser printer for team usage", 410),
        ("Brother MFC-L8900", "multi-function office printer", 780),
        ("Canon ImageRunner C3226", "color office copier and scanner", 2400),
    ],
}

vendors = []


def generate_vendors(target_rows: int, start_index: int = 1) -> list[dict]:
    category_names = list(CATEGORY_CATALOG.keys())
    prefixes = ["Prime", "Global", "Summit", "Vertex", "BluePeak", "Nova", "Atlas", "Pioneer"]
    suffixes = ["Supplies", "Distribution", "Technologies", "Partners", "Procurement", "Direct"]

    generated = []
    for i in range(start_index, start_index + target_rows):
        p = random.choice(prefixes)
        s = random.choice(suffixes)
        vendor_name = f"{p} {s} {i:02d}"
        categories = sorted(random.sample(category_names, k=random.randint(1, 3)))
        generated.append(
            {
                "vendor_id": f"V-{i:03d}",
                "vendor_name": vendor_name,
                "categories": categories,
                "preferred": i <= max(10, target_rows // 4),
                "region": random.choice(REGIONS),
                "sla_days": random.randint(2, 14),
                "rating": round(random.uniform(3.6, 4.9), 1),
            }
        )
    return generated


def generate_inventory(target_rows: int, vendors_pool: list[dict], start_index: int = 1) -> list[dict]:
    generated = []
    for i in range(start_index, start_index + target_rows):
        category = random.choice(list(CATEGORY_CATALOG.keys()))
        item_name, description, base_price = random.choice(CATEGORY_CATALOG[category])
        matching_vendors = [v for v in vendors_pool if category in v["categories"]]
        vendor = random.choice(matching_vendors) if matching_vendors else random.choice(vendors_pool)

        generated.append(
            {
                "item_id": f"INV-{i:03d}",
                "item_name": f"{item_name} {random.choice(['Standard', 'Plus', 'Pro'])}",
                "category": category,
                "description": description,
                "unit": "each",
                "stock_qty": random.randint(0, 150),
                "preferred_vendor": vendor["vendor_name"],
                "estimated_price": int(base_price * random.uniform(0.92, 1.18)),
                "lead_time_days": random.randint(2, 21),
            }
        )
    return generated


def inventory_uniqueness_key(row: dict) -> tuple:
    return (
        row["item_name"],
        row["category"],
        row["description"],
        row["preferred_vendor"],
        row["estimated_price"],
        row["lead_time_days"],
        row["stock_qty"],
    )


def generate_unique_inventory(
    target_rows: int,
    vendors_pool: list[dict],
    existing_rows: list[dict],
    start_index: int = 1,
) -> list[dict]:
    existing_keys = {inventory_uniqueness_key(r) for r in existing_rows}
    unique_rows: list[dict] = []
    next_index = start_index

    while len(unique_rows) < target_rows:
        candidate = generate_inventory(1, vendors_pool, start_index=next_index)[0]
        next_index += 1
        key = inventory_uniqueness_key(candidate)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        unique_rows.append(candidate)

    return unique_rows


policies = {
    "policy_approval.md": """# Procurement Policy

- Purchases under 500 USD may proceed with manager review.
- Purchases from preferred vendors are allowed when pricing is within 10% of the lowest non-preferred quote.
- Furniture purchases require facilities approval.
- Restricted categories must be escalated to procurement leadership.
- The assistant must not place orders automatically.
""",
    "policy_vendor.md": """# Vendor Policy

- Prefer approved vendors for standard IT equipment.
- If the preferred vendor is unavailable, compare at least two alternatives.
- Select the vendor with the best combination of price, SLA, and rating.
""",
    "policy_escalation.md": """# Escalation Policy

- If confidence is low, ask a clarification question.
- If policy conflicts appear, return the conflict and request human review.
- If no suitable item is found, recommend sourcing assistance.
""",
    "policy_security_it.md": """# Security & IT Procurement Policy

- All endpoint devices must support disk encryption and endpoint protection tools.
- Network devices must include current firmware support and security patch commitments.
- Procurement for software/SaaS must include security questionnaire review.
- High-risk suppliers require InfoSec and legal approval before PO creation.
""",
    "policy_contracts.md": """# Contracting & Legal Policy

- Any purchase above 25,000 USD requires legal contract review.
- Auto-renewal terms above 12 months require procurement approval.
- Non-standard indemnity clauses must be escalated to legal.
- Payment terms should target Net 30 unless approved exception exists.
""",
    "policy_budgeting.md": """# Budget & Cost-Control Policy

- Requests above department budget cap must include finance approval.
- For purchases above 10,000 USD, compare at least 3 quotes.
- Include total cost of ownership (license, support, shipping) in recommendation.
- Bundle opportunities should be highlighted if cost savings exceed 8%.
""",
    "policy_sustainability.md": """# Sustainability Procurement Policy

- Prefer vendors with published environmental compliance reporting.
- Prefer devices certified under recognized energy standards.
- Consolidate shipments where feasible to reduce logistics emissions.
- Recommend refurbishment options for non-critical equipment refresh.
""",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_additional_policy_docs(existing_count: int, target_rows: int) -> dict[str, str]:
    sections = [
        "Approval workflow",
        "Risk controls",
        "SLA requirements",
        "Vendor due diligence",
        "Cost optimization",
        "Compliance checks",
    ]
    generated = {}
    for i in range(existing_count + 1, existing_count + target_rows + 1):
        section = random.choice(sections)
        generated[f"policy_generated_{i:03d}.md"] = f"""# Generated Procurement Policy {i:03d}

- Scope: Applies to category-controlled purchases and standard replenishment.
- Focus Area: {section}
- Requirement: Obtain and document at least two qualified quotes before recommendation.
- Control: Ensure recommended vendors meet contractual, security, and budget constraints.
- Escalation: Route exceptions to procurement and finance approvers for final decision.
"""
    return generated

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_DIR.mkdir(parents=True, exist_ok=True)

    inventory_path = DATA_DIR / "inventory.jsonl"
    vendors_path = DATA_DIR / "vendors.jsonl"

    existing_vendors = read_jsonl(vendors_path)
    existing_inventory = read_jsonl(inventory_path)

    next_vendor_index = len(existing_vendors) + 1
    new_vendors = generate_vendors(ADDITIONAL_VENDOR_ROWS, start_index=next_vendor_index)
    all_vendors = existing_vendors + new_vendors

    next_inventory_index = len(existing_inventory) + 1
    new_inventory_items = generate_unique_inventory(
        ADDITIONAL_INVENTORY_ROWS,
        all_vendors,
        existing_inventory,
        start_index=next_inventory_index,
    )

    append_jsonl(vendors_path, new_vendors)
    append_jsonl(inventory_path, new_inventory_items)

    for name, content in policies.items():
        (POLICY_DIR / name).write_text(content, encoding="utf-8")

    existing_generated_policy_count = len(list(POLICY_DIR.glob("policy_generated_*.md")))
    generated_policies = build_additional_policy_docs(
        existing_generated_policy_count,
        ADDITIONAL_POLICY_DOCS,
    )
    for name, content in generated_policies.items():
        (POLICY_DIR / name).write_text(content, encoding="utf-8")

    print(
        f"Added {len(new_inventory_items)} inventory rows, {len(new_vendors)} vendor rows, "
        f"and {len(generated_policies)} policy docs in data/synthetic/."
    )

if __name__ == "__main__":
    main()