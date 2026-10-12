import os
import json
import shutil
import subprocess
from decimal import Decimal
from dotenv import load_dotenv

load_dotenv()


def resolve_modal_bin():
    if shutil.which("modal"):
        return ["modal"]
    uvx_path = shutil.which("uvx")
    if uvx_path:
        return [uvx_path, "modal"]
    fallback = os.path.expanduser("~/.local/bin/uvx")
    if os.path.exists(fallback):
        return [fallback, "modal"]
    return ["modal"]


def query_modal_api(args):
    """Executes modal CLI with --json and returns parsed JSON data."""
    cmd = resolve_modal_bin() + args + ["--json"]
    env = os.environ.copy()
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        return None, res.stderr.strip() or res.stdout.strip()
    try:
        return json.loads(res.stdout), None
    except json.JSONDecodeError:
        return None, "Failed to parse JSON response from Modal API"


def main():
    token_id = os.getenv("MODAL_TOKEN_ID", "")
    masked_token = f"{token_id[:6]}...{token_id[-4:]}" if len(token_id) > 10 else "N/A"

    print("=" * 60)
    print(f"MODAL USAGE REPORT | Auth: {masked_token}")
    print("=" * 60)

    # 1. Billing Summary (via official Modal API JSON)
    billing_data, err = query_modal_api(["billing", "summary"])
    print("\n[BILLING & USAGE (FROM MODAL API)]")
    if err:
        print(f"Error querying billing: {err}")
    elif billing_data:
        metered = float(Decimal(billing_data.get("metered_cost", "0")))
        billed = float(Decimal(billing_data.get("billed_cost", "0")))
        breakdown = billing_data.get("metered_cost_breakdown", {})
        adjustments = billing_data.get("adjustments", {})

        print(f"  Metered Cost (Total Used):   ${metered:.4f}")
        for k, v in breakdown.items():
            cost_val = float(Decimal(str(v)))
            if cost_val > 0:
                print(f"    - {k}: ${cost_val:.4f}")

        print(f"  Free Credits / Adjustments:  ")
        for k, v in adjustments.items():
            adj_val = float(Decimal(str(v)))
            if adj_val != 0:
                print(f"    - {k.replace('_', ' ').title()}: ${adj_val:.4f}")

        print(f"  Billed Cost (To Pay):        ${billed:.4f}")

        # Modal Free Community Tier Policy: $30/month
        free_tier_allowance = float(os.getenv("MODAL_MONTHLY_LIMIT", "30.0"))
        remaining_budget = max(0.0, free_tier_allowance - billed)
        est_t4_hours = remaining_budget / 0.59

        print("-" * 60)
        print(f"  Modal Community Free Tier:   ${free_tier_allowance:.2f}/month")
        print(f"  Current Billed vs Free Tier: ${billed:.2f} / ${free_tier_allowance:.2f}")
        print(f"  Remaining Free Allowance:    ${remaining_budget:.2f} (~{est_t4_hours:.1f} hrs GPU T4)")
    else:
        print("  No billing data returned.")

    # 2. Active Apps (via official Modal API JSON)
    apps_data, err = query_modal_api(["app", "list"])
    print("\n[ACTIVE APPS (FROM MODAL API)]")
    if err:
        print(f"Error querying apps: {err}")
    elif isinstance(apps_data, list):
        active_apps = [a for a in apps_data if a.get("state") in ("running", "deploying")]
        if active_apps:
            for app in active_apps:
                print(f"  - App: {app.get('app_id')} | Name: {app.get('description')} | State: {app.get('state')}")
        else:
            print("  No apps currently running (Compute Idle: $0/hr).")
    else:
        print("  No apps found.")

    # 3. Storage Volumes (via official Modal API JSON)
    volumes_data, err = query_modal_api(["volume", "list"])
    print("\n[PERSISTENT VOLUMES (FROM MODAL API)]")
    if err:
        print(f"Error querying volumes: {err}")
    elif isinstance(volumes_data, list):
        if volumes_data:
            for vol in volumes_data:
                name = vol.get("name", "unnamed")
                created = vol.get("created_at", "N/A")
                print(f"  - {name:<30} (Created: {created})")
        else:
            print("  No volumes found.")
    else:
        print("  No volumes data.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
