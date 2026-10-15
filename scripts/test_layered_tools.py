"""
Script kiểm tra hoạt động của tầng Services và Tools mới tái cấu trúc.
"""
import os
import json
from dotenv import load_dotenv

load_dotenv()

from src.harness.registry import PluginRegistry
import src.harness.tools  # Kích hoạt đăng ký decorator
from src.services.catalog_service import CatalogService
from src.services.recsys_service import RecSysService
from src.services.cart_service import CartService


def test_services_and_tools():
    print("=" * 60)
    print("🚀 1. KIỂM TRA ĐĂNG KÝ TOOLS TRONG PLUGIN REGISTRY")
    print("=" * 60)
    tools_schema = PluginRegistry.get_tools_for_llm()
    print(f"Tổng số Tools đã đăng ký: {len(tools_schema)}")
    for t in tools_schema:
        func = t["function"]
        print(f"  - Tool: {func['name']} -> {func['description'][:60]}...")

    print("\n" + "=" * 60)
    print("📦 2. KIỂM TRA TẦNG SERVICES TRỰC TIẾP")
    print("=" * 60)
    catalog = CatalogService()
    recsys = RecSysService(catalog_service=catalog)
    cart = CartService(catalog_service=catalog)

    # Test catalog
    prods = catalog.get_products_by_asins(["B08N5WRWNW", "B09ABC1234"])
    print(f"Catalog query: {len(prods)} sản phẩm:")
    for p in prods:
        print(f"  * {p['asin']} | {p['title']} | ${p['price']} | {p['category']}")

    # Test recsys
    recs = recsys.get_personalized_recommendations(user_id="user_test_01", max_price=60.0, top_k=2)
    print(f"\nRecSys recommendations (max_price=60$): {len(recs)} món:")
    for r in recs:
        print(f"  * {r['asin']} | {r['title']} | ${r['price']}")

    # Test tool execution qua PluginRegistry.execute
    print("\n" + "=" * 60)
    print("⚡ 3. KIỂM TRA THỰC THI TOOL QUA PLUGIN REGISTRY")
    print("=" * 60)
    exec_res = PluginRegistry.execute(
        "add_to_cart",
        user_id="user_test_01",
        asin="B08N5WRWNW",
        quantity=2
    )
    print("PluginRegistry.execute('add_to_cart') result:\n", json.dumps(exec_res, ensure_ascii=False, indent=2))

    pref_res = PluginRegistry.execute(
        "record_user_preference",
        user_id="user_test_01",
        key="brand",
        value="Razer",
        sentiment="negative"
    )
    print("\nPluginRegistry.execute('record_user_preference') result:\n", json.dumps(pref_res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_services_and_tools()
