from typing import List, Optional, Dict
from .registry import PluginRegistry


@PluginRegistry.register()
def get_i2i_recommendations(
    item_id: str, 
    category: Optional[str], 
    max_prices: Optional[int],
) -> List[Dict[str, Any]]:
    pass


@PluginRegistry.register()
def get_product_details(product_id: str) -> Dict[str, Any]:
    pass


@PluginRegistry.register()
def get_recommedations(
    user_id: str, 
    category: Optional[str], 
    max_prices: Optional[str]
) -> List[Dict[str, Any]]:
    pass


@PluginRegistry.register()
def get_user_history(
    user_id: str,
    limit: Optional[str],
) -> List[Dict[str, Any]]:
    pass


@PluginRegistry.register()
def get_user_profile(
    user_id: str,
) -> Dict[str, Any]:
    pass






