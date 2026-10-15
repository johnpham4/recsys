import inspect
from typing import Any, List, Optional, Dict, Callable


def func_to_schema(func: Callable) -> Dict[str, Any]:
    """Convert Python function to JSON Schema for LLM."""
    sig = inspect.signature(func)
    doc = func.__doc__ or "No Description provided"

    properties = {}
    required = []

    type_map = {
        str: "string", 
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        Any: "any"
    }

    for param_name, param in sig.parameters.items():
        param_type = type_map.get(param.annotation, "string")
        properties[param_name] = {"type": param_type}

        # if no default value and not Union, add to required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

class PluginRegistry:
    _tools: Dict[str, Any] = {}

    @classmethod
    def register(cls, func: Callable = None):
        def decorator(f: Callable):
            cls._tools[f.__name__] = f
            return f
        return decorator

    @classmethod
    def get_tools_for_llm(cls) -> List[Dict[str, Any]]:
        return [func_to_schema(f) for f in cls._tools.values()]

    @classmethod
    def execute(cls, tool_name: str, **kwargs) -> Any:
        if tool_name not in cls._tools:
            return f"Error: Tool {tool_name} not found"
        try:
            return cls._tools[tool_name](**kwargs)
        except Exception as e:
            return f"Execution error in '{tool_name}': {str(e)}"



    

