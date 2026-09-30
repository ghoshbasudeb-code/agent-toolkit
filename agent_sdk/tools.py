from langchain_core.tools import tool

@tool
def calculate_metrics(data_points: list[float]) -> dict:
    """Calculates mean and total for a list of numeric values."""
    if not data_points:
        return {"mean": 0.0, "total": 0.0}
    return {
        "mean": sum(data_points) / len(data_points),
        "total": sum(data_points)
    }

@tool
def format_json_response(data: dict) -> str:
    """Formats a dictionary into a clean string representation."""
    import json
    return json.dumps(data, indent=2)
