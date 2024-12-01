from pathlib import Path
from typing import Dict, Any

class ModelService:
    """Service for handling model-related operations."""

    # Allowed plant types and their associated file paths
    PLANT_TYPES = [
        "tomato", "lettuce", "spinach", "chili", "melon", "strawberry", "cucumber"
    ]

    @classmethod
    def validate_plant_type(cls, plant_type: str) -> Dict[str, Any]:
        """
        Validates if the provided plant type is allowed and its path exists.

        Args:
            plant_type (str): The type of plant to validate.

        Returns:
            bool: The file path of the corresponding model.

        Raises:
            HTTPException: If the plant type is invalid or the model file is missing.
        """
        if plant_type not in cls.PLANT_TYPES:
            return {
                "status": "error",
                "error_message": f"Invalid plant type. Supported types: {', '.join(cls.PLANT_TYPES)}",
            }

        return None
