from pydantic import BaseModel

class ShipPricePred(BaseModel):
    year: float
    model: str 
    category: str 
    length: float  
    fuel_type: str  
    hull_material: str 
    country: str 
    description: str