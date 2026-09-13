from pydantic import BaseModel, Field


class RGBResponse(BaseModel):
    r: float = Field(..., description="Average Red color component (0-255)")
    g: float = Field(..., description="Average Green color component (0-255)")
    b: float = Field(..., description="Average Blue color component (0-255)")


class CIELABResponse(BaseModel):
    l: float = Field(..., description="Lightness component (0-100)")
    a: float = Field(..., description="Red-Green component")
    b: float = Field(..., description="Yellow-Blue component")
    chroma: float = Field(default=0.0, description="Chroma / Color Saturation C* = sqrt(a*^2 + b*^2)")
    hue_angle: float = Field(default=0.0, description="Hue Angle h° = arctan2(b*, a*) in degrees")


class PredictionResponse(BaseModel):
    predicted_score: int = Field(..., description="Predicted Yolk Color Fan Score (Integer 1-15)")
    raw_score: float = Field(..., description="Raw continuous prediction score from ML model")
    rgb: RGBResponse = Field(..., description="Extracted average RGB color features")
    cielab: CIELABResponse = Field(..., description="Extracted CIELAB color features")
