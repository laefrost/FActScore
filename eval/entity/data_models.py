"""Pydantic models of the entity annotation pipeline.

Ported from ``annotation_pipeline/data_models.py`` in hallucination_probes.
``DatasetItem``, which wrapped a HuggingFace conversation row, is not needed
here. ``label`` and ``verification_note`` have defaults so that an item the
model left incomplete (a truncated answer) still validates.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

LABELS = ("Supported", "Not Supported", "Insufficient Information")


class AnnotatedSpan(BaseModel):
    """Represents an entity with its verification status"""
    span: str = Field(description="The minimal span containing just the entity (e.g., 'Sarah Chen', not 'Dr. Sarah Chen from MIT')")
    label: Optional[Literal["Supported", "Not Supported", "Insufficient Information"]] = Field(
        default=None,
        description="Whether the entity/fact is verified as real, fabricated, or unverifiable. Downstream we treat 'Insufficient Information' in the same way as 'Not Supported'."
    )
    verification_note: str = Field(default="", description="Brief explanation of the verification result")
    index: Optional[int] = Field(default=None, description="The index of the span within the completion/corpus of text")

    @model_validator(mode='before')
    @classmethod
    def validate_label(cls, values):
        """Validate and normalize the label field"""
        if isinstance(values, dict) and 'label' in values:
            if values['label'] not in LABELS:
                # Set to a default value instead of throwing an error
                values['label'] = None
        return values
