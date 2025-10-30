"""
Forecast Generation Service for AI Forecasting Pipeline.

Generates evidence-based forecasts using LLM with structured reasoning,
probability estimation, and uncertainty quantification.
"""

import json
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field, field_validator

from services.llm_client import LLMClient, create_llm_client, ParsingError
from services.timeline import Timeline, TimelineEntry
from db.models import Forecast as ForecastModel

logger = structlog.get_logger(__name__)


# Prompt version for tracking
FORECAST_PROMPT_VERSION = "v1.0"


class ConfidenceLevel(str, Enum):
    """Confidence levels for forecasts."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PredictionType(str, Enum):
    """Types of predictions."""
    PROBABILITY = "probability"     # Numeric probability (0-100)
    QUALITATIVE = "qualitative"     # HIGH/MEDIUM/LOW
    BINARY = "binary"               # YES/NO


class ForecastError(Exception):
    """Base exception for forecast-related errors."""
    pass


class ForecastOutput(BaseModel):
    """
    Pydantic model for validating LLM forecast output.

    This ensures the LLM returns all required fields in the expected format.
    """
    prediction: Union[float, str] = Field(
        ...,
        description="Probability (0-100) or qualitative prediction (HIGH/MEDIUM/LOW/YES/NO)"
    )
    prediction_type: str = Field(
        ...,
        description="Type of prediction: 'probability', 'qualitative', or 'binary'"
    )
    reasoning: List[str] = Field(
        ...,
        min_length=1,
        description="List of reasoning steps, each as a separate string"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="List of limitations and uncertainties"
    )
    key_evidence_events: List[int] = Field(
        default_factory=list,
        description="List of event indices used as key evidence"
    )
    confidence_level: str = Field(
        ...,
        description="Overall confidence: 'high', 'medium', or 'low'"
    )

    @field_validator('confidence_level')
    @classmethod
    def validate_confidence(cls, v):
        """Validate confidence level."""
        valid = ['high', 'medium', 'low']
        if v.lower() not in valid:
            raise ValueError(f"Confidence must be one of {valid}")
        return v.lower()

    @field_validator('prediction_type')
    @classmethod
    def validate_prediction_type(cls, v):
        """Validate prediction type."""
        valid = ['probability', 'qualitative', 'binary']
        if v.lower() not in valid:
            raise ValueError(f"Prediction type must be one of {valid}")
        return v.lower()


@dataclass
class Forecast:
    """
    A complete forecast with reasoning and evidence.

    Attributes:
        question: Original forecasting question
        prediction: Probability estimate or qualitative prediction
        prediction_type: Type of prediction
        reasoning_steps: List of reasoning steps
        caveats: List of caveats and limitations
        key_evidence_events: Indices of key evidence timeline entries
        confidence_level: Overall confidence level
        raw_response: Raw LLM response for debugging
        prompt_version: Version of prompt used
    """
    question: str
    prediction: Union[float, str]
    prediction_type: PredictionType
    reasoning_steps: List[str]
    caveats: List[str] = field(default_factory=list)
    key_evidence_events: List[int] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    raw_response: Optional[str] = None
    prompt_version: str = FORECAST_PROMPT_VERSION

    def to_model(self, run_id: int) -> ForecastModel:
        """
        Convert to database model.

        Args:
            run_id: ID of the pipeline run

        Returns:
            ForecastModel instance
        """
        # Convert prediction to probability float if possible
        probability = None
        if self.prediction_type == PredictionType.PROBABILITY:
            if isinstance(self.prediction, (int, float)):
                probability = float(self.prediction) / 100.0  # Convert to 0-1 range
            else:
                try:
                    probability = float(self.prediction) / 100.0
                except (ValueError, TypeError):
                    pass

        return ForecastModel(
            run_id=run_id,
            probability=probability,
            reasoning=json.dumps(self.reasoning_steps),
            caveats=json.dumps(self.caveats),
            raw_response=self.raw_response,
        )

    def format_markdown(self) -> str:
        """Format forecast as Markdown for display."""
        lines = []

        lines.append("# Forecast")
        lines.append("")
        lines.append(f"**Question:** {self.question}")
        lines.append("")

        # Prediction
        if self.prediction_type == PredictionType.PROBABILITY:
            lines.append(f"**Prediction:** {self.prediction}% probability")
        else:
            lines.append(f"**Prediction:** {self.prediction}")

        lines.append(f"**Confidence:** {self.confidence_level.value.upper()}")
        lines.append("")

        # Reasoning
        lines.append("## Reasoning")
        lines.append("")
        for i, step in enumerate(self.reasoning_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

        # Key Evidence
        if self.key_evidence_events:
            lines.append("## Key Evidence")
            lines.append("")
            lines.append(f"Based on {len(self.key_evidence_events)} key events from the timeline.")
            lines.append("")

        # Caveats
        if self.caveats:
            lines.append("## Caveats & Limitations")
            lines.append("")
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
            lines.append("")

        return "\n".join(lines)


class ForecastGenerator:
    """
    Service for generating forecasts from timelines using LLM.

    Features:
    - Structured prompt with timeline context
    - JSON output parsing and validation
    - Fallback retry on missing fields
    - Support for multiple prediction types
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_timeline_entries: int = 50,
        temperature: float = 0.3,
    ):
        """
        Initialize forecast generator.

        Args:
            llm_client: LLM client to use (defaults to Gemini)
            max_timeline_entries: Maximum timeline entries to include in prompt
            temperature: LLM temperature for generation
        """
        self.llm_client = llm_client or create_llm_client()
        self.max_timeline_entries = max_timeline_entries
        self.temperature = temperature
        self.logger = structlog.get_logger(__name__)

        self.logger.info(
            "forecast_generator_initialized",
            model=self.llm_client.model_name,
            max_timeline_entries=max_timeline_entries,
            temperature=temperature,
        )

    def _format_timeline_for_prompt(self, timeline: Timeline) -> str:
        """
        Format timeline entries for inclusion in prompt.

        Args:
            timeline: Timeline object

        Returns:
            Formatted string
        """
        lines = []

        # Limit number of entries
        entries = timeline.entries[:self.max_timeline_entries]

        lines.append(f"Timeline Coverage: {timeline.coverage_level.value}")
        lines.append(f"Total Events: {len(entries)}")

        if timeline.date_range and timeline.date_range[0]:
            lines.append(f"Date Range: {timeline.date_range[0]} to {timeline.date_range[1]}")

        lines.append("")
        lines.append("Events (chronological order):")
        lines.append("")

        for i, entry in enumerate(entries, 1):
            lines.append(f"[{i}] {entry.event_time}")
            lines.append(f"    {entry.summary}")

            if entry.actor_tags:
                lines.append(f"    Actors: {', '.join(entry.actor_tags[:3])}")

            if entry.citations:
                lines.append(f"    Sources: {len(entry.citations)} citation(s)")

            lines.append("")

        if len(timeline.entries) > self.max_timeline_entries:
            lines.append(f"... and {len(timeline.entries) - self.max_timeline_entries} more events")

        return "\n".join(lines)

    def _build_forecast_prompt(
        self,
        question: str,
        timeline: Timeline,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Build the forecast generation prompt.

        Args:
            question: Forecasting question
            timeline: Timeline object
            additional_context: Optional additional context

        Returns:
            Formatted prompt string
        """
        timeline_text = self._format_timeline_for_prompt(timeline)

        prompt = f"""You are an expert forecaster analyzing evidence to make predictions. Your task is to forecast the answer to a specific question based on a timeline of relevant events.

QUESTION TO FORECAST:
{question}

EVIDENCE TIMELINE:
{timeline_text}
"""

        if additional_context:
            prompt += f"\nADDITIONAL CONTEXT:\n{additional_context}\n"

        prompt += """
YOUR TASK:
Analyze the timeline and generate a forecast with transparent reasoning. You must output a JSON object with the following structure:

{
  "prediction": <probability 0-100 as number, or qualitative string like "HIGH"/"MEDIUM"/"LOW"/"YES"/"NO">,
  "prediction_type": "probability" | "qualitative" | "binary",
  "reasoning": [
    "First reasoning step explaining key evidence...",
    "Second reasoning step analyzing trends...",
    "Third reasoning step considering uncertainties...",
    ... (3-7 steps total)
  ],
  "caveats": [
    "First limitation or uncertainty...",
    "Second limitation or uncertainty...",
    ... (2-5 caveats)
  ],
  "key_evidence_events": [1, 5, 12, ...],  // Indices of most important events
  "confidence_level": "high" | "medium" | "low"
}

INSTRUCTIONS:
1. **Reasoning**: Provide 3-7 clear reasoning steps. Each step should:
   - Reference specific events from the timeline (use event numbers like [3], [7])
   - Explain how the evidence supports or contradicts the prediction
   - Consider trends, patterns, and trajectories
   - Address uncertainties and alternative scenarios

2. **Prediction**:
   - If the question asks for a probability, provide a numeric value 0-100
   - If asking "Will X happen?", you can use YES/NO or a probability
   - If uncertain, use qualitative predictions: HIGH/MEDIUM/LOW

3. **Caveats**: List 2-5 important limitations:
   - Missing information or blind spots
   - Assumptions made
   - External factors not captured in timeline
   - Potential for unexpected developments

4. **Key Evidence**: List the event indices (numbers) that were most influential

5. **Confidence**:
   - HIGH: Strong evidence, clear trends, few uncertainties
   - MEDIUM: Reasonable evidence, some uncertainties
   - LOW: Limited evidence, high uncertainty, conflicting signals

OUTPUT ONLY THE JSON OBJECT, NO OTHER TEXT.
"""

        return prompt

    def _parse_forecast_response(
        self,
        response: str,
        question: str,
    ) -> Forecast:
        """
        Parse and validate LLM response into Forecast object.

        Args:
            response: Raw LLM response (should be JSON)
            question: Original question

        Returns:
            Forecast object

        Raises:
            ForecastError: If parsing or validation fails
        """
        # Try to extract JSON if response has extra text
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first and last lines if they're code fence markers
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(
                "forecast_json_parse_error",
                error=str(e),
                response_preview=response[:500],
            )
            raise ForecastError(f"Failed to parse forecast JSON: {e}")

        # Validate with Pydantic
        try:
            forecast_output = ForecastOutput(**data)
        except Exception as e:
            self.logger.error(
                "forecast_validation_error",
                error=str(e),
                data=data,
            )
            raise ForecastError(f"Forecast validation failed: {e}")

        # Determine prediction type enum
        pred_type_map = {
            'probability': PredictionType.PROBABILITY,
            'qualitative': PredictionType.QUALITATIVE,
            'binary': PredictionType.BINARY,
        }
        prediction_type = pred_type_map.get(
            forecast_output.prediction_type,
            PredictionType.QUALITATIVE
        )

        # Determine confidence level enum
        conf_map = {
            'high': ConfidenceLevel.HIGH,
            'medium': ConfidenceLevel.MEDIUM,
            'low': ConfidenceLevel.LOW,
        }
        confidence_level = conf_map.get(
            forecast_output.confidence_level.lower(),
            ConfidenceLevel.MEDIUM
        )

        # Create Forecast object
        forecast = Forecast(
            question=question,
            prediction=forecast_output.prediction,
            prediction_type=prediction_type,
            reasoning_steps=forecast_output.reasoning,
            caveats=forecast_output.caveats,
            key_evidence_events=forecast_output.key_evidence_events,
            confidence_level=confidence_level,
            raw_response=response,
            prompt_version=FORECAST_PROMPT_VERSION,
        )

        return forecast

    def _generate_with_fallback(
        self,
        question: str,
        timeline: Timeline,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Generate forecast with fallback retry on parse errors.

        Args:
            question: Forecasting question
            timeline: Timeline object
            additional_context: Optional additional context

        Returns:
            LLM response string

        Raises:
            ForecastError: If generation fails after retries
        """
        # Build prompt
        prompt = self._build_forecast_prompt(question, timeline, additional_context)

        # Try to generate
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=2000,
                response_format="json",
            )
            return response
        except ParsingError as e:
            self.logger.warning(
                "forecast_generation_parse_error",
                error=str(e),
                attempt="initial",
            )

            # Try fallback prompt emphasizing JSON format
            fallback_prompt = prompt + "\n\nREMINDER: Output ONLY valid JSON, no additional text."

            try:
                response = self.llm_client.generate(
                    prompt=fallback_prompt,
                    temperature=self.temperature,
                    max_tokens=2000,
                )
                return response
            except Exception as e2:
                self.logger.error(
                    "forecast_generation_fallback_failed",
                    error=str(e2),
                )
                raise ForecastError(f"Forecast generation failed: {e2}")

        except Exception as e:
            self.logger.error(
                "forecast_generation_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ForecastError(f"Forecast generation failed: {e}")

    def generate_forecast(
        self,
        question: str,
        timeline: Timeline,
        additional_context: Optional[str] = None,
    ) -> Forecast:
        """
        Generate a forecast for a question based on a timeline.

        Args:
            question: Forecasting question
            timeline: Timeline object with events
            additional_context: Optional additional context

        Returns:
            Forecast object with prediction and reasoning

        Raises:
            ForecastError: If forecast generation fails
        """
        self.logger.info(
            "forecast_generation_start",
            question=question[:100],
            timeline_entries=timeline.total_events,
            coverage=timeline.coverage_level,
        )

        # Generate LLM response
        response = self._generate_with_fallback(question, timeline, additional_context)

        # Parse into Forecast object
        try:
            forecast = self._parse_forecast_response(response, question)
        except ForecastError:
            raise
        except Exception as e:
            self.logger.error(
                "forecast_parse_unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ForecastError(f"Unexpected error parsing forecast: {e}")

        self.logger.info(
            "forecast_generation_complete",
            prediction=forecast.prediction,
            prediction_type=forecast.prediction_type,
            confidence=forecast.confidence_level,
            reasoning_steps=len(forecast.reasoning_steps),
            caveats=len(forecast.caveats),
        )

        # Warn if low confidence or coverage
        if forecast.confidence_level == ConfidenceLevel.LOW:
            self.logger.warning(
                "forecast_low_confidence",
                confidence=forecast.confidence_level,
            )

        if timeline.coverage_level.value == "LOW":
            self.logger.warning(
                "forecast_low_coverage",
                coverage=timeline.coverage_level,
                n_events=timeline.total_events,
            )

        return forecast


# Convenience functions

def generate_forecast(
    question: str,
    timeline: Timeline,
    additional_context: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
) -> Forecast:
    """
    Convenience function to generate a forecast.

    Args:
        question: Forecasting question
        timeline: Timeline object
        additional_context: Optional additional context
        llm_client: Optional custom LLM client

    Returns:
        Forecast object
    """
    generator = ForecastGenerator(llm_client=llm_client)
    return generator.generate_forecast(question, timeline, additional_context)


def get_forecast_prompt_template() -> str:
    """
    Get the forecast prompt template for reference.

    Returns:
        Prompt template string
    """
    return ForecastGenerator()._build_forecast_prompt(
        question="<QUESTION>",
        timeline=Timeline(entries=[], coverage_level="ADEQUATE", total_events=0),
        additional_context=None,
    )
