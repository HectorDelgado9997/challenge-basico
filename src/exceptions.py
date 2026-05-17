class GlassdoorBaseError(Exception):
    """Base exception for all project-level errors."""


class DataIngestionError(GlassdoorBaseError):
    """
    Raised when data loading or validation fails.

    Examples
    --------
    - CSV file not found
    - Missing required columns
    - Empty DataFrame after loading
    """


class PreprocessingError(GlassdoorBaseError):
    """
    Raised when text preprocessing fails.

    Examples
    --------
    - Missing review_text column
    - Empty DataFrame after normalization
    """


class SentimentAnalysisError(GlassdoorBaseError):
    """
    Raised when sentiment scoring fails.

    Examples
    --------
    - VADER scoring error
    - pysentimiento model loading failure
    """


class EvaluationError(GlassdoorBaseError):
    """
    Raised when evaluation or report generation fails.

    Examples
    --------
    - Missing sentiment columns
    - Empty results DataFrame
    - Output directory creation failure
    """


class MLflowPipelineError(GlassdoorBaseError):
    """
    Raised when the MLflow pipeline fails.

    Examples
    --------
    - Missing or invalid params.yaml
    - MLflow tracking server unreachable
    - Artifact logging failure
    """
