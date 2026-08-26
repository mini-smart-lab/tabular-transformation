from src.agent.schemas import FinalSubmission


def validate_submission(submission: FinalSubmission) -> None:
    """Revalidate at the Tier-2 boundary, independent of discovery state."""
    FinalSubmission.model_validate(submission.model_dump())
