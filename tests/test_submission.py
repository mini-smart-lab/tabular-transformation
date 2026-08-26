import pytest
from pydantic import ValidationError

from src.agent.schemas import FinalSubmission


VALID_SCRIPT = "def transform(df):\n    return df.copy()\n"


def test_valid_submission():
    submission = FinalSubmission(
        script=VALID_SCRIPT,
        transformations=[{"column": "x", "operators": ["square"]}],
    )
    assert submission.transformations[0].column == "x"


@pytest.mark.parametrize(
    "payload",
    [
        {"script": "", "transformations": []},
        {"script": "x = 1", "transformations": []},
        {"script": VALID_SCRIPT, "transformations": [{"column": "x", "operators": []}]},
        {"script": "import os\n" + VALID_SCRIPT, "transformations": []},
        {"script": "def transform(df):\n    return __import__('os')\n", "transformations": []},
        {"script": "def transform(df):\n    return df.to_csv('x.csv')\n", "transformations": []},
    ],
)
def test_invalid_submission(payload):
    with pytest.raises(ValidationError):
        FinalSubmission.model_validate(payload)
