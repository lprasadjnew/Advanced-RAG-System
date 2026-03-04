"""Domain validation using Gemini.

Samples text from the START, MIDDLE, and END of the document so that a
document cannot slip through by having an on-topic introduction followed
by completely off-topic body content (or vice-versa).
"""

from google import genai
from config import settings

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)


_PROMPT_TEMPLATE = """\
You are a strict domain classifier.

Allowed domain  : {domain}
Domain purpose  : {description}

Below are three text samples taken from the BEGINNING, MIDDLE, and END of \
the document being evaluated. Each section is clearly labelled.

{samples}

Task: Decide whether this document as a whole belongs to the allowed domain.

Rules:
1. ALL three sections must be consistent with the allowed domain topics.
2. If ANY section clearly belongs to a different, unrelated domain, \
classify the document as UNRELATED.
3. Brief, incidental mentions of other topics are acceptable as long as \
the primary subject throughout all three sections is the allowed domain.

Respond with EXACTLY one word — either RELATED or UNRELATED — followed by \
a pipe "|" and a single short reason (≤15 words) that references the \
overall document content.

Example responses:
RELATED   | All sections consistently discuss health, nutrition, and wellness topics.
UNRELATED | Middle and end sections focus on stock market analysis, not health.
"""


def validate_domain(samples: str) -> tuple[bool, str]:
    """
    Validate that start, middle, and end samples all belong to the domain.

    Args:
        samples: Labelled multi-section text from extract_preview_text().

    Returns:
        (is_valid: bool, reason: str)
    """
    prompt = _PROMPT_TEMPLATE.format(
        domain=settings.DOMAIN,
        description=settings.DOMAIN_DESCRIPTION,
        samples=samples,
    )
    try:
        response = _client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        raw = (response.text or "").strip()
        parts = raw.split("|", 1)
        verdict = parts[0].strip().upper()
        reason = parts[1].strip() if len(parts) > 1 else "No reason provided."
        is_valid = verdict == "RELATED"
        return is_valid, reason
    except Exception as exc:
        return False, f"Validation error: {exc}"
