"""
Prompt builder for text-classification datasets.

Owns PROMPT_TEMPLATES (previously in src/config.py) and the prompt-rendering
logic previously in src/generate.py (_format_prompt, build_prompts).

src/config.py re-exports PROMPT_TEMPLATES from here for backward
compatibility with code that does ``from src.config import PROMPT_TEMPLATES``.
src/generate.py re-imports build_prompts from here for backward compatibility.
"""

from typing import List, Optional, Tuple

from src.prompts.base import PromptBuilder


# ---------------------------------------------------------------------------
# Prompt templates
#
# Moved here from src/config.py.  Each entry specifies:
#   user_message    – instruction with {label} and {example} placeholders
#   response_prefix – the opening tokens of the model's response that the
#                     generated text should continue from
#   public_seed     – a short generic (non-sensitive) example for the public
#                     prompt; does not affect the privacy guarantee
#   labels          – mapping from integer label to human-readable name
#
# Reference: Appendix F of Amin et al. (2024).
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES = {
    "agnews": {
        "user_message": (
            "Here are texts with News Type: {label}.\n\n"
            "Text: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Text:",
        "public_seed": "Officials announced new policy changes effective immediately.",
        "labels": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    },
    "trec": {
        "user_message": (
            "Here are questions with Answer Type: {label}.\n\n"
            "```\nText: {example}\n```\n\n"
            "Please give me another one."
        ),
        "response_prefix": "```\nQuestion:",
        "public_seed": "What is the capital of France?",
        "labels": {
            0: "Abbreviation", 1: "Entity", 2: "Description",
            3: "Human", 4: "Location", 5: "Number",
        },
    },
    "dbpedia": {
        "user_message": (
            "Here are entries of Category: {label}.\n\n"
            "Entry: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Entry:",
        "public_seed": (
            "The Springfield Institute is a research organization "
            "founded in 1952 and based in the United States."
        ),
        "labels": {
            0: "Company", 1: "School", 2: "Artist", 3: "Athlete",
            4: "Politician", 5: "Transportation", 6: "Building",
            7: "Nature", 8: "Village", 9: "Animal", 10: "Plant",
            11: "Album", 12: "Film", 13: "Book",
        },
    },
    "imdb": {
        "user_message": (
            "Here are texts with Sentiment: {label}.\n\n"
            "Text: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Text:",
        "public_seed": (
            "This film was a solid effort with strong performances "
            "from the entire cast."
        ),
        "labels": {0: "Negative", 1: "Positive"},
    },
    "yelp": {
        "user_message": (
            "Here are texts with Sentiment: {label}.\n\n"
            "Text: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Text:",
        "public_seed": "The service was prompt and the food arrived fresh.",
        "labels": {0: "Negative", 1: "Positive"},
    },
}


# ---------------------------------------------------------------------------
# Internal helper (unchanged from src/generate.py::_format_prompt)
# ---------------------------------------------------------------------------

def _format_prompt(tokenizer, user_content: str, response_prefix: str) -> str:
    """Wrap a user message in the model's native chat template.

    Uses tokenizer.apply_chat_template so that instruction-tuned models
    receive proper role markers instead of raw text markers.

    The leading <bos> emitted by some templates is stripped because the
    tokenizer already adds one during tokenization.
    """
    messages = [{"role": "user", "content": user_content}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if formatted.startswith("<bos>"):
        formatted = formatted[len("<bos>"):]
    return formatted + response_prefix


# ---------------------------------------------------------------------------
# Standalone function (backward-compatible; used directly by generate.py)
# ---------------------------------------------------------------------------

def build_prompts(
    examples: List[dict],
    dataset_name: str,
    text_column: str,
    label: int,
    tokenizer=None,
) -> Tuple[List[str], str]:
    """Build private prompts for a batch plus the corresponding public prompt.

    This is the same function previously defined in src/generate.py.
    It is preserved here with an identical signature for backward compatibility.

    When *tokenizer* is provided the prompts are wrapped in the model's
    chat template.  This is strongly recommended for instruction-tuned
    models so that the model properly enters response mode.

    Args:
        examples: batch of example dicts.
        dataset_name: key in PROMPT_TEMPLATES.
        text_column: key for the text field in each dict.
        label: integer label for this batch.
        tokenizer: optional HuggingFace tokenizer.

    Returns:
        (private_prompts, public_prompt)
    """
    templates = PROMPT_TEMPLATES[dataset_name]
    label_name = templates["labels"][label]
    user_msg_tpl = templates["user_message"]
    response_prefix = templates["response_prefix"]

    private_prompts = []
    for ex in examples:
        user_content = user_msg_tpl.format(
            label=label_name, example=ex[text_column],
        )
        if tokenizer is not None:
            prompt = _format_prompt(tokenizer, user_content, response_prefix)
        else:
            prompt = f"# [User]\n{user_content}\n\n# [Assistant]\n{response_prefix}"
        private_prompts.append(prompt)

    public_seed = templates.get("public_seed", "")
    public_user = user_msg_tpl.format(label=label_name, example=public_seed)
    if tokenizer is not None:
        public_prompt = _format_prompt(tokenizer, public_user, response_prefix)
    else:
        public_prompt = f"# [User]\n{public_user}\n\n# [Assistant]\n{response_prefix}"

    return private_prompts, public_prompt


# ---------------------------------------------------------------------------
# Class-based interface
# ---------------------------------------------------------------------------

class TextClassificationPromptBuilder(PromptBuilder):
    """PromptBuilder for text-classification datasets.

    Wraps build_prompts() in the PromptBuilder interface so callers can
    program to the abstract type.  Internally delegates to the standalone
    function so the behavior is identical.
    """

    def __init__(self, dataset_name: str) -> None:
        if dataset_name not in PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Available: {sorted(PROMPT_TEMPLATES)}"
            )
        self.dataset_name = dataset_name
        self._template = PROMPT_TEMPLATES[dataset_name]

    @property
    def label_names(self):
        return self._template["labels"]

    @property
    def response_prefix(self) -> str:
        return self._template["response_prefix"]

    def build_prompts(
        self,
        examples: List[dict],
        text_column: str,
        label: int,
        tokenizer=None,
    ) -> Tuple[List[str], str]:
        return build_prompts(
            examples, self.dataset_name, text_column, label, tokenizer,
        )
