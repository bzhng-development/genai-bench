import os
from pathlib import Path
import datetime

from transformers import PreTrainedTokenizer

def convert_latency_value(
    value_seconds: float | None, target_unit: str = "seconds"
) -> float | None:
    """
    this fn converts latency values between seconds and milliseconds.
    """
    if value_seconds is None:
        return None

    if target_unit == "seconds":
        return datetime.timedelta(seconds=value_seconds).total_seconds()
    elif target_unit == "milliseconds":
        return (
            datetime.timedelta(seconds=value_seconds).total_seconds() * 1000
        )  # maybe simpler but suffices.
    else:
        raise ValueError(f"Unsupported target unit: {target_unit}")


def convert_label(label: str, target_unit: str = "seconds") -> str:
    """
    Convert a latency label to the specified unit.
    """
    # assume that all original labels are in seconds
    # so no need to handle the miliseconds -> seconds conversion
    if target_unit == "seconds":
        return label
    elif target_unit == "milliseconds":
        return (
            label.replace(" (s)", " (ms)")
            .replace(" (s)", " (ms)")
            .replace(" seconds", " milliseconds")
            .replace(" seconds", " milliseconds")
        )
        # TODO: add more replacements if needed AND manually check the labels
    else:
        raise ValueError(f"Unsupported target unit: {target_unit}")

def sanitize_string(input_str: str):
    """
    Sanitize a string to be used in filenames by replacing problematic
    characters.
    """
    return (
        input_str.replace("/", "_").replace(",", "_").replace("(", "").replace(")", "")
    )


def is_single_experiment_folder(folder_name: str) -> bool:
    """
    Checks whether the folder contains only one experiment by inspecting
    whether the folder has files or subfolders.

    Returns True if it appears to be a single experiment folder, False
    otherwise.
    """
    # Check if the folder has any subdirectories (which would indicate
    # multiple experiments)
    subfolders = [
        f
        for f in os.listdir(folder_name)
        if os.path.isdir(os.path.join(folder_name, f))
    ]
    return len(subfolders) == 0


def calculate_sonnet_char_token_ratio(tokenizer: PreTrainedTokenizer) -> float:
    """Calculate the ratio of character to token using model tokenizer."""
    sonnet_file = Path(__file__).parent.resolve() / "data/sonnet.txt"
    with open(sonnet_file, "r") as f:
        content = f.read()

    total_chars = len(content)
    tokens = tokenizer.encode(content, add_special_tokens=False)
    total_tokens = len(tokens)

    char_token_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    return char_token_ratio
