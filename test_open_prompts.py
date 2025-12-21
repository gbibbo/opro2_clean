#!/usr/bin/env python3
"""
Test script for open-ended prompts and response parsing.

Tests:
1. Sanitizer accepts open-ended prompts (without SPEECH/NONSPEECH keywords)
2. Normalizer correctly parses various response formats

Usage:
    python test_open_prompts.py
"""

import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.qsm.utils.normalize import normalize_to_binary, llm_fallback_interpret


def sanitize_prompt_candidate(prompt: str) -> tuple[str, bool]:
    """
    Sanitize and validate a prompt candidate.

    Copied from opro_classic_optimize.py to avoid import dependencies.

    Returns:
        (cleaned_prompt, is_valid)
    """
    # Forbidden tokens
    forbidden_tokens = [
        "<|audio_bos|>",
        "<|audio_eos|>",
        "<|AUDIO|>",
        "<audio>",
        "</audio>",
    ]

    cleaned = prompt.strip()

    for token in forbidden_tokens:
        if token in cleaned:
            print(f"      ⚠️  Rejected: Contains forbidden token '{token}'")
            return cleaned, False

    # Check length
    if len(cleaned) < 10:
        print(f"      ⚠️  Rejected: Too short ({len(cleaned)} chars)")
        return cleaned, False

    if len(cleaned) > 300:
        print(f"      ⚠️  Rejected: Too long ({len(cleaned)} chars)")
        return cleaned, False

    # REMOVED: Keyword restriction to allow open-ended prompts
    # The normalize_to_binary() function handles various response formats including:
    # - Binary labels (SPEECH/NONSPEECH)
    # - Yes/No responses
    # - Synonyms (voice, talking, music, noise, etc.)
    # - Open descriptions

    # Remove multiple spaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned, True


def test_sanitizer():
    """Test that sanitizer accepts open-ended prompts."""
    print("=" * 80)
    print("TEST 1: Sanitizer Accepts Open-Ended Prompts")
    print("=" * 80)

    test_prompts = [
        # Open-ended questions
        ("What do you hear in this audio?", True),
        ("Describe the sound.", True),
        ("What type of audio is this?", True),

        # Traditional binary prompts
        ("Is this SPEECH or NONSPEECH?", True),
        ("Classify: SPEECH or NON-SPEECH", True),

        # Edge cases
        ("Audio?", False),  # Too short
        ("x" * 350, False),  # Too long
        ("<|audio_bos|>Test", False),  # Forbidden token
    ]

    passed = 0
    failed = 0

    for prompt, expected_valid in test_prompts:
        display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        cleaned, is_valid = sanitize_prompt_candidate(prompt)

        if is_valid == expected_valid:
            status = "✓ PASS"
            passed += 1
        else:
            status = f"✗ FAIL (expected {expected_valid}, got {is_valid})"
            failed += 1

        print(f"{status}: {display_prompt}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_normalizer():
    """Test that normalizer correctly parses various response formats."""
    print("\n" + "=" * 80)
    print("TEST 2: Normalizer Parses Various Response Formats")
    print("=" * 80)

    test_cases = [
        # Format: (response, expected_label, description)

        # Binary labels
        ("SPEECH", "SPEECH", "Direct SPEECH label"),
        ("NONSPEECH", "NONSPEECH", "Direct NONSPEECH label"),
        ("NON-SPEECH", "NONSPEECH", "Hyphenated NON-SPEECH"),

        # Open-ended descriptions - SPEECH
        ("I hear a person talking", "SPEECH", "Open: person talking"),
        ("This is a human voice speaking", "SPEECH", "Open: human voice"),
        ("Someone is having a conversation", "SPEECH", "Open: conversation"),
        ("The audio contains spoken words", "SPEECH", "Open: spoken words"),
        ("A voice saying something", "SPEECH", "Open: voice"),

        # Open-ended descriptions - NONSPEECH
        ("This is music", "NONSPEECH", "Open: music"),
        ("I hear background noise", "NONSPEECH", "Open: noise"),
        ("It sounds like beeping tones", "NONSPEECH", "Open: beeping"),
        ("There's silence", "NONSPEECH", "Open: silence"),
        ("Just environmental sounds", "NONSPEECH", "Open: environmental"),

        # Yes/No responses
        ("Yes", "SPEECH", "Yes (assuming speech detection)"),
        ("No", "NONSPEECH", "No (assuming speech detection)"),

        # Ambiguous/mixed (should be handled by LLM fallback)
        ("The audio has music and maybe some talking", None, "Mixed content (needs LLM)"),
        ("I hear noise but no voice", "NONSPEECH", "Negated voice"),
    ]

    passed = 0
    failed = 0

    for response, expected_label, description in test_cases:
        label, confidence = normalize_to_binary(response)

        if label == expected_label:
            status = "✓ PASS"
            passed += 1
        else:
            status = f"✗ FAIL (expected {expected_label}, got {label})"
            failed += 1

        print(f"{status}: {description}")
        print(f"  Response: '{response}'")
        print(f"  Parsed: {label} (confidence: {confidence:.2f})")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_edge_cases():
    """Test edge cases and potential failure modes."""
    print("=" * 80)
    print("TEST 3: Edge Cases")
    print("=" * 80)

    test_cases = [
        # Responses that should return None (unparseable)
        ("Blue", None, "Completely unrelated word"),
        ("42", None, "Number"),
        ("", None, "Empty string"),

        # Ambiguous responses
        ("Maybe", None, "Ambiguous answer"),
        ("I'm not sure", None, "Uncertain response"),
        ("Could be either", None, "Non-committal"),
    ]

    passed = 0
    failed = 0

    for response, expected_label, description in test_cases:
        label, confidence = normalize_to_binary(response)

        if label == expected_label:
            status = "✓ PASS"
            passed += 1
        else:
            status = f"✗ FAIL (expected {expected_label}, got {label})"
            failed += 1

        print(f"{status}: {description}")
        print(f"  Response: '{response}'")
        print(f"  Parsed: {label} (confidence: {confidence:.2f})")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_llm_fallback():
    """Test LLM fallback for ambiguous responses."""
    print("\n" + "=" * 80)
    print("TEST 4: LLM Fallback for Ambiguous Responses")
    print("=" * 80)

    test_cases = [
        # Cases that should use fallback (mixed content)
        ("The audio has music and maybe some talking", "SPEECH", "Mixed: music + talking"),
        ("Background music with someone speaking", "SPEECH", "Mixed: music + speaking"),
        ("I hear noise but no voice", "NONSPEECH", "Negated voice"),
        ("Just instrumental sounds, no speech", "NONSPEECH", "Negated speech"),

        # Cases the fallback should handle well
        ("Someone is talking over background music", "SPEECH", "Speech over music"),
        ("Pure instrumental track", "NONSPEECH", "Pure instrumental"),
        ("A narrator is describing something", "SPEECH", "Narrator"),
        ("Environmental sounds from nature", "NONSPEECH", "Environmental"),
    ]

    passed = 0
    failed = 0

    for response, expected_label, description in test_cases:
        # First try normalize
        label, conf = normalize_to_binary(response)

        # If normalize fails, use fallback
        if label is None:
            label, conf = llm_fallback_interpret(response)

        if label == expected_label:
            status = "✓ PASS"
            passed += 1
        else:
            status = f"✗ FAIL (expected {expected_label}, got {label})"
            failed += 1

        print(f"{status}: {description}")
        print(f"  Response: '{response}'")
        print(f"  Parsed: {label} (confidence: {conf:.2f})")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    """Run all tests."""
    print("\nOPEN-ENDED PROMPTS TEST SUITE")
    print("=" * 80)
    print("Testing modifications to allow open-ended prompts in OPRO")
    print("=" * 80)
    print()

    results = {
        "Sanitizer": test_sanitizer(),
        "Normalizer": test_normalizer(),
        "Edge Cases": test_edge_cases(),
        "LLM Fallback": test_llm_fallback(),
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        all_passed = all_passed and passed

    print("=" * 80)

    if all_passed:
        print("\n✓ ALL TESTS PASSED - Ready for Opción B (LLM fallback)")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Review failures before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
