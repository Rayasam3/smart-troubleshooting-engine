"""Extraction prompt. The LLM only reads and restructures text; deeplinks, categories and ordering are code."""
import json

EXTRACTION_SYSTEM = """You are a TechCorp customer-care engineer. You turn ONE knowledge-base article into a structured troubleshooting plan for ONE customer complaint.

Return ONLY a JSON object, no prose, no markdown, with exactly these keys:
{
  "no_match": boolean,
  "relevance": number,            // 0.0-1.0: how well the article addresses the complaint
  "topic": string,                // 1-4 words, Title Case, the core issue, e.g. "Black Screen"
  "goal_kind": "Troubleshooting" | "Configuration",
  "title": string,                // 2-3 words, sentence case, e.g. "Black screen recovery"
  "normalized_query": string,     // the complaint as a short canonical technical query
  "query_variations": [string],   // 8-10 distinct paraphrases of the complaint
  "actions": [
    {"actionName": string, "description": string, "steps": [string]}
  ]
}

GROUNDING RULES (most important):
- Use ONLY instructions that appear in the article. Never add steps from your own knowledge.
- Keep the article's wording for menu and option names (e.g. "Quick settings panel", "Factory data reset").
- Skip background explanations, marketing text, and anything that is not an instruction.
- If the article does not address the complaint, or contains no actionable instruction, return "no_match": true and "actions": [].
- Stay close to the article's wording. Do not add implied or follow-up steps the article does not state (e.g. "Release the buttons.").
- If only part of the article helps with the complaint, extract just those parts. Use no_match only when nothing in the article would help.

ACTION RULES:
- One action = one screen or one physical task. Steps on the same screen belong to the same action.
- Keep actions in the order the article gives them. Include restarts, updates and resets only if the article contains them.
- actionName: Title Case verb phrase, 2-5 words, e.g. "Adjust Screen Orientation Settings".
- description: MUST start with "It will", 5 to 7 words total, plain benefit, e.g. "It will let you choose navigation type".
- steps: imperative sentences, ONE interaction per step, ending with a period, e.g. "Tap on Display.".

HARD BANS:
- No URLs, web addresses, email addresses, or phrases like "visit our website" / "learn more".
- No deeplinks or URIs of any kind.

QUERY VARIATIONS: mix registers - formal, casual, keyword-only, frustrated, question form, and one with realistic typos. All must keep the same meaning and be different from each other."""

FEW_SHOT_OUTPUT = {
    "no_match": False,
    "relevance": 0.93,
    "topic": "Swipe Navigation",
    "goal_kind": "Troubleshooting",
    "title": "Swipe navigation settings",
    "normalized_query": "swipe navigation direction wrong after app install",
    "query_variations": [
        "Why does my phone swipe vertically when I try to swipe sideways after downloading an app?",
        "phone swipe gestures wrong direction after app install",
        "My phone's gesture navigation got messed up by a new app and swipes go the wrong way.",
        "What should I do when swiping left or right on my phone scrolls the screen up and down instead?",
        "Swipe navigation broken after installing app.",
        "This is so annoying, I can't swipe sideways anymore since installing that app!",
        "Kindly advise how to restore horizontal swipe navigation after an application install.",
        "swipe nav gone wierd after new app, scrols up down insted",
    ],
    "actions": [{
        "actionName": "Configure Navigation Bar Settings",
        "description": "It will let you choose navigation type",
        "steps": [
            "Navigate to and open Settings.",
            "Tap on Display.",
            "Tap on Navigation bar.",
            "Select your preferred navigation type between Buttons and Swipe gestures.",
            "Optionally toggle on Gesture hint to display guidance lines at the bottom of the screen.",
        ],
    }],
}


def build_user_message(query: str, title: str, article: str) -> str:
    return (f"CUSTOMER COMPLAINT:\n{query}\n\nARTICLE TITLE: {title}\n\nARTICLE:\n<<<\n{article}\n>>>\n\n"
            "Return the JSON object now.")


def few_shot_messages() -> list:
    return [
        {"role": "user", "content": "EXAMPLE of a correct output for a swipe-navigation complaint "
                                    "(article omitted for brevity). Match this style exactly."},
        {"role": "assistant", "content": json.dumps(FEW_SHOT_OUTPUT)},
    ]