import json

import pytest

from app.config import SAMPLES_DIR
from app.pipeline.categorize import categorize
from app.pipeline.grounding import Grounder
from app.pipeline.scrub import contains_url, scrub_text
from app.pipeline.validators import (GOAL_RE, audit_response, build_goal, clean_step, fix_description,
                                     fix_title, to_title_case, words)
from app.schema import actionCategory as C


@pytest.mark.parametrize("text", ["Visit https://techcorp.com/support.", "Go to www.techcorp.com now.",
                                  "Email kidshome.pin@TechCorp.com.", "See [guide](https://x.io) first."])
def test_scrubber_removes_links(text):
    assert contains_url(text) and not contains_url(scrub_text(text))


def test_title_case():
    assert to_title_case("force a restart") == "Force a Restart"
    assert to_title_case("check USB and Wi-Fi") == "Check USB and Wi-Fi"


def test_goal_pattern():
    assert GOAL_RE.match(build_goal("black screen", "Troubleshooting"))


@pytest.mark.parametrize("desc", ["", "fix it", "It will facilitate secure data transfer between your devices",
                                  "Lets you choose navigation type.", "It will https://x.com help"])
def test_description_always_valid(desc):
    d = fix_description(desc, "Back Up Phone Data")
    assert d.startswith("It will ") and 5 <= len(words(d)) <= 7 and "http" not in d


@pytest.mark.parametrize("title", ["", "Screen", "Blank or black display on a smartphone or tablet"])
def test_title_always_2_to_3_words(title):
    assert 2 <= len(words(fix_title(title, "Display Issue"))) <= 3


def test_clean_step():
    assert clean_step("Please tap on Display") == "Tap on Display."
    assert clean_step("Learn more at our website.") is None


@pytest.mark.parametrize("name,expected", [("Perform a Factory Data Reset", C.critical),
                                           ("Contact Customer Support", C.manual),
                                           ("Adjust Screen Orientation Settings", C.auto)])
def test_categories(name, expected):
    assert categorize(name, ["Open the Quick settings panel."]) == expected


def test_grounding_drops_invented_step():
    g = Grounder("Swipe down to open the Quick settings panel.")
    kept, dropped = g.filter(["Open the Quick settings panel.", "Download the Galaxy Repair Wizard app."])
    assert len(kept) == 1 and "Download" in dropped[0]["step"]


def test_sample_output_matches_schema():
    payload = json.loads((SAMPLES_DIR / "sample_output.json").read_text())
    assert audit_response(payload["response"])["schema_valid"]