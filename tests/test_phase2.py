"""Phase 2: deeplink mapping + ordering."""
import pytest

from app.data_loader import catalog_uris, paired_inputs
from app.pipeline.deeplink_index import DeeplinkIndex
from app.pipeline.deeplink_mapper import DeeplinkMapper, name_match
from app.pipeline.extract import extract
from app.pipeline.extract_rules import RuleExtractor
from app.pipeline.ordering import order_actions
from app.pipeline.pipeline import finalize
from app.pipeline.validators import audit_response
from app.retrieval.embedder import Embedder
from app.schema import Action, Goal, StepGroup


@pytest.fixture(scope="module")
def mapper():
    return DeeplinkMapper(DeeplinkIndex(Embedder("tfidf")))


@pytest.mark.parametrize("target,name", [
    ("Super steady mode", "Easy mode"), ("Factory data reset", "Auto factory reset"),
    ("Navigation bar", "Hide status and navigation bars"),
    ("Navigation bar", "Show input method button on navigation bar"),
    ("Display", "Privacy display"), ("Screen lock and biometrics", "Lock screen"),
    ("Quick settings panel", "Quick Access panels"), ("More", "More options"),
])
def test_decoys_rejected(target, name):
    assert name_match(target, name) == 0.0


@pytest.mark.parametrize("target,name", [
    ("Navigation bar", "Navigation bar"), ("Back up data", "Back up data"),
    ("Touch sensitivity setting", "Touch sensitivity"), ("Dark mode", "Dark mode settings"),
])
def test_real_matches_accepted(target, name):
    assert name_match(target, name) >= 0.6


def _act(name, steps, cat="auto"):
    return Action(actionName=name, description="It will help you fix it",
                  stepGroups=[StepGroup(steps=steps)], category=cat)


@pytest.mark.parametrize("name,steps,expected_id,expected_value", [
    ("Configure Navigation Bar", ["Open Settings.", "Tap on Display.", "Tap on Navigation bar."], "DL-0169", None),
    ("Back Up Phone Data", ["Open Settings.", "Tap Accounts and backup.", "Select Back up data."], "DL-0542", "True"),
    ("Increase Touch Sensitivity", ["Open Settings.", "Tap Display.", "Turn on Touch sensitivity."], "DL-0126", "True"),
    ("Protection Off", ["Open Settings.", "Turn off Accidental touch protection."], "DL-0216", "False"),
    ("Enable Touch Sensitivity", ["Open Settings.", "Tap Display.",
                                  "Toggle the switch next to Touch sensitivity on."], "DL-0126", "True"),
    ("Turn Off Touch Sensitivity", ["Open Settings.", "Tap Display.",
                                    "Toggle the switch next to Touch sensitivity off."], "DL-0125", "False"),
])
def test_exact_screen_and_direction(mapper, name, steps, expected_id, expected_value):
    action, decision = mapper.map_action(_act(name, steps))
    assert decision.chosen_id == expected_id
    val = action.stepGroups[0].validationDeeplink
    assert (val.value if val else None) == expected_value


def test_backup_matches_official_sample(mapper):
    action, _ = mapper.map_action(_act("Back Up Phone Data", ["Open Settings.", "Select Back up data."]))
    assert action.stepGroups[0].actionableDeeplink.deeplink == "voiceassist://masked/act/b3ed3ed663"


def test_missing_screen_gets_placeholder_with_real_name(mapper):
    action, d = mapper.map_action(_act("Adjust Screen Orientation", [
        "Swipe down to open the Quick settings panel.", "Tap the Auto rotate icon to enable it."]))
    link = action.stepGroups[0].actionableDeeplink
    assert d.kind == "dummy" and link.deeplink == "voiceassist://dummy_positive"
    assert "Auto rotate" in link.message and 5 <= len(link.message.split()) <= 7


def test_manual_never_gets_link(mapper):
    action, _ = mapper.map_action(_act("Contact Customer Support", ["Open Settings and call support."], "manual"))
    assert action.stepGroups[0].actionableDeeplink is None


def test_ordering_least_disruptive_first():
    g = Goal(goal="x", title="a b", score=1, actions=[
        _act("Perform a Factory Data Reset", ["x y."], "critical"), _act("Contact Customer Support", ["x y."], "manual"),
        _act("Restart Your Device", ["x y."], "critical"), _act("Check for Damage", ["x y."], "manual"),
        _act("Adjust Screen Timeout", ["x y."], "auto")])
    names = [a.actionName for a in order_actions(g).actions]
    assert names == ["Adjust Screen Timeout", "Check for Damage", "Contact Customer Support",
                     "Restart Your Device", "Perform a Factory Data Reset"]


@pytest.mark.parametrize("query,siis", paired_inputs())
def test_full_offline_pipeline_has_zero_rule_errors(mapper, query, siis):
    ext = extract(query, siis, RuleExtractor())
    if ext.goal is None:
        return
    goal, _ = finalize(ext.goal, mapper)
    report = audit_response({"contexts": [goal.model_dump(mode="json")]}, set(catalog_uris()))
    assert report["schema_valid"]
    assert [i for i in report["issues"] if i["severity"] == "error"] == []