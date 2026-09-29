"""auto / manual / critical classification with rules (deterministic and explainable)."""
import re
from typing import Iterable, Optional

from app.schema import actionCategory

CRITICAL = re.compile(r"factory (data )?reset|reset (all |network )?settings|\bre-?start|\breboot|"
                      r"safe mode|software update|update (the |your )?(device )?software|firmware|"
                      r"recovery (mode|menu)|wipe (the )?cache|erase all|remove the battery|"
                      r"power (it |the device )?off and on", re.I)
MANUAL = re.compile(r"service cent|customer support|support cent|contact (us|support|customer|the)|"
                    r"\brepair|\bclean|\binspect|\bexamine|physical damage|liquid|corrosion|"
                    r"remove (any )?(the )?(case|cover|accessor|screen protector)|charger|charging port|"
                    r"usb (cable|adapter|connection)|mouse|keyboard|monitor|lighting|flashlight|"
                    r"ejector|sim tray|power button|side button|volume (down|up) button|"
                    r"press and hold|place the devices|proof of purchase", re.I)
SETTINGS = re.compile(r"\bsettings?\b|quick settings|\btoggle\b|\bswitch (on|off)\b|\benable\b|"
                      r"\bdisable\b|\bturn (on|off)\b|\bmenu\b|\bapp\b|\bmode\b", re.I)


def categorize(action_name: str, steps: Iterable[str], has_deeplink: Optional[bool] = None) -> actionCategory:
    name = action_name or ""
    body = " ".join(steps or [])
    if CRITICAL.search(name):
        return actionCategory.critical
    if MANUAL.search(name):
        return actionCategory.manual
    if CRITICAL.search(body) and not SETTINGS.search(name):
        return actionCategory.critical
    if has_deeplink:
        return actionCategory.auto
    if SETTINGS.search(body) or SETTINGS.search(name):
        manual_hits = len(MANUAL.findall(body))
        settings_hits = len(SETTINGS.findall(body)) + len(SETTINGS.findall(name))
        return actionCategory.manual if manual_hits > settings_hits else actionCategory.auto
    return actionCategory.manual