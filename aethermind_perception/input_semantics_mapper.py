# input_semantics_mapper.py
def map_raw_to_semantic_action(raw):
    key_map = {
        "W":"move_forward","A":"move_left","S":"move_backward","D":"move_right",
        "E":"interact","SPACE":"jump","I":"inventory",
    }
    mouse_map = {"left":"click_primary","right":"click_secondary","middle":"click_middle"}
    action = None; valid = False
    if isinstance(raw.get("keys"), list) and raw["keys"]:
        # choose first mapped, or None
        for k in raw["keys"]:
            if k in key_map: action = key_map[k]; valid = True; break
    if not valid and isinstance(raw.get("mouse",{}).get("buttons"), dict):
        for btn, down in raw["mouse"]["buttons"].items():
            if down and btn in mouse_map: action = mouse_map[btn]; valid = True; break
    return {"action": action, "valid_for_game": bool(valid)}

def _normalize_mouse(raw, resolution):
    w,h = resolution
    x,y = raw.get("mouse",{}).get("position",[0,0])
    return [max(0,min(1,x/float(max(1,w)))), max(0,min(1,y/float(max(1,h))))]

def process_actions(raw_actions, resolution=(1920,1080)):
    out = []
    for a in raw_actions:
        sem = map_raw_to_semantic_action(a)
        a2 = {
            "ts": a["ts"],
            "raw": a,
            "semantic": {**sem, "mouse_norm": _normalize_mouse(a, resolution)}
        }
        out.append(a2)
    return out
