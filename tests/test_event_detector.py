
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from aethermind_perception.event_detector import detect_events

def test_detect_events_on_all_sessions():
    sessions_root = os.path.join(os.path.dirname(__file__), '../chunks')
    for session_name in os.listdir(sessions_root):
        session_dir = os.path.join(sessions_root, session_name)
        if not os.path.isdir(session_dir):
            continue
        session_json = os.path.join(session_dir, 'session.json')
        if not os.path.exists(session_json):
            continue
        events = detect_events(session_dir)
        assert isinstance(events, list), f"Output for {session_dir} is not a list"
        assert len(events) > 0, f"No events found for {session_dir}"
        for event in events:
            assert isinstance(event, dict), f"Event is not a dict in {session_dir}"
            assert 'event_score' in event, f"Missing 'event_score' in {session_dir}"
            assert 'is_event' in event, f"Missing 'is_event' in {session_dir}"
            assert isinstance(event['event_score'], float), f"'event_score' not float in {session_dir}"
            assert isinstance(event['is_event'], bool), f"'is_event' not bool in {session_dir}"


if __name__ == "__main__":
    try:
        test_detect_events_on_all_sessions()
        print("All tests passed.")
    except AssertionError as e:
        print(f"Test failed: {e}")
        exit(1)
