import unittest

from app.session import InMemorySessionStore


class SessionStoreTests(unittest.TestCase):
    def test_append_and_limit_history(self):
        store = InMemorySessionStore(max_turns_per_session=4)
        session_id = store.create_session()
        store.append_exchange(session_id, "u1", "a1")
        store.append_exchange(session_id, "u2", "a2")
        store.append_exchange(session_id, "u3", "a3")
        history = store.get_history(session_id)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["content"], "u2")

    def test_clear_all(self):
        store = InMemorySessionStore()
        store.create_session()
        store.create_session()
        self.assertEqual(store.clear_all(), 2)
        self.assertEqual(store.size(), 0)


if __name__ == "__main__":
    unittest.main()

