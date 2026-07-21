from __future__ import annotations

import socket
import unittest

from cyrus.security import OfflineViolation, offline_guard


class OfflinePolicyTests(unittest.TestCase):
    def test_outbound_connection_is_blocked_before_network_use(self) -> None:
        with offline_guard():
            with self.assertRaises(OfflineViolation):
                socket.create_connection(("127.0.0.1", 9), timeout=0.01)
            connection = socket.socket()
            try:
                with self.assertRaises(OfflineViolation):
                    connection.connect(("127.0.0.1", 9))
            finally:
                connection.close()

    def test_guard_restores_socket_api(self) -> None:
        original = socket.create_connection
        with offline_guard():
            self.assertIsNot(socket.create_connection, original)
        self.assertIs(socket.create_connection, original)


if __name__ == "__main__":
    unittest.main()

