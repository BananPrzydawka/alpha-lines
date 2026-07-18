    def export_state(self, index: int) -> str:
        """Export game at index to a portable string: '{height}x{width}:{base64}'."""
        board = self.boards[index]
        values = []
        for r in range(height):
            for c in range(width):
                if (r + c) % 2 == 0:
                    values.append(board[r, c].item() - 1)  # 1..4 → 0..3

        packed = bytearray((len(values) + 3) // 4)
        for i, v in enumerate(values):
            packed[i // 4] |= (v & 0x3) << (2 * (i % 4))

        encoded = _base64.b64encode(bytes(packed)).rstrip(b"=").decode()
        return f"{height}x{width}:{encoded}"

    @classmethod
    def import_state(cls, code: str, device: str = "cpu", number: int = 1) -> "batched_lines_game":
        header, encoded = code.split(":")
        h, w = map(int, header.split("x"))
        assert h == height and w == width, f"Size mismatch: got {h}x{w}, expected {height}x{width}"

        pad = "=" * (-len(encoded) % 4)
        packed = _base64.b64decode(encoded + pad)

        raw = []
        for byte in packed:
            for shift in (0, 2, 4, 6):
                raw.append((byte >> shift) & 0x3)

        game = cls(num_games=number, device=device)
        idx = 0
        for r in range(h):
            for c in range(w):
                if (r + c) % 2 == 0:
                    game.boards[:, r, c] = raw[idx] + 1  # broadcast to all copies
                    idx += 1

        game._update_scores()

        non_initial = (
            (game.boards[0] != cls.PLAYABLE_SQUARE)
            & (game.boards[0] != cls.NON_PLAYABLE_SQUARE)
        )
        game.move_counts[:] = 1 if non_initial.any() else 0
        game.finished[:] = not (game.boards[0] == cls.PLAYABLE_SQUARE).any()

        return game