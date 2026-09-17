#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

// Write [2B,7,10,16] in channels-last order, all player-0 rows then player-1.
// T is the native BF16 type; conversion rounds scores exactly once.
template <typename T>
void encode_planes(const uint8_t* cells, const int32_t* scores, size_t batch, T* out) {
    constexpr uint8_t marks[] = {1, 3, 4, 2};
    std::fill(out, out + 2 * batch * 160 * 7, T(0.0f));
    for (size_t r = 0; r < batch; ++r) {
        const auto* board = cells + r * 80;
        const bool opening = std::all_of(board, board + 80, [](uint8_t c) { return c == 0; });
        const T score0(static_cast<float>(scores[2 * r]) / 80.0f);
        const T score1(static_cast<float>(scores[2 * r + 1]) / 80.0f);
        for (size_t y = 0; y < 10; ++y) {
            for (size_t x = 0; x < 16; ++x) {
                const bool playable = x % 2 == y % 2;
                const auto mark = playable ? marks[board[y * 8 + x / 2]] : 0;
                auto* p0 = out + (r * 160 + y * 16 + x) * 7;
                auto* p1 = out + ((batch + r) * 160 + y * 16 + x) * 7;
                if (mark != 1 || !opening || x < 8) p0[mark] = T(1.0f);
                const auto swapped = mark == 3 ? 4 : mark == 4 ? 3 : mark;
                if (mark != 1 || !opening || x >= 8) p1[swapped] = T(1.0f);
                p0[5] = score0; p0[6] = score1;
                p1[5] = score1; p1[6] = score0;
            }
        }
    }
}
