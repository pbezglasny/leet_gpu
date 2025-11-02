#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

constexpr long long INF = 1e15;

void best_distance_1d(std::vector<long long> &arr) {
    int n = arr.size();
    long long best = INF;

    for (int i = 0; i < n; ++i) {
        best = std::min(arr[i], best + 1);
        arr[i] = best;
    }

    best = INF;
    for (int i = n - 1; i >= 0; --i) {
        best = std::min(arr[i], best + 1);
        arr[i] = best;
    }
}

void best_distance_2d(std::vector<std::vector<long long> > &grid, int n, int m) {
    // Process rows in-place
    for (int r = 0; r < n; ++r) {
        best_distance_1d(grid[r]);
    }

    // Process columns
    std::vector<long long> col(n);
    for (int c = 0; c < m; ++c) {
        for (int r = 0; r < n; ++r) {
            col[r] = grid[r][c];
        }
        best_distance_1d(col);
        for (int r = 0; r < n; ++r) {
            grid[r][c] = col[r];
        }
    }
}

struct PairHash {
    std::size_t operator()(const std::pair<int, int> &p) const {
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};

int main() {
    int n, m;
    std::cin >> n >> m;

    int sx, sy;
    std::cin >> sx >> sy;
    --sx;
    --sy;

    std::vector<std::string> lines(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> lines[i];
    }

    std::string s;
    std::cin >> s;

    // Deduplicate consecutive characters in-place
    if (!s.empty()) {
        int write_pos = 1;
        for (int i = 1; i < s.length(); ++i) {
            if (s[i] != s[i - 1]) {
                s[write_pos++] = s[i];
            }
        }
        s.resize(write_pos);
    }
    int k = s.length();

    if (k == 0) {
        std::cout << 0 << '\n';
        return 0;
    }

    std::vector<std::vector<std::pair<int, int> > > letter_positions(26);
    // Reserve space to avoid reallocations (approximate)
    for (auto &vec : letter_positions) {
        vec.reserve((n * m) / 26 + 1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            letter_positions[lines[i][j] - 'a'].emplace_back(i, j);
        }
    }

    std::vector<std::vector<long long> > src(n, std::vector<long long>(m, INF));
    src[sx][sy] = 0;
    best_distance_2d(src, n, m);

    int first = s[0] - 'a';
    std::unordered_map<std::pair<int, int>, long long, PairHash> dp;
    for (const auto &[i, j]: letter_positions[first]) {
        dp[{i, j}] = src[i][j];
    }

    std::vector<std::vector<long long> > buf(n, std::vector<long long>(m, INF));
    std::unordered_map<std::pair<int, int>, long long, PairHash> next_dp;

    for (int idx = 1; idx < k; ++idx) {
        int cur_letter = s[idx - 1] - 'a';
        int nxt_letter = s[idx] - 'a';

        // Reset buffer
        for (auto &row: buf) {
            std::fill(row.begin(), row.end(), INF);
        }

        // Fill buffer with current positions
        for (const auto &[i, j]: letter_positions[cur_letter]) {
            auto it = dp.find({i, j});
            if (it != dp.end() && it->second < buf[i][j]) {
                buf[i][j] = it->second;
            }
        }

        best_distance_2d(buf, n, m);

        // Reuse next_dp instead of creating new map
        next_dp.clear();
        next_dp.reserve(letter_positions[nxt_letter].size());
        for (const auto &[i, j]: letter_positions[nxt_letter]) {
            next_dp[{i, j}] = buf[i][j];
        }
        std::swap(dp, next_dp);
    }

    long long ans = INF;
    for (const auto &[pos, val]: dp) {
        ans = std::min(ans, val);
    }

    std::cout << (ans < INF / 2 ? ans : -1) << '\n';
    return 0;
}
