#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

constexpr long long INF = 1e15;

std::vector<long long> best_distance_1d(const std::vector<long long> &arr) {
    int n = arr.size();
    std::vector out(n, INF);
    long long best = INF;

    for (int i = 0; i < n; ++i) {
        long long v = arr[i];
        if (v < best + 1) {
            best = v;
        } else {
            best += 1;
        }
        out[i] = best;
    }

    best = INF;
    for (int i = n - 1; i >= 0; --i) {
        long long v = arr[i];
        if (v < best + 1) {
            best = v;
        } else {
            best += 1;
        }
        if (best < out[i]) {
            out[i] = best;
        }
    }
    return out;
}

void best_distance_2d(std::vector<std::vector<long long> > &grid, int n, int m) {
    for (int r = 0; r < n; ++r) {
        grid[r] = best_distance_1d(grid[r]);
    }

    std::vector<long long> col(n);
    for (int c = 0; c < m; ++c) {
        for (int r = 0; r < n; ++r) {
            col[r] = grid[r][c];
        }
        col = best_distance_1d(col);
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

    std::string s2;
    char prev = '\0';
    for (char ch: s) {
        if (ch != prev) {
            s2 += ch;
            prev = ch;
        }
    }
    s = s2;
    int k = s.length();

    if (k == 0) {
        std::cout << 0 << '\n';
        return 0;
    }

    std::vector<std::vector<std::pair<int, int> > > letter_positions(26);
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
    for (int idx = 1; idx < k; ++idx) {
        int cur_letter = s[idx - 1] - 'a';
        int nxt_letter = s[idx] - 'a';

        for (auto &row: buf) {
            std::fill(row.begin(), row.end(), INF);
        }

        for (const auto &[i, j]: letter_positions[cur_letter]) {
            auto it = dp.find({i, j});
            if (it != dp.end()) {
                long long val = it->second;
                if (val < buf[i][j]) {
                    buf[i][j] = val;
                }
            }
        }

        best_distance_2d(buf, n, m);

        std::unordered_map<std::pair<int, int>, long long, PairHash> new_dp;
        for (const auto &[i, j]: letter_positions[nxt_letter]) {
            new_dp[{i, j}] = buf[i][j];
        }
        dp = std::move(new_dp);
    }

    long long ans = INF;
    for (const auto &[pos, val]: dp) {
        ans = std::min(ans, val);
    }

    std::cout << (ans < INF / 2 ? ans : -1) << '\n';
    return 0;
}
