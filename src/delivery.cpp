#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

constexpr long long INF = static_cast<long long>(4e14);

void best_distance_1d(std::vector<long long> &arr) {
    const int n = static_cast<int>(arr.size());
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

void best_distance_2d(std::vector<std::vector<long long> > &grid) {
    const int n = static_cast<int>(grid.size());
    if (n == 0) {
        return;
    }
    const int m = static_cast<int>(grid[0].size());
    for (int i = 0; i < n; ++i) {
        best_distance_1d(grid[i]);
    }
    std::vector<long long> col(n);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            col[i] = grid[i][j];
        }
        best_distance_1d(col);
        for (int i = 0; i < n; ++i) {
            grid[i][j] = col[i];
        }
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    if (!(std::cin >> n >> m)) {
        return 0;
    }
    int sx, sy;
    std::cin >> sx >> sy;
    --sx;
    --sy;

    std::vector<std::string> grid(n);
    for (int i = 0; i < n; ++i) {
        std::cin >> grid[i];
    }
    std::string s;
    std::cin >> s;

    std::vector dp(n, std::vector(m, INF));
    dp[sx][sy] = 0;

    for (char target: s) {
        best_distance_2d(dp);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] != target) {
                    dp[i][j] = INF;
                }
            }
        }
    }

    long long answer = INF;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            answer = std::min(answer, dp[i][j]);
        }
    }

    std::cout << answer << '\n';
    return 0;
}
