#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

struct Point {
    double x;
    double y;
};

void dfs_size(int v, int parent, const std::vector<std::vector<int>> &graph, std::vector<int> &sz) {
    sz[v] = 1;
    for (int u : graph[v]) {
        if (u == parent) {
            continue;
        }
        dfs_size(u, v, graph, sz);
        sz[v] += sz[u];
    }
}

void assign_points(
        int v,
        int parent,
        std::vector<int> point_ids,
        const std::vector<std::vector<int>> &graph,
        const std::vector<Point> &pts,
        const std::vector<int> &sz,
        std::vector<int> &answer) {
    constexpr double EPS = 1e-9;

    int best = point_ids[0];
    for (int idx : point_ids) {
        if (pts[idx].x < pts[best].x - EPS) {
            best = idx;
        } else if (std::abs(pts[idx].x - pts[best].x) <= EPS && pts[idx].y < pts[best].y) {
            best = idx;
        }
    }

    answer[v] = best;

    std::vector<int> rest;
    rest.reserve(point_ids.size() - 1);
    for (int idx : point_ids) {
        if (idx != best) {
            rest.push_back(idx);
        }
    }

    const Point base = pts[best];
    std::sort(rest.begin(), rest.end(), [&](int lhs, int rhs) {
        double angle_l = std::atan2(pts[lhs].y - base.y, pts[lhs].x - base.x);
        double angle_r = std::atan2(pts[rhs].y - base.y, pts[rhs].x - base.x);
        return angle_l < angle_r;
    });

    int offset = 0;
    for (int u : graph[v]) {
        if (u == parent) {
            continue;
        }
        std::vector<int> subset;
        subset.reserve(sz[u]);
        for (int i = 0; i < sz[u]; ++i) {
            subset.push_back(rest[offset + i]);
        }
        offset += sz[u];
        assign_points(u, v, std::move(subset), graph, pts, sz, answer);
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int t;
    if (!(std::cin >> t)) {
        return 0;
    }

    while (t--) {
        int n;
        std::cin >> n;

        std::vector<std::vector<int>> graph(n);
        for (int i = 0; i < n - 1; ++i) {
            int u, v;
            std::cin >> u >> v;
            --u;
            --v;
            graph[u].push_back(v);
            graph[v].push_back(u);
        }

        std::vector<Point> pts(n);
        for (int i = 0; i < n; ++i) {
            std::cin >> pts[i].x >> pts[i].y;
        }

        std::vector<int> sz(n);
        dfs_size(0, -1, graph, sz);

        std::vector<int> answer(n);
        std::vector<int> all_points(n);
        for (int i = 0; i < n; ++i) {
            all_points[i] = i;
        }

        assign_points(0, -1, all_points, graph, pts, sz, answer);

        for (int i = 0; i < n; ++i) {
            if (i) {
                std::cout << ' ';
            }
            std::cout << answer[i] + 1;
        }
        std::cout << '\n';
    }

    return 0;
}
