"""
Remove Highly Correlated Variates

This script reads the correlation matrix from a _summary.json file and uses a greedy
algorithm to determine which variates to remove so that no remaining pair has
correlation above a specified threshold.

Algorithm:
1. Build a graph where edges connect variates with |correlation| > threshold
2. Use a greedy approach: repeatedly remove the node with highest degree
3. Priority: prefer to remove nodes that are already marked as `kept=0` (recommended for deletion)

Usage:
    python -m timebench.remove_correlated --summary_path <path_to_summary.json> [--threshold 0.95] [--dry_run]
"""

import argparse
import json
import re
from collections import defaultdict


def load_summary(summary_path: str) -> dict:
    """Load the _summary.json file"""
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_var_suffix(var_name: str) -> str:
    """Remove suffix tags like [rw], [sp], [drop] from variable name"""
    return re.sub(r'\[.*?\]', '', var_name)


def build_high_corr_graph(
    correlation_matrix: dict,
    variates_info: dict,
    threshold: float = 0.95
) -> tuple[dict[str, set], dict[tuple, float]]:
    """
    Build a graph where edges connect variates with |correlation| > threshold

    Returns:
        adjacency: {variate: set of connected variates}
        edge_corr: {(var1, var2): correlation_value}
    """
    adjacency = defaultdict(set)
    edge_corr = {}

    # Get all variate names from correlation matrix
    variates = list(correlation_matrix.keys())

    for i, var1 in enumerate(variates):
        base_var1 = strip_var_suffix(var1)
        for var2 in variates[i+1:]:
            base_var2 = strip_var_suffix(var2)

            # Get correlation value
            corr_val = correlation_matrix.get(var1, {}).get(var2)
            if corr_val is None:
                corr_val = correlation_matrix.get(var2, {}).get(var1)

            if corr_val is not None and abs(corr_val) > threshold:
                adjacency[base_var1].add(base_var2)
                adjacency[base_var2].add(base_var1)
                pair = tuple(sorted([base_var1, base_var2]))
                edge_corr[pair] = corr_val

    return dict(adjacency), edge_corr


def get_variate_priority(var_name: str, variates_info: dict) -> int:
    """
    Get priority for removing a variate (higher = more likely to remove)

    Priority factors:
    - kept=0 (recommended for deletion): +100
    - Higher degree in correlation graph: handled separately
    """
    base_var = strip_var_suffix(var_name)

    # Check in variates_info (may have suffix or not)
    for key, info in variates_info.items():
        base_key = strip_var_suffix(key)
        if base_key == base_var:
            if info.get("kept", 1) == 0:
                return 100  # High priority to remove
            return 0

    return 0


def greedy_remove_high_corr(
    adjacency: dict[str, set],
    edge_corr: dict[tuple, float],
    variates_info: dict
) -> list[str]:
    """
    Greedy algorithm to find minimum set of variates to remove
    to eliminate all high-correlation pairs

    Strategy:
    1. Score each node: priority + degree
    2. Remove node with highest score
    3. Repeat until no edges remain
    """
    # Make a copy of adjacency to modify
    adj = {k: set(v) for k, v in adjacency.items()}
    removed = []

    while True:
        # Find nodes that still have edges
        nodes_with_edges = [n for n, neighbors in adj.items() if len(neighbors) > 0]

        if not nodes_with_edges:
            break

        # Score each node: priority + degree
        scores = {}
        for node in nodes_with_edges:
            priority = get_variate_priority(node, variates_info)
            degree = len(adj[node])
            # Score = priority * 1000 + degree (priority dominates)
            scores[node] = priority * 1000 + degree

        # Find node with highest score
        node_to_remove = max(nodes_with_edges, key=lambda n: scores[n])
        removed.append(node_to_remove)

        # Remove this node and all its edges
        neighbors = adj.pop(node_to_remove, set())
        for neighbor in neighbors:
            if neighbor in adj:
                adj[neighbor].discard(node_to_remove)

    return removed


def analyze_correlation_clusters(
    adjacency: dict[str, set],
    edge_corr: dict[tuple, float]
) -> list[set]:
    """
    Find connected components (clusters) in the high-correlation graph
    """
    visited = set()
    clusters = []

    def dfs(node: str, cluster: set):
        if node in visited:
            return
        visited.add(node)
        cluster.add(node)
        for neighbor in adjacency.get(node, []):
            dfs(neighbor, cluster)

    for node in adjacency:
        if node not in visited:
            cluster = set()
            dfs(node, cluster)
            if cluster:
                clusters.append(cluster)

    return sorted(clusters, key=len, reverse=True)


def main():
    parser = argparse.ArgumentParser(
        description="Remove highly correlated variates from dataset"
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        required=True,
        help="Path to _summary.json file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Correlation threshold (default: 0.95)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would be removed, don't generate commands"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default=None,
        help="Target directory for remove commands (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Load summary
    print(f"Loading summary from: {args.summary_path}")
    summary = load_summary(args.summary_path)

    # Get correlation matrix
    correlation_matrix = summary.get("correlation_matrix", {})
    if not correlation_matrix:
        print("❌ No correlation_matrix found in summary file")
        return

    variates_info = summary.get("variates", {})

    print(f"Found {len(correlation_matrix)} variates in correlation matrix")
    print(f"Correlation threshold: |r| > {args.threshold}")

    # Build high-correlation graph
    adjacency, edge_corr = build_high_corr_graph(
        correlation_matrix, variates_info, args.threshold
    )

    total_high_corr_pairs = len(edge_corr)
    nodes_with_high_corr = len(adjacency)

    print(f"\n{'='*60}")
    print(f"High Correlation Analysis")
    print(f"{'='*60}")
    print(f"Total high-correlation pairs (|r| > {args.threshold}): {total_high_corr_pairs}")
    print(f"Variates involved in high-correlation: {nodes_with_high_corr}")

    if total_high_corr_pairs == 0:
        print("✅ No high-correlation pairs found. No action needed.")
        return

    # Analyze clusters
    clusters = analyze_correlation_clusters(adjacency, edge_corr)
    print(f"Number of correlation clusters: {len(clusters)}")

    print(f"\n{'='*60}")
    print(f"Correlation Clusters (connected components)")
    print(f"{'='*60}")
    for i, cluster in enumerate(clusters[:10]):  # Show top 10 clusters
        print(f"Cluster {i+1} ({len(cluster)} variates): {sorted(cluster)[:5]}{'...' if len(cluster) > 5 else ''}")
    if len(clusters) > 10:
        print(f"... and {len(clusters) - 10} more clusters")

    # Run greedy algorithm
    variates_to_remove = greedy_remove_high_corr(adjacency, edge_corr, variates_info)

    print(f"\n{'='*60}")
    print(f"Greedy Algorithm Result")
    print(f"{'='*60}")
    print(f"Variates to remove: {len(variates_to_remove)}")
    print(f"Variates to keep: {len(correlation_matrix) - len(variates_to_remove)}")
    print(f"Reduction: {len(variates_to_remove) / len(correlation_matrix) * 100:.1f}%")

    # Categorize removed variates
    removed_already_drop = []
    removed_kept = []
    for var in variates_to_remove:
        priority = get_variate_priority(var, variates_info)
        if priority > 0:
            removed_already_drop.append(var)
        else:
            removed_kept.append(var)

    print(f"\nBreakdown of variates to remove:")
    print(f"  - Already marked for deletion (kept=0): {len(removed_already_drop)}")
    print(f"  - Currently kept (kept=1): {len(removed_kept)}")

    # Show removed variates
    print(f"\n{'='*60}")
    print(f"Variates to Remove ({len(variates_to_remove)} total)")
    print(f"{'='*60}")

    # Sort by whether already marked for deletion
    for var in sorted(removed_already_drop):
        print(f"  - {var} (already marked [drop])")
    for var in sorted(removed_kept):
        print(f"  - {var}")

    # Verify: check remaining pairs
    remaining_adj = {k: set(v) for k, v in adjacency.items()}
    for var in variates_to_remove:
        neighbors = remaining_adj.pop(var, set())
        for neighbor in neighbors:
            if neighbor in remaining_adj:
                remaining_adj[neighbor].discard(var)

    remaining_pairs = sum(len(v) for v in remaining_adj.values()) // 2
    print(f"\n✅ Verification: {remaining_pairs} high-correlation pairs remaining after removal")

    if remaining_pairs > 0:
        print("⚠️  Warning: Some pairs still remain (should be 0)")

    # Generate removal command
    if not args.dry_run:
        # Auto-detect target_dir from summary_path
        import os
        if args.target_dir:
            target_dir = args.target_dir
        else:
            # Convert from processed_summary to processed_csv
            summary_dir = os.path.dirname(args.summary_path)
            target_dir = summary_dir.replace("processed_summary", "processed_csv")

        print(f"\n{'='*60}")
        print(f"Removal Commands")
        print(f"{'='*60}")

        # Generate command
        variates_str = ",".join(sorted(variates_to_remove))

        print(f"\nTo remove all {len(variates_to_remove)} variates at once:")
        print(f"python -m timebench.preprocess --remove_variate {variates_str} --target_dir {target_dir}")

        print(f"\nTo preview without making changes (dry run):")
        print(f"python -m timebench.preprocess --remove_variate {variates_str} --target_dir {target_dir} --dry_run")

        # If some are already marked for drop, suggest using remove_drop_marked first
        if removed_already_drop:
            print(f"\n💡 Tip: {len(removed_already_drop)} variates are already marked [drop].")
            print(f"   You can first remove those with:")
            print(f"   python -m timebench.preprocess --remove_drop_marked --target_dir {target_dir}")

        # Save removal list to file
        removal_list_path = os.path.join(os.path.dirname(args.summary_path), "_remove_for_correlation.json")
        removal_info = {
            "threshold": args.threshold,
            "total_high_corr_pairs": total_high_corr_pairs,
            "variates_to_remove": sorted(variates_to_remove),
            "removed_already_drop": sorted(removed_already_drop),
            "removed_kept": sorted(removed_kept),
            "target_dir": target_dir
        }
        with open(removal_list_path, "w", encoding="utf-8") as f:
            json.dump(removal_info, f, indent=4, ensure_ascii=False)
        print(f"\n📄 Removal list saved to: {removal_list_path}")


if __name__ == "__main__":
    main()

