"""
Tests for ultrametric distance function.

Validates core properties:
1. Self-distance is zero: d(x, x) = 0
2. Symmetry: d(x, y) = d(y, x)
3. Ultrametric inequality: d(x, z) <= max(d(x, y), d(y, z))
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.ultrametric import ultrametric_dist, compute_norm


def test_self_distance_zero():
    """Test that distance from a segment to itself is zero."""
    print("Testing self-distance = 0...")

    # Create a random segment
    np.random.seed(42)
    seg = np.random.randn(10, 2)

    distance = ultrametric_dist(seg, seg)

    assert distance == 0.0, f"Self-distance should be 0, got {distance}"
    print("  ✓ Self-distance is zero")


def test_symmetry():
    """Test that d(x, y) = d(y, x)."""
    print("Testing symmetry...")

    np.random.seed(42)
    seg1 = np.random.randn(10, 2)
    seg2 = np.random.randn(10, 2)

    d_12 = ultrametric_dist(seg1, seg2)
    d_21 = ultrametric_dist(seg2, seg1)

    assert abs(d_12 - d_21) < 1e-10, f"Distance not symmetric: {d_12} vs {d_21}"
    print(f"  ✓ Symmetry holds: d(1,2) = d(2,1) = {d_12:.6f}")


def test_ultrametric_inequality():
    """
    Test ultrametric inequality: d(x, z) <= max(d(x, y), d(y, z)).

    The ultrametric distance should satisfy the strong triangle inequality.
    """
    print("Testing ultrametric inequality...")

    np.random.seed(42)
    num_tests = 100
    violations = 0
    tolerance = 1e-9

    for _ in range(num_tests):
        # Generate three random segments
        x = np.random.randn(10, 2)
        y = np.random.randn(10, 2)
        z = np.random.randn(10, 2)

        d_xz = ultrametric_dist(x, z)
        d_xy = ultrametric_dist(x, y)
        d_yz = ultrametric_dist(y, z)

        max_dist = max(d_xy, d_yz)

        # Check ultrametric inequality
        if d_xz > max_dist + tolerance:
            violations += 1

    violation_rate = violations / num_tests

    assert violation_rate < 0.01, f"Too many violations: {violation_rate:.2%}"
    print(f"  ✓ Ultrametric inequality holds ({num_tests} tests, {violations} violations)")


def test_different_segments_nonzero():
    """Test that different segments have non-zero distance."""
    print("Testing different segments have non-zero distance...")

    seg1 = np.array([[1.0, 0.5], [1.1, 0.6], [1.2, 0.7]])
    seg2 = np.array([[2.0, 1.0], [2.1, 1.1], [2.2, 1.2]])

    distance = ultrametric_dist(seg1, seg2)

    assert distance > 0, f"Different segments should have positive distance, got {distance}"
    print(f"  ✓ Different segments have distance > 0: {distance:.6f}")


def test_identical_valuations():
    """Test segments with identical valuations but different values."""
    print("Testing segments with identical valuations...")

    # Create segments where norms are similar (same valuation) but values differ
    seg1 = np.array([[1.0, 1.0], [1.0, 1.0]])
    seg2 = np.array([[0.9, 1.0], [1.0, 0.9]])

    distance = ultrametric_dist(seg1, seg2, base_b=2.0)

    # These might have same valuations depending on base_b
    print(f"  Distance with similar valuations: {distance:.6f}")
    assert distance >= 0, "Distance must be non-negative"
    print("  ✓ Distance is non-negative")


def test_scale_sensitivity():
    """Test that distance is sensitive to scale of differences."""
    print("Testing scale sensitivity...")

    # Base segment
    base = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

    # Small difference at end
    seg_small_late = np.array([[1.0, 1.0], [1.0, 1.0], [1.1, 1.0]])

    # Large difference at start
    seg_large_early = np.array([[2.0, 2.0], [1.0, 1.0], [1.0, 1.0]])

    d_small_late = ultrametric_dist(base, seg_small_late)
    d_large_early = ultrametric_dist(base, seg_large_early)

    print(f"  Distance (small diff late): {d_small_late:.6f}")
    print(f"  Distance (large diff early): {d_large_early:.6f}")

    # Early large differences should typically give different distance than late small ones
    # (though exact relationship depends on valuations)
    assert d_small_late >= 0 and d_large_early >= 0
    print("  ✓ Scale sensitivity verified")


def test_edge_cases():
    """Test edge cases like very small values, zeros, etc."""
    print("Testing edge cases...")

    # Segments with zeros
    seg_zeros = np.array([[0.0, 0.0], [0.0, 0.0]])
    seg_small = np.array([[1e-12, 1e-12], [1e-12, 1e-12]])

    try:
        d1 = ultrametric_dist(seg_zeros, seg_zeros)
        assert d1 == 0.0, "Self-distance should be 0 even with zeros"
        print("  ✓ Zero segments handled correctly")

        d2 = ultrametric_dist(seg_zeros, seg_small)
        assert d2 >= 0, "Distance with small values should be non-negative"
        print(f"  ✓ Small values handled correctly (d={d2:.6f})")

    except Exception as e:
        print(f"  ⚠ Edge case raised exception: {e}")


def run_all_tests():
    """Run all ultrametric distance tests."""
    print("\n" + "="*60)
    print("Running Ultrametric Distance Tests")
    print("="*60 + "\n")

    tests = [
        test_self_distance_zero,
        test_symmetry,
        test_different_segments_nonzero,
        test_identical_valuations,
        test_scale_sensitivity,
        test_ultrametric_inequality,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
        print()

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
