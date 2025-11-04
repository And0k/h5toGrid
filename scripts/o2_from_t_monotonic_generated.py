import numpy as np
from scipy.interpolate import LinearNDInterpolator


def extract_last_monotonic_section(z_profile):
    """
    Extract the last monotonic section from a z profile.
    Searches backwards from the end until monotonicity breaks.

    Parameters:
    -----------
    z_profile : array, shape (n_y,)
        Z values along y direction

    Returns:
    --------
    start_idx : int
        Starting index of last monotonic section
    """
    n = len(z_profile)

    # Handle NaN values
    valid_mask = ~np.isnan(z_profile)
    if not np.any(valid_mask):
        return n  # No valid data

    # Start from the end
    valid_indices = np.where(valid_mask)[0]
    last_valid_idx = valid_indices[-1]

    # Get the last valid section
    z_valid = z_profile[valid_mask]

    # Check if increasing or decreasing from the end
    if len(z_valid) < 2:
        return last_valid_idx

    # Determine direction from last two points
    is_increasing = z_valid[-1] > z_valid[-2]

    # Walk backwards until monotonicity breaks
    start_valid_idx = len(z_valid) - 1
    for i in range(len(z_valid) - 2, -1, -1):
        if is_increasing:
            if z_valid[i] > z_valid[i + 1]:  # Breaks monotonicity
                break
        else:
            if z_valid[i] < z_valid[i + 1]:  # Breaks monotonicity
                break
        start_valid_idx = i

    # Map back to original indices
    return valid_indices[start_valid_idx]


def prepare_monotonic_data(x_grid, y_grid, z_grid):
    """
    Prepare data by keeping only last monotonic section in y for each x.

    Parameters:
    -----------
    x_grid : array, shape (n_x,)
        X coordinates
    y_grid : array, shape (n_y,)
        Y coordinates
    z_grid : array, shape (n_x, n_y)
        Z values at each (x, y) point

    Returns:
    --------
    x_points : array
        X coordinates of filtered points
    z_points : array
        Z coordinates of filtered points
    y_points : array
        Y coordinates of filtered points (these will be interpolated)
    """
    x_points = []
    y_points = []
    z_points = []

    for i, x_val in enumerate(x_grid):
        # Get z profile at this x
        z_profile = z_grid[i, :]

        # Find start of last monotonic section
        start_idx = extract_last_monotonic_section(z_profile)

        # Keep only the monotonic section
        y_section = y_grid[start_idx:]
        z_section = z_profile[start_idx:]

        # Filter out NaN values
        valid_mask = ~np.isnan(z_section)
        if not np.any(valid_mask):
            continue

        y_section = y_section[valid_mask]
        z_section = z_section[valid_mask]

        # Add to points list
        n_points = len(y_section)
        x_points.extend([x_val] * n_points)
        y_points.extend(y_section)
        z_points.extend(z_section)

    return np.array(x_points), np.array(z_points), np.array(y_points)


def find_y_for_z_isosurface(x_grid, y_grid, z_grid, z_target):
    """
    Find y position where z equals z_target for each x, using only last monotonic section.

    Parameters:
    -----------
    x_grid : array, shape (n_x,)
        X coordinates (regular grid)
    y_grid : array, shape (n_y,)
        Y coordinates (regular grid)
    z_grid : array, shape (n_x, n_y)
        Z values at each (x, y) point
    z_target : float or array
        Target z value(s) to find

    Returns:
    --------
    x_out : array
        X coordinates (same as x_grid)
    y_out : array
        Y positions where z crosses z_target for each x
    """
    # Prepare monotonic data
    x_points, z_points, y_points = prepare_monotonic_data(x_grid, y_grid, z_grid)

    if len(x_points) == 0:
        return x_grid, np.full(len(x_grid), np.nan)

    # Create interpolator: given (x, z) -> return y
    interpolator = LinearNDInterpolator(np.column_stack([x_points, z_points]), y_points)

    # Query for each x at target z
    z_target_scalar = np.atleast_1d(z_target)[0] if np.ndim(z_target) == 0 else z_target

    if np.isscalar(z_target) or len(np.atleast_1d(z_target)) == 1:
        # Single z target
        query_points = np.column_stack([x_grid, np.full(len(x_grid), z_target_scalar)])
        y_out = interpolator(query_points)
    else:
        # Multiple z targets - assume one per x
        z_target_array = np.asarray(z_target)
        if len(z_target_array) != len(x_grid):
            raise ValueError("z_target array must have same length as x_grid")
        query_points = np.column_stack([x_grid, z_target_array])
        y_out = interpolator(query_points)

    return x_grid, y_out


def find_y_direct_search(x_grid, y_grid, z_grid, z_target):
    """
    Alternative: Direct search without interpolator (more robust for edge cases).
    Find y where z crosses z_target using last monotonic section.

    Parameters:
    -----------
    x_grid : array, shape (n_x,)
        X coordinates
    y_grid : array, shape (n_y,)
        Y coordinates
    z_grid : array, shape (n_x, n_y)
        Z values
    z_target : float
        Target z value

    Returns:
    --------
    x_out : array
        X coordinates
    y_out : array
        Y positions where z crosses z_target
    """
    y_out = np.full(len(x_grid), np.nan)

    for i, x_val in enumerate(x_grid):
        z_profile = z_grid[i, :]

        # Get last monotonic section
        start_idx = extract_last_monotonic_section(z_profile)
        z_section = z_profile[start_idx:]
        y_section = y_grid[start_idx:]

        # Remove NaN
        valid_mask = ~np.isnan(z_section)
        if not np.any(valid_mask):
            continue

        z_section = z_section[valid_mask]
        y_section = y_section[valid_mask]

        if len(z_section) < 2:
            continue

        # Check if z_target is in range
        z_min, z_max = np.min(z_section), np.max(z_section)
        if not (z_min <= z_target <= z_max or z_max <= z_target <= z_min):
            continue

        # Find crossing
        diff = z_section - z_target
        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

        if len(sign_changes) > 0:
            # Take last crossing in the monotonic section
            idx = sign_changes[-1]

            # Linear interpolation
            y1, y2 = y_section[idx], y_section[idx + 1]
            z1, z2 = z_section[idx], z_section[idx + 1]

            t = (z_target - z1) / (z2 - z1)
            y_out[i] = y1 + t * (y2 - y1)

    return x_grid, y_out


# Example usage and visualization
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create test data with non-monotonic z profile
    x_grid = np.linspace(0, 10, 50)
    y_grid = np.linspace(0, 20, 100)
    X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")

    # Create z field with non-monotonic behavior: rises then falls then rises again
    Z = 5 + 2 * np.sin(X) - 0.1 * Y + 0.01 * Y**2 - 0.0005 * Y**3

    # Target z value
    z_target = 5.0

    # Method 1: Using LinearNDInterpolator with monotonic preprocessing
    x_result1, y_result1 = find_y_for_z_isosurface(x_grid, y_grid, Z, z_target)

    # Method 2: Direct search (more robust)
    x_result2, y_result2 = find_y_direct_search(x_grid, y_grid, Z, z_target)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Full z field
    ax = axes[0, 0]
    contour = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    ax.contour(X, Y, Z, levels=[z_target], colors="red", linewidths=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Z field with multiple crossings at z={z_target}")
    plt.colorbar(contour, ax=ax, label="Z value")

    # Plot 2: Results comparison
    ax = axes[0, 1]
    ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.3)
    ax.plot(x_result1, y_result1, "r-", linewidth=3, label="LinearNDInterpolator", alpha=0.7)
    ax.plot(x_result2, y_result2, "b--", linewidth=2, label="Direct search")
    ax.contour(X, Y, Z, levels=[z_target], colors="black", linewidths=1, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Last isoline at z={z_target}")
    ax.legend()

    # Plot 3: Show monotonic sections at a few x positions
    ax = axes[1, 0]
    x_samples = [5, 10, 20, 30, 40]
    for idx in x_samples:
        z_profile = Z[idx, :]
        start_idx = extract_last_monotonic_section(z_profile)

        # Plot full profile
        ax.plot(y_grid, z_profile, "gray", alpha=0.3, linewidth=1)
        # Highlight monotonic section
        ax.plot(y_grid[start_idx:], z_profile[start_idx:], linewidth=2, label=f"x={x_grid[idx]:.1f}")

    ax.axhline(z_target, color="red", linestyle="--", label=f"z_target={z_target}")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("Z profiles at sample x positions\n(highlighted = last monotonic section)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Scatter plot of monotonic data points
    ax = axes[1, 1]
    x_points, z_points, y_points = prepare_monotonic_data(x_grid, y_grid, Z)
    scatter = ax.scatter(x_points, y_points, c=z_points, s=1, alpha=0.5, cmap="viridis")
    ax.plot(x_result2, y_result2, "r-", linewidth=2, label=f"Isoline z={z_target}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Monotonic data points used for interpolation")
    plt.colorbar(scatter, ax=ax, label="Z value")
    ax.legend()

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"LinearNDInterpolator: {np.sum(~np.isnan(y_result1))} valid points")
    print(f"Direct search: {np.sum(~np.isnan(y_result2))} valid points")
    print(f"Difference (where both valid): {np.nanmean(np.abs(y_result1 - y_result2)):.4f}")
