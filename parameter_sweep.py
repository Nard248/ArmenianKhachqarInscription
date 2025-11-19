"""
Tests multiple parameter combinations on Image1 and saves results
to separate folders for visual inspection.
"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import morphology, measure
from skimage.transform import hough_line, hough_line_peaks
from pathlib import Path
from datetime import datetime
from itertools import product
import json

# ============================================================================
# CONFIGURATION: Define parameter ranges to test
# ============================================================================

PARAM_GRID = {
    'edge_threshold': [25, 30, 35, 40, 45],
    'morph_iterations': [1, 2, 3],
    'denoise_kernel': [3, 5],
    'filter_large': [30, 40, 50, 60, 70],
    'filter_small': [2, 3, 4],
    'num_peaks': [5, 10, 15],
    'angle_min': [85],
    'angle_max': [95],
    'search_window': [40, 50, 60]
}

TEST_ALL_COMBINATIONS = True

# Image to process
IMAGE_NAME = 'Image1'
IMAGE_DIR = Path('./Images')
OUTPUT_DIR = Path('./Results/parameter_sweep')

# ============================================================================
# Pipeline Functions
# ============================================================================

def detect_vertical_edges(img, threshold):
    """Detect vertical edges using Sobel operator"""
    sobel_east = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_west = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    east_response = cv2.filter2D(img, -1, sobel_east)
    west_response = cv2.filter2D(img, -1, sobel_west)

    east_edges = (np.abs(east_response) > threshold).astype(np.uint8) * 255
    west_edges = (np.abs(west_response) > threshold).astype(np.uint8) * 255
    combined_edges = cv2.bitwise_or(east_edges, west_edges)

    return combined_edges

def morphological_enhancement(img, iterations):
    """Apply horizontal dilation and vertical erosion"""
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    dilated = cv2.dilate(img, horizontal_kernel, iterations=iterations)
    morphed = cv2.erode(dilated, vertical_kernel, iterations=iterations)

    return morphed

def denoise_binary(img, kernel_size):
    """Denoise with median filter and re-binarize"""
    filtered = cv2.medianBlur(img, kernel_size)
    denoised = (filtered > 127).astype(np.uint8) * 255
    return denoised

def bandpass_filter(img, filter_large, filter_small):
    """Apply bandpass filter in frequency domain"""
    img_float = img.astype(np.float32)
    fft = fft2(img_float)
    fft_shift = fftshift(fft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    low_cutoff = min(rows, cols) / (2 * filter_large)
    high_cutoff = min(rows, cols) / (2 * filter_small)

    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - crow)**2 + (j - ccol)**2)
            if low_cutoff < dist < high_cutoff:
                mask[i, j] = 1.0

    fft_filtered = fft_shift * mask
    fft_ishift = ifftshift(fft_filtered)
    img_filtered = ifft2(fft_ishift)
    img_filtered = np.real(img_filtered)

    img_filtered = img_filtered - img_filtered.min()
    if img_filtered.max() > 0:
        img_filtered = img_filtered / img_filtered.max() * 255
    bandpass = np.clip(img_filtered, 0, 255).astype(np.uint8)

    return bandpass

def skeletonize_image(img):
    """Skeletonize binary image"""
    binary = img > 127
    skeleton = morphology.skeletonize(binary)
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton

def hough_transform_lines(img, num_peaks, angle_min, angle_max):
    """Apply Hough transform to detect lines"""
    thetas = np.deg2rad(np.arange(0, 180, 1))
    hough_space, angles, distances = hough_line(img, theta=thetas)

    h, theta, d = hough_line_peaks(hough_space, angles, distances, num_peaks=num_peaks)

    detected_lines = []
    for votes, angle, rho in zip(h, theta, d):
        angle_deg = np.rad2deg(angle)
        if angle_min <= angle_deg <= angle_max:
            detected_lines.append({
                'rho': rho,
                'theta': angle,
                'votes': int(votes),
                'angle_deg': float(angle_deg)
            })

    return detected_lines

def draw_hough_lines(img, lines):
    """Draw detected lines on image"""
    if len(img.shape) == 2:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        output = img.copy()

    for line in lines:
        rho, theta = line['rho'], line['theta']
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 2000 * (-b)), int(y0 + 2000 * a)
        x2, y2 = int(x0 - 2000 * (-b)), int(y0 - 2000 * a)
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output

def compute_horizontal_projection(img):
    """Compute horizontal projection profile"""
    projection = np.sum(img > 0, axis=1)
    return projection

def find_projection_valleys(projection, peak_positions, search_window):
    """Find valley positions around each peak"""
    boundaries = []

    for peak_y in peak_positions:
        if peak_y < 0 or peak_y >= len(projection):
            continue

        search_start = max(0, peak_y - search_window)
        search_end = peak_y
        top_region = projection[search_start:search_end]
        top_boundary = search_start + np.argmin(top_region) if len(top_region) > 0 else peak_y

        search_start = peak_y
        search_end = min(len(projection), peak_y + search_window)
        bottom_region = projection[search_start:search_end]
        bottom_boundary = search_start + np.argmin(bottom_region) if len(bottom_region) > 0 else peak_y

        boundaries.append((int(top_boundary), int(bottom_boundary)))

    return boundaries

def draw_text_line_boundaries(img, boundaries):
    """Draw top and bottom boundaries for each text line"""
    if len(img.shape) == 2:
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        output = img.copy()

    w = output.shape[1]
    for top, bottom in boundaries:
        cv2.line(output, (0, top), (w, top), (0, 255, 0), 2)
        cv2.line(output, (0, bottom), (w, bottom), (255, 0, 0), 2)

    return output

# ============================================================================
# Main Pipeline
# ============================================================================

def process_with_params(original_gray, params):
    """
    Process image with given parameters and return all intermediate results
    """
    results = {}

    # Step 1: Edge Detection
    edges = detect_vertical_edges(original_gray, params['edge_threshold'])
    results['edges'] = edges
    results['edge_pixels'] = int(np.sum(edges > 0))

    # Step 2: Morphology
    morphed = morphological_enhancement(edges, params['morph_iterations'])
    results['morphed'] = morphed

    # Step 3: Denoising
    denoised = denoise_binary(morphed, params['denoise_kernel'])
    results['denoised'] = denoised

    # Step 4: Bandpass
    bandpass = bandpass_filter(denoised, params['filter_large'], params['filter_small'])
    results['bandpass'] = bandpass

    # Step 5: Skeleton
    skeleton = skeletonize_image(bandpass)
    results['skeleton'] = skeleton
    results['skeleton_pixels'] = int(np.sum(skeleton > 0))

    # Step 6: Hough Transform
    detected_lines = hough_transform_lines(
        skeleton,
        params['num_peaks'],
        params['angle_min'],
        params['angle_max']
    )
    results['detected_lines'] = detected_lines
    results['num_lines'] = len(detected_lines)

    # Step 7: Line Overlay
    line_overlay = draw_hough_lines(original_gray, detected_lines)
    results['line_overlay'] = line_overlay

    # Step 8: Boundaries
    projection = compute_horizontal_projection(denoised)
    line_positions = [int(line['rho']) for line in detected_lines]
    boundaries = find_projection_valleys(projection, line_positions, params['search_window'])
    boundary_img = draw_text_line_boundaries(original_gray, boundaries)
    results['boundary_img'] = boundary_img
    results['num_boundaries'] = len(boundaries)

    return results

def save_results(results, params, folder_path, original_gray):
    """Save all results to folder"""
    folder_path.mkdir(parents=True, exist_ok=True)

    # Save images
    cv2.imwrite(str(folder_path / '00_original.png'), original_gray)
    cv2.imwrite(str(folder_path / '01_edges.png'), results['edges'])
    cv2.imwrite(str(folder_path / '02_morphed.png'), results['morphed'])
    cv2.imwrite(str(folder_path / '03_denoised.png'), results['denoised'])
    cv2.imwrite(str(folder_path / '04_bandpass.png'), results['bandpass'])
    cv2.imwrite(str(folder_path / '05_skeleton.png'), results['skeleton'])
    cv2.imwrite(str(folder_path / '06_lines.png'), cv2.cvtColor(results['line_overlay'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(folder_path / '07_boundaries.png'), cv2.cvtColor(results['boundary_img'], cv2.COLOR_RGB2BGR))

    # Save comparison
    comparison = np.hstack([
        cv2.cvtColor(original_gray, cv2.COLOR_GRAY2RGB),
        results['line_overlay'],
        results['boundary_img']
    ])
    cv2.imwrite(str(folder_path / '08_comparison.png'), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    # Save config
    config_data = {
        'parameters': params,
        'statistics': {
            'edge_pixels': results['edge_pixels'],
            'skeleton_pixels': results['skeleton_pixels'],
            'num_lines_detected': results['num_lines'],
            'num_boundaries': results['num_boundaries']
        },
        'detected_lines': results['detected_lines']
    }

    with open(folder_path / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=2)

    # Save readable config
    with open(folder_path / 'config.txt', 'w') as f:
        f.write("PARAMETERS\n")
        f.write("="*50 + "\n")
        for key, value in params.items():
            f.write(f"{key:20s} = {value}\n")
        f.write("\n")
        f.write("STATISTICS\n")
        f.write("="*50 + "\n")
        f.write(f"{'Edge pixels':20s} = {results['edge_pixels']:,}\n")
        f.write(f"{'Skeleton pixels':20s} = {results['skeleton_pixels']:,}\n")
        f.write(f"{'Lines detected':20s} = {results['num_lines']}\n")
        f.write(f"{'Boundaries found':20s} = {results['num_boundaries']}\n")
        f.write("\n")
        f.write("DETECTED LINES\n")
        f.write("="*50 + "\n")
        for i, line in enumerate(results['detected_lines'], 1):
            f.write(f"Line {i}: rho={line['rho']:.1f}, angle={line['angle_deg']:.1f}°, votes={line['votes']}\n")

# ============================================================================
# Generate parameter combinations
# ============================================================================

def generate_param_combinations(grid, test_all=False):
    """
    Generate parameter combinations to test

    If test_all is False, uses a smart sampling strategy to reduce
    the total number of combinations while still covering the space well.
    """
    if test_all:
        # Generate ALL combinations (warning: can be very large!)
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        combinations = []
        for combo in product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        return combinations

    else:
        # Smart sampling strategy
        combinations = []

        # 1. Test each parameter individually at different values (keep others at median)
        median_params = {k: grid[k][len(grid[k])//2] for k in grid.keys()}

        for param_name in grid.keys():
            for value in grid[param_name]:
                params = median_params.copy()
                params[param_name] = value
                if params not in combinations:
                    combinations.append(params.copy())

        # 2. Add some interesting combinations
        # Low threshold + high filter (for noisy images)
        combinations.append({
            'edge_threshold': min(grid['edge_threshold']),
            'morph_iterations': grid['morph_iterations'][len(grid['morph_iterations'])//2],
            'denoise_kernel': max(grid['denoise_kernel']),
            'filter_large': max(grid['filter_large']),
            'filter_small': min(grid['filter_small']),
            'num_peaks': grid['num_peaks'][len(grid['num_peaks'])//2],
            'angle_min': grid['angle_min'][0],
            'angle_max': grid['angle_max'][0],
            'search_window': grid['search_window'][len(grid['search_window'])//2]
        })

        # High threshold + low filter (for clean images)
        combinations.append({
            'edge_threshold': max(grid['edge_threshold']),
            'morph_iterations': grid['morph_iterations'][len(grid['morph_iterations'])//2],
            'denoise_kernel': min(grid['denoise_kernel']),
            'filter_large': min(grid['filter_large']),
            'filter_small': max(grid['filter_small']),
            'num_peaks': grid['num_peaks'][len(grid['num_peaks'])//2],
            'angle_min': grid['angle_min'][0],
            'angle_max': grid['angle_max'][0],
            'search_window': grid['search_window'][len(grid['search_window'])//2]
        })

        # Aggressive processing
        combinations.append({
            'edge_threshold': min(grid['edge_threshold']),
            'morph_iterations': max(grid['morph_iterations']),
            'denoise_kernel': max(grid['denoise_kernel']),
            'filter_large': max(grid['filter_large']),
            'filter_small': min(grid['filter_small']),
            'num_peaks': max(grid['num_peaks']),
            'angle_min': grid['angle_min'][0],
            'angle_max': grid['angle_max'][0],
            'search_window': max(grid['search_window'])
        })

        # Conservative processing
        combinations.append({
            'edge_threshold': max(grid['edge_threshold']),
            'morph_iterations': min(grid['morph_iterations']),
            'denoise_kernel': min(grid['denoise_kernel']),
            'filter_large': min(grid['filter_large']),
            'filter_small': max(grid['filter_small']),
            'num_peaks': min(grid['num_peaks']),
            'angle_min': grid['angle_min'][0],
            'angle_max': grid['angle_max'][0],
            'search_window': min(grid['search_window'])
        })

        return combinations

def create_folder_name(params):
    """Create a descriptive folder name from parameters"""
    return (f"et{params['edge_threshold']}_"
            f"mi{params['morph_iterations']}_"
            f"dk{params['denoise_kernel']}_"
            f"fl{params['filter_large']}_"
            f"fs{params['filter_small']}_"
            f"np{params['num_peaks']}_"
            f"sw{params['search_window']}")

# ============================================================================
# Main execution
# ============================================================================

def main():
    print("="*70)
    print("Armenian Inscription - Parameter Sweep")
    print("="*70)
    print()

    # Load image
    image_path = IMAGE_DIR / f"{IMAGE_NAME}.png"
    if not image_path.exists():
        # Try other extensions
        for ext in ['.jpg', '.jpeg']:
            alt_path = IMAGE_DIR / f"{IMAGE_NAME}{ext}"
            if alt_path.exists():
                image_path = alt_path
                break

    if not image_path.exists():
        print(f"ERROR: Could not find image: {IMAGE_NAME}")
        return

    print(f"Loading image: {image_path}")
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    print(f"Image size: {original_gray.shape}")
    print()

    # Generate parameter combinations
    print("Generating parameter combinations...")
    combinations = generate_param_combinations(PARAM_GRID, TEST_ALL_COMBINATIONS)
    print(f"Total combinations to test: {len(combinations)}")
    print()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = OUTPUT_DIR / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {sweep_dir}")
    print()
    print("="*70)
    print()

    # Process each combination
    for i, params in enumerate(combinations, 1):
        print(f"[{i}/{len(combinations)}] Processing: {create_folder_name(params)}")

        try:
            # Process
            results = process_with_params(original_gray, params)

            # Save
            folder_name = create_folder_name(params)
            folder_path = sweep_dir / folder_name
            save_results(results, params, folder_path, original_gray)

            # Print summary
            print(f"    Lines detected: {results['num_lines']}, Boundaries: {results['num_boundaries']}")

        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    print()
    print("="*70)
    print("PARAMETER SWEEP COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {sweep_dir}")
    print(f"Total configurations tested: {len(combinations)}")

    # Create index HTML for easy browsing
    create_html_index(sweep_dir, combinations)
    print(f"Open: {sweep_dir / 'index.html'} in a browser")
    print()

def create_html_index(sweep_dir, combinations):
    """Create an HTML index for easy browsing of results"""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Parameter Sweep Results - Image1</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 20px; }
        .result { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .result img { width: 100%; border-radius: 4px; }
        .result h3 { margin-top: 0; font-size: 14px; color: #666; }
        .result pre { background: #f9f9f9; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto; }
        .stats { color: #0066cc; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Parameter Sweep Results - Image1</h1>
    <p>Click on any image to see full size. Check config.txt for complete details.</p>
    <div class="grid">
"""

    for params in combinations:
        folder_name = create_folder_name(params)
        folder_path = Path(folder_name)

        # Read config to get statistics
        config_path = sweep_dir / folder_name / 'config.json'
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            stats = config['statistics']
        else:
            stats = {}

        html += f"""
        <div class="result">
            <h3>{folder_name}</h3>
            <a href="{folder_name}/08_comparison.png" target="_blank">
                <img src="{folder_name}/08_comparison.png" alt="{folder_name}">
            </a>
            <div class="stats">
                Lines: {stats.get('num_lines_detected', '?')} |
                Boundaries: {stats.get('num_boundaries', '?')}
            </div>
            <pre>edge_threshold: {params['edge_threshold']}
morph_iter: {params['morph_iterations']}
denoise: {params['denoise_kernel']}
filter_large: {params['filter_large']}
filter_small: {params['filter_small']}
num_peaks: {params['num_peaks']}
search_window: {params['search_window']}</pre>
            <a href="{folder_name}/config.txt" target="_blank">View full config</a>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(sweep_dir / 'index.html', 'w') as f:
        f.write(html)

if __name__ == '__main__':
    main()
