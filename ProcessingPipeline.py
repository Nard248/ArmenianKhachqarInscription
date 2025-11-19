"""
Armenian Inscription Analysis - Submission Generator
CS 260/371 Image Processing Project

Generates all required deliverables for the \Lines subfolder submission.
Processes all images through the 13-step pipeline as specified in the assignment.

"""

import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage import morphology, measure
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

PARAMS = {
    'edge_threshold': 25,
    'morph_iterations': 1,
    'denoise_kernel': 5,
    'filter_large': 50,
    'filter_small': 2,
    'min_particle_size': 100,
    'num_peaks': 15,
    'angle_min': 85,
    'angle_max': 95,
    'search_window': 60
}

# Paths
IMAGE_DIR = Path('./Images')
SUBMISSION_DIR = Path('./Results/Submission_Lines')

# Image files to process
IMAGES_TO_PROCESS = ['Image1', 'Image2', 'Image3', 'Image4']

# ============================================================================
# STEP FUNCTIONS
# ============================================================================

def step_2_detect_edges(img, threshold):
    """
    Step 2: Detect vertical edges (east and west) using Edge Detection Operators
    """
    # Define Sobel kernels for vertical edge detection
    sobel_east = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_west = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)

    # Apply filters
    east_response = cv2.filter2D(img, -1, sobel_east)
    west_response = cv2.filter2D(img, -1, sobel_west)

    # Binarize
    east_edges = (np.abs(east_response) > threshold).astype(np.uint8) * 255
    west_edges = (np.abs(west_response) > threshold).astype(np.uint8) * 255

    # Combine
    combined_edges = cv2.bitwise_or(east_edges, west_edges)

    kernels = {'east': sobel_east, 'west': sobel_west}

    return east_edges, west_edges, combined_edges, kernels

def step_3_morphology(img, iterations):
    """
    Step 3: Strengthen vertical edges with horizontal dilation and vertical erosion
    """
    # Define structuring elements
    horizontal_se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    vertical_se = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))

    # Apply operations
    dilated = cv2.dilate(img, horizontal_se, iterations=iterations)
    eroded = cv2.erode(dilated, vertical_se, iterations=iterations)

    se_dict = {'horizontal': horizontal_se, 'vertical': vertical_se}

    return dilated, eroded, se_dict

def step_4_denoise(img, kernel_size):
    """
    Step 4: Denoise using median filter and maintain binary format
    """
    # Apply median filter
    filtered = cv2.medianBlur(img, kernel_size)

    # Re-binarize to maintain binary format
    denoised = (filtered > 127).astype(np.uint8) * 255

    filter_info = {
        'type': 'Median Filter (nonlinear)',
        'kernel_size': kernel_size,
        'radius': kernel_size // 2
    }

    return denoised, filter_info

def step_5_bandpass(img, filter_large, filter_small):
    """
    Step 5: Apply Bandpass Filter to detect high-frequency text regions
    """
    img_float = img.astype(np.float32)

    # FFT
    fft = fft2(img_float)
    fft_shift = fftshift(fft)

    # Create bandpass mask
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

    # Apply filter
    fft_filtered = fft_shift * mask
    fft_ishift = ifftshift(fft_filtered)
    img_filtered = ifft2(fft_ishift)
    img_filtered = np.real(img_filtered)

    # Normalize
    img_filtered = img_filtered - img_filtered.min()
    if img_filtered.max() > 0:
        img_filtered = img_filtered / img_filtered.max() * 255
    filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)

    filter_params = {
        'filter_large': filter_large,
        'filter_small': filter_small,
        'low_cutoff_freq': low_cutoff,
        'high_cutoff_freq': high_cutoff
    }

    return filtered, filter_params

def step_6_binary_mask(denoised_edges, bandpass_result):
    """
    Step 6: Apply bandpass result as binary mask using AND operation
    """
    bandpass_binary = (bandpass_result > 127).astype(np.uint8) * 255
    masked = cv2.bitwise_and(denoised_edges, bandpass_binary)
    return masked

def step_8_particle_analysis(img, min_size):
    """
    Step 8: Analyze particles and fit ellipses
    """
    # Label connected components
    labeled = measure.label(img > 127, connectivity=2)
    regions = measure.regionprops(labeled)

    # Create RGB image for drawing
    ellipse_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    particles = []
    for region in regions:
        if region.area >= min_size:
            # Draw ellipse
            center = (int(region.centroid[1]), int(region.centroid[0]))
            axes = (int(region.minor_axis_length / 2), int(region.major_axis_length / 2))
            angle = np.degrees(region.orientation)

            cv2.ellipse(ellipse_img, center, axes, angle, 0, 360, (255, 0, 0), 2)

            particles.append({
                'area': int(region.area),
                'centroid': (int(region.centroid[1]), int(region.centroid[0])),
                'major_axis': float(region.major_axis_length),
                'minor_axis': float(region.minor_axis_length),
                'orientation': float(np.degrees(region.orientation))
            })

    return ellipse_img, particles

def step_9_skeletonize(img):
    """
    Step 9: Skeletonize the filtered image
    """
    binary = img > 127
    skeleton = morphology.skeletonize(binary)
    skeleton = (skeleton * 255).astype(np.uint8)
    return skeleton

def step_10_hough_transform(img):
    """
    Step 10: Apply Hough Transform to detect horizontal lines
    """
    thetas = np.deg2rad(np.arange(0, 180, 1))
    hough_space, angles, distances = hough_line(img, theta=thetas)
    return hough_space, angles, distances

def step_11_threshold_hough(hough_space, angles, distances, angle_min, angle_max, percentile=90):
    """
    Step 11: Threshold Hough space around pi/2 (90 degrees) to locate horizontal lines
    """
    # Create binary version of Hough space
    threshold_value = np.percentile(hough_space, percentile)
    binary_hough = (hough_space > threshold_value).astype(np.uint8) * 255

    # Filter region around 90 degrees
    angle_deg = np.rad2deg(angles)
    angle_mask = (angle_deg >= angle_min) & (angle_deg <= angle_max)

    # Create filtered region image
    filtered_region = np.zeros_like(hough_space)
    filtered_region[:, angle_mask] = hough_space[:, angle_mask]

    return binary_hough, filtered_region

def step_12_locate_lines(original_img, hough_space, angles, distances, num_peaks, angle_min, angle_max):
    """
    Step 12: Locate horizontal lines detected by Hough Transform on original image
    """
    # Find peaks
    h, theta, d = hough_line_peaks(hough_space, angles, distances, num_peaks=num_peaks)

    # Filter for horizontal lines
    detected_lines = []
    for votes, angle, rho in zip(h, theta, d):
        angle_deg = np.rad2deg(angle)
        if angle_min <= angle_deg <= angle_max:
            detected_lines.append({
                'rho': float(rho),
                'theta': float(angle),
                'angle_deg': float(angle_deg),
                'votes': int(votes)
            })

    # Draw lines on original image
    if len(original_img.shape) == 2:
        line_overlay = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        line_overlay = original_img.copy()

    for line in detected_lines:
        rho, theta = line['rho'], line['theta']
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 2000 * (-b)), int(y0 + 2000 * a)
        x2, y2 = int(x0 - 2000 * (-b)), int(y0 - 2000 * a)
        cv2.line(line_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return line_overlay, detected_lines

def step_13_projection_boundaries(denoised_img, detected_lines, search_window):
    """
    Step 13: Compute horizontal projection and locate boundaries
    """
    # Compute projection
    projection = np.sum(denoised_img > 0, axis=1)

    # Extract line positions
    line_positions = [int(line['rho']) for line in detected_lines]

    # Find boundaries
    boundaries = []
    for peak_y in line_positions:
        if peak_y < 0 or peak_y >= len(projection):
            continue

        # Search upward
        search_start = max(0, peak_y - search_window)
        search_end = peak_y
        top_region = projection[search_start:search_end]
        top_boundary = search_start + np.argmin(top_region) if len(top_region) > 0 else peak_y

        # Search downward
        search_start = peak_y
        search_end = min(len(projection), peak_y + search_window)
        bottom_region = projection[search_start:search_end]
        bottom_boundary = search_start + np.argmin(bottom_region) if len(bottom_region) > 0 else peak_y

        boundaries.append({
            'top': int(top_boundary),
            'bottom': int(bottom_boundary),
            'height': int(bottom_boundary - top_boundary)
        })

    # Draw boundaries
    if len(denoised_img.shape) == 2:
        boundary_overlay = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2RGB)
    else:
        boundary_overlay = denoised_img.copy()

    w = boundary_overlay.shape[1]
    for boundary in boundaries:
        cv2.line(boundary_overlay, (0, boundary['top']), (w, boundary['top']), (0, 255, 0), 2)
        cv2.line(boundary_overlay, (0, boundary['bottom']), (w, boundary['bottom']), (255, 0, 0), 2)

    return boundary_overlay, boundaries, projection

# ============================================================================
# VISUALIZATION AND SAVING FUNCTIONS
# ============================================================================

def save_kernel_visualization(kernels, filepath):
    """Save visualization of edge detection kernels"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(kernels['east'], cmap='RdBu', vmin=-2, vmax=2)
    axes[0].set_title('East Edge Kernel (Sobel)', fontweight='bold')
    axes[0].axis('off')
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f'{kernels["east"][i, j]:.0f}',
                        ha='center', va='center', fontsize=12, fontweight='bold')

    axes[1].imshow(kernels['west'], cmap='RdBu', vmin=-2, vmax=2)
    axes[1].set_title('West Edge Kernel (Sobel)', fontweight='bold')
    axes[1].axis('off')
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernels["west"][i, j]:.0f}',
                        ha='center', va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def save_se_visualization(se_dict, filepath):
    """Save visualization of structuring elements"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(se_dict['horizontal'], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Horizontal SE (Dilation)', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(se_dict['vertical'], cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Vertical SE (Erosion)', fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def save_hough_visualization(hough_space, angles, distances, filepath):
    """Save Hough transform accumulator visualization"""
    plt.figure(figsize=(12, 8))
    plt.imshow(hough_space,
              extent=[np.rad2deg(angles[0]), np.rad2deg(angles[-1]),
                     distances[-1], distances[0]],
              cmap='hot', aspect='auto')
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Distance (pixels)', fontsize=12)
    plt.title('Hough Transform Accumulator', fontsize=14, fontweight='bold')
    plt.axvline(90, color='cyan', linestyle='--', linewidth=2, label='90° (horizontal)')
    plt.colorbar(label='Votes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

def save_projection_plot(projection, boundaries, filepath):
    """Save horizontal projection profile plot"""
    plt.figure(figsize=(10, 8))
    plt.plot(projection, range(len(projection)), 'b-', linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel('Pixel Count', fontsize=12)
    plt.ylabel('Row (y)', fontsize=12)
    plt.title('Horizontal Projection Profile', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Mark boundaries
    for i, boundary in enumerate(boundaries):
        label_top = 'Top Boundary' if i == 0 else None
        label_bottom = 'Bottom Boundary' if i == 0 else None
        plt.axhline(boundary['top'], color='green', linestyle='--', alpha=0.7, label=label_top)
        plt.axhline(boundary['bottom'], color='red', linestyle='--', alpha=0.7, label=label_bottom)

    if boundaries:
        plt.legend()

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single_image(image_name, img_gray, img_color, params, output_dir):
    """
    Process a single image through all 13 steps
    """
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"{'='*70}\n")

    # Step 1: Original grayscale (already done during loading)
    results['original_gray'] = img_gray
    results['original_color'] = img_color

    # Step 2: Edge Detection
    print("  Step 2: Detecting vertical edges...")
    east, west, combined, kernels = step_2_detect_edges(img_gray, params['edge_threshold'])
    results['step2'] = {
        'east_edges': east,
        'west_edges': west,
        'combined_edges': combined,
        'kernels': kernels
    }

    # Step 3: Morphology
    print("  Step 3: Morphological enhancement...")
    dilated, eroded, se_dict = step_3_morphology(combined, params['morph_iterations'])
    results['step3'] = {
        'dilated': dilated,
        'eroded': eroded,
        'structuring_elements': se_dict
    }

    # Step 4: Denoising
    print("  Step 4: Denoising...")
    denoised, filter_info = step_4_denoise(eroded, params['denoise_kernel'])
    results['step4'] = {
        'denoised': denoised,
        'filter_info': filter_info
    }

    # Step 5: Bandpass Filter
    print("  Step 5: Applying bandpass filter...")
    bandpass, bp_params = step_5_bandpass(denoised, params['filter_large'], params['filter_small'])
    results['step5'] = {
        'filtered': bandpass,
        'params': bp_params
    }

    # Step 6: Binary Masking (optional)
    print("  Step 6: Binary masking...")
    masked = step_6_binary_mask(denoised, bandpass)
    results['step6'] = {
        'masked': masked
    }

    # Step 7: Second Bandpass (optional, on masked image)
    print("  Step 7: Second bandpass filter...")
    bandpass2, bp2_params = step_5_bandpass(masked, params['filter_large'], params['filter_small'])
    results['step7'] = {
        'filtered_masked': bandpass2,
        'params': bp2_params
    }

    # Decide which image to use for subsequent steps (step 5 or step 7)
    # Using step 5 result as primary path
    working_image = bandpass

    # Step 8: Particle Analysis
    print("  Step 8: Particle analysis...")
    ellipse_img, particles = step_8_particle_analysis(working_image, params['min_particle_size'])
    results['step8'] = {
        'ellipse_image': ellipse_img,
        'particles': particles
    }

    # Step 9: Skeletonization
    print("  Step 9: Skeletonizing...")
    skeleton = step_9_skeletonize(working_image)
    results['step9'] = {
        'skeleton': skeleton
    }

    # Step 10: Hough Transform
    print("  Step 10: Hough transform...")
    hough_space, angles, distances = step_10_hough_transform(skeleton)
    results['step10'] = {
        'hough_space': hough_space,
        'angles': angles,
        'distances': distances
    }

    # Step 11: Threshold Hough Space
    print("  Step 11: Thresholding Hough space...")
    binary_hough, filtered_region = step_11_threshold_hough(
        hough_space, angles, distances,
        params['angle_min'], params['angle_max']
    )
    results['step11'] = {
        'binary_hough': binary_hough,
        'filtered_region': filtered_region
    }

    # Step 12: Locate Lines
    print("  Step 12: Locating horizontal lines...")
    line_overlay, detected_lines = step_12_locate_lines(
        img_gray, hough_space, angles, distances,
        params['num_peaks'], params['angle_min'], params['angle_max']
    )
    results['step12'] = {
        'line_overlay': line_overlay,
        'detected_lines': detected_lines
    }

    # Step 13: Projection and Boundaries
    print("  Step 13: Computing projection and boundaries...")
    boundary_overlay, boundaries, projection = step_13_projection_boundaries(
        img_gray, detected_lines, params['search_window']
    )
    results['step13'] = {
        'boundary_overlay': boundary_overlay,
        'boundaries': boundaries,
        'projection': projection
    }

    print(f"  ✓ Detected {len(detected_lines)} lines")
    print(f"  ✓ Found {len(boundaries)} boundaries")
    print(f"  ✓ Analyzed {len(particles)} particles")

    return results

def save_submission_files(image_name, results, output_dir):
    """Save all required submission files for one image"""

    # Step 1: Original grayscale
    step1_dir = output_dir / 'Step1_Original'
    step1_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step1_dir / f'{image_name}_grayscale.png'), results['original_gray'])

    # Step 2: Edge Detection
    step2_dir = output_dir / 'Step2_EdgeDetection'
    step2_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step2_dir / f'{image_name}_east_edges.png'), results['step2']['east_edges'])
    cv2.imwrite(str(step2_dir / f'{image_name}_west_edges.png'), results['step2']['west_edges'])
    cv2.imwrite(str(step2_dir / f'{image_name}_combined_edges.png'), results['step2']['combined_edges'])
    save_kernel_visualization(results['step2']['kernels'], step2_dir / f'{image_name}_kernels.png')

    # Step 3: Morphology
    step3_dir = output_dir / 'Step3_Morphology'
    step3_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step3_dir / f'{image_name}_after_dilation.png'), results['step3']['dilated'])
    cv2.imwrite(str(step3_dir / f'{image_name}_after_erosion.png'), results['step3']['eroded'])
    save_se_visualization(results['step3']['structuring_elements'], step3_dir / f'{image_name}_structuring_elements.png')

    # Step 4: Denoising
    step4_dir = output_dir / 'Step4_Denoising'
    step4_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step4_dir / f'{image_name}_denoised.png'), results['step4']['denoised'])
    with open(step4_dir / f'{image_name}_filter_info.txt', 'w') as f:
        f.write("DENOISING FILTER INFORMATION\n")
        f.write("="*50 + "\n\n")
        for key, value in results['step4']['filter_info'].items():
            f.write(f"{key}: {value}\n")

    # Step 5: Bandpass
    step5_dir = output_dir / 'Step5_BandpassFilter'
    step5_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step5_dir / f'{image_name}_bandpass_filtered.png'), results['step5']['filtered'])
    with open(step5_dir / f'{image_name}_bandpass_params.txt', 'w') as f:
        f.write("BANDPASS FILTER PARAMETERS\n")
        f.write("="*50 + "\n\n")
        for key, value in results['step5']['params'].items():
            f.write(f"{key}: {value}\n")

    # Step 6: Binary Mask
    step6_dir = output_dir / 'Step6_BinaryMasking'
    step6_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step6_dir / f'{image_name}_masked.png'), results['step6']['masked'])

    # Step 7: Second Bandpass
    step7_dir = output_dir / 'Step7_SecondBandpass'
    step7_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step7_dir / f'{image_name}_filtered_masked.png'), results['step7']['filtered_masked'])

    # Step 8: Particle Analysis
    step8_dir = output_dir / 'Step8_ParticleAnalysis'
    step8_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step8_dir / f'{image_name}_ellipses.png'),
                cv2.cvtColor(results['step8']['ellipse_image'], cv2.COLOR_RGB2BGR))
    with open(step8_dir / f'{image_name}_particles.json', 'w') as f:
        json.dump(results['step8']['particles'], f, indent=2)

    # Step 9: Skeletonization
    step9_dir = output_dir / 'Step9_Skeletonization'
    step9_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step9_dir / f'{image_name}_skeleton.png'), results['step9']['skeleton'])

    # Step 10: Hough Transform
    step10_dir = output_dir / 'Step10_HoughTransform'
    step10_dir.mkdir(exist_ok=True)
    save_hough_visualization(
        results['step10']['hough_space'],
        results['step10']['angles'],
        results['step10']['distances'],
        step10_dir / f'{image_name}_hough_space.png'
    )

    # Step 11: Threshold Hough
    step11_dir = output_dir / 'Step11_ThresholdHough'
    step11_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step11_dir / f'{image_name}_binary_hough.png'), results['step11']['binary_hough'])
    # Save filtered region as heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(results['step11']['filtered_region'], cmap='hot', aspect='auto')
    plt.title(f'Hough Space Filtered Around 90°', fontweight='bold')
    plt.colorbar(label='Votes')
    plt.savefig(step11_dir / f'{image_name}_filtered_region.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Step 12: Line Overlay
    step12_dir = output_dir / 'Step12_LocateLines'
    step12_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step12_dir / f'{image_name}_detected_lines.png'),
                cv2.cvtColor(results['step12']['line_overlay'], cv2.COLOR_RGB2BGR))
    with open(step12_dir / f'{image_name}_lines.json', 'w') as f:
        json.dump(results['step12']['detected_lines'], f, indent=2)

    # Step 13: Boundaries
    step13_dir = output_dir / 'Step13_Boundaries'
    step13_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(step13_dir / f'{image_name}_boundaries.png'),
                cv2.cvtColor(results['step13']['boundary_overlay'], cv2.COLOR_RGB2BGR))
    save_projection_plot(
        results['step13']['projection'],
        results['step13']['boundaries'],
        step13_dir / f'{image_name}_projection.png'
    )
    with open(step13_dir / f'{image_name}_boundaries.json', 'w') as f:
        json.dump(results['step13']['boundaries'], f, indent=2)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create main submission directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_root = SUBMISSION_DIR / f"Submission_{timestamp}"
    submission_root.mkdir(parents=True, exist_ok=True)

    print(f"Submission directory: {submission_root}")
    print(f"Parameters: {PARAMS}")
    print()

    # Save global parameters
    with open(submission_root / 'parameters.json', 'w') as f:
        json.dump(PARAMS, f, indent=2)

    with open(submission_root / 'parameters.txt', 'w') as f:
        f.write("PROCESSING PARAMETERS\n")
        f.write("="*50 + "\n\n")
        for key, value in PARAMS.items():
            f.write(f"{key:25s} = {value}\n")

    # Process each image
    all_results = {}

    for image_name in IMAGES_TO_PROCESS:
        # Find image file
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif']:
            test_path = IMAGE_DIR / f"{image_name}{ext}"
            if test_path.exists():
                image_path = test_path
                break

        if not image_path:
            print(f"WARNING: Could not find {image_name}")
            continue

        # Load image
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Process
        image_output_dir = submission_root / image_name
        results = process_single_image(image_name, img_gray, img_rgb, PARAMS, image_output_dir)
        all_results[image_name] = results

        # Save submission files
        print(f"  Saving submission files...")
        save_submission_files(image_name, results, image_output_dir)
        print(f"  ✓ Saved to: {image_output_dir}")

    summary_path = submission_root / 'SUMMARY_REPORT.txt'
    with open(summary_path, 'w') as f:
        f.write("PROCESSING PARAMETERS\n")
        f.write("-"*70 + "\n")
        for key, value in PARAMS.items():
            f.write(f"  {key:25s} = {value}\n")
        f.write("\n")

        f.write("RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")

        for image_name, results in all_results.items():
            f.write(f"{image_name}:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Lines detected:    {len(results['step12']['detected_lines'])}\n")
            f.write(f"  Boundaries found:  {len(results['step13']['boundaries'])}\n")
            f.write(f"  Particles analyzed: {len(results['step8']['particles'])}\n")
            f.write("\n  Detected Lines:\n")
            for i, line in enumerate(results['step12']['detected_lines'], 1):
                f.write(f"    Line {i}: rho={line['rho']:.1f}, angle={line['angle_deg']:.1f}°, votes={line['votes']}\n")
            f.write("\n")

    print(f"Summary report saved: {summary_path}")

    # Create README for submission
    readme_path = submission_root / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write("FOLDER STRUCTURE:\n")
        f.write("-"*70 + "\n\n")

        for image_name in IMAGES_TO_PROCESS:
            if image_name in all_results:
                f.write(f"{image_name}/\n")
                for step_num in range(1, 14):
                    step_name = f"Step{step_num}"
                    f.write(f"  ├── {step_name}_*/ (outputs for step {step_num})\n")
                f.write("\n")

        f.write("\nFILES:\n")
        f.write("-"*70 + "\n")
        f.write("  parameters.json - Processing parameters (JSON format)\n")
        f.write("  parameters.txt - Processing parameters (human-readable)\n")
        f.write("  SUMMARY_REPORT.txt - Complete results summary\n")
        f.write("  README.txt - This file\n")
        f.write("\n")

        f.write("STEP-BY-STEP OUTPUTS:\n")
        f.write("-"*70 + "\n")
        f.write("  Step 1: Original grayscale images\n")
        f.write("  Step 2: Edge detection (east, west, combined) + kernels\n")
        f.write("  Step 3: Morphology (dilation, erosion) + structuring elements\n")
        f.write("  Step 4: Denoised images + filter information\n")
        f.write("  Step 5: Bandpass filtered images + parameters\n")
        f.write("  Step 6: Binary masked images\n")
        f.write("  Step 7: Second bandpass filtered images\n")
        f.write("  Step 8: Particle analysis with ellipses + particle data\n")
        f.write("  Step 9: Skeletonized images\n")
        f.write("  Step 10: Hough transform visualizations\n")
        f.write("  Step 11: Thresholded Hough space\n")
        f.write("  Step 12: Detected lines overlaid on original + line data\n")
        f.write("  Step 13: Text line boundaries + projection profiles\n")

    print("\n" + "="*70)
    print("SUBMISSION GENERATION COMPLETE!")
    print("="*70)
    print(f"\nSubmission folder: {submission_root}")
    print(f"\nProcessed {len(all_results)} images:")
    for image_name in all_results.keys():
        print(f"  ✓ {image_name}")
    print(f"\nTotal files generated: Check {submission_root}")
    print()

if __name__ == '__main__':
    main()
