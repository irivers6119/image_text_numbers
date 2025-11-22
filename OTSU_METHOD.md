# Otsu's Method: Automatic Thresholding

**Otsu's method** is an automatic thresholding algorithm that finds the optimal threshold value to separate an image into foreground and background without manual tuning.

## The Problem It Solves

With a simple fixed approach, we hardcode the threshold:
```python
binary = 1 if pixel < 128 else 0
```

This works when lighting is consistent, but fails when:
- Image is too dark (digits might be gray, not pure black)
- Image is too bright (background might be light gray)
- Uneven lighting (shadows on one side)
- Different cameras/scanners with varying contrast

## How Otsu's Method Works

It automatically finds the best threshold by **maximizing the variance between the two classes** (foreground vs background).

### Algorithm Steps

1. **Calculate histogram** of all pixel intensities (0-255)
2. **For each possible threshold** t (0-255):
   - Split pixels into two groups: below threshold and above threshold
   - Calculate the mean intensity of each group
   - Calculate variance within each group
   - Calculate variance between the two groups
3. **Choose threshold** that maximizes the **between-class variance**

### Mathematical Intuition

**Good threshold:** Two distinct peaks in histogram (dark text cluster, light background cluster)
- High variance *between* the two groups
- Low variance *within* each group

**Bad threshold:** Cuts through middle of a cluster
- Low variance between groups
- High variance within groups

The between-class variance is calculated as:
```
σ²(t) = w₀(t) · w₁(t) · [μ₀(t) - μ₁(t)]²

where:
  w₀ = weight (proportion) of background class
  w₁ = weight (proportion) of foreground class
  μ₀ = mean intensity of background
  μ₁ = mean intensity of foreground
```

## Implementation

### Basic Version

```python
def otsu_threshold(img):
    """Find optimal threshold using Otsu's method."""
    # Flatten image to 1D array
    pixels = [pixel for row in img for pixel in row]
    
    # Calculate histogram (count of each intensity 0-255)
    histogram = [0] * 256
    for pixel in pixels:
        histogram[pixel] += 1
    
    total = len(pixels)
    
    # Try all possible thresholds, find one with max between-class variance
    best_threshold = 0
    max_variance = 0
    
    sum_total = sum(i * histogram[i] for i in range(256))
    sum_bg = 0
    weight_bg = 0
    
    for t in range(256):
        # Update background weight and sum
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        
        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        
        # Calculate between-class variance
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        
        if variance > max_variance:
            max_variance = variance
            best_threshold = t
    
    return best_threshold


# Usage in your extractor
def binarize_adaptive(img):
    """Binarize using automatic threshold."""
    threshold = otsu_threshold(img)
    return [[1 if pixel < threshold else 0 for pixel in row] for row in img]
```

### Optimized Version (Single Pass)

```python
def otsu_threshold_fast(img):
    """Optimized Otsu's method with single histogram pass."""
    # Build histogram
    histogram = [0] * 256
    for row in img:
        for pixel in row:
            histogram[pixel] += 1
    
    total = sum(histogram)
    
    # Calculate global mean
    sum_total = sum(i * histogram[i] for i in range(256))
    
    sum_bg = 0
    weight_bg = 0
    max_variance = 0
    threshold = 0
    
    for i in range(256):
        weight_bg += histogram[i]
        if weight_bg == 0:
            continue
        
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        
        sum_bg += i * histogram[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        
        # Between-class variance
        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = i
    
    return threshold
```

## Example: Before and After

### Fixed Threshold (128)
```
Dark image with gray digits:
  Pixel values: 40, 45, 50 (digits), 100, 105, 110 (background)
  Threshold 128: Everything becomes background! ❌

Bright image:
  Pixel values: 200, 205, 210 (digits), 245, 250, 255 (background)
  Threshold 128: Everything becomes foreground! ❌
```

### Otsu's Threshold (Adaptive)
```
Dark image:
  Histogram peaks at ~45 and ~105
  Otsu finds threshold ≈ 75
  Correctly separates digits from background ✓

Bright image:
  Histogram peaks at ~205 and ~250
  Otsu finds threshold ≈ 227
  Correctly separates digits from background ✓
```

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Build histogram | O(W × H) | O(256) = O(1) |
| Try all thresholds | O(256) = O(1) | O(1) |
| **Total** | **O(W × H)** | **O(1)** |

The method adds negligible overhead — the histogram pass is linear in image size, and trying 256 thresholds is constant time.

## When to Use Otsu's Method

### ✅ Good For:
- Unknown lighting conditions
- Varying image quality (different scanners/cameras)
- Production systems with diverse inputs
- Outdoor scenes with variable illumination
- Historical documents with aging/fading

### ❌ Not Needed For:
- Controlled environments (fixed scanner, studio lighting)
- High-contrast images (pure black on pure white)
- Interview problems with simple test cases
- Real-time systems where fixed threshold is "good enough"

### ⚠️ Limitations:
- Assumes **bimodal histogram** (two distinct peaks)
- Fails with:
  - Very small text (few foreground pixels)
  - Uniform images (no clear separation)
  - Multi-colored text on multi-colored backgrounds
- For complex cases, use **multi-level Otsu** or **local adaptive thresholding**

## Interview Discussion Points

Mentioning Otsu's method demonstrates you understand:

1. **Adaptive algorithms** that adjust to input characteristics
2. **Statistical methods** in computer vision (variance, histogram analysis)
3. **Trade-offs** between simplicity (fixed threshold) and robustness (adaptive)
4. **When optimization matters** and when it doesn't

### Sample Interview Exchange

**Interviewer:** "Your threshold is hardcoded at 128. What if the image is darker?"

**You:** "Great question! For production, I'd use Otsu's method — it automatically finds the optimal threshold by analyzing the histogram. It maximizes the variance between foreground and background classes. The complexity is still O(W×H) since we just add one histogram pass. For this interview problem with controlled test images, the fixed threshold keeps the code simple and focuses on the core algorithm."

## Extensions

### Multi-Level Otsu
Find multiple thresholds for images with more than two intensity levels (e.g., text + graphics + background):

```python
def otsu_multi_level(img, num_thresholds=2):
    """Find multiple optimal thresholds."""
    # Similar approach but maximize sum of between-class variances
    # across all classes
    # Complexity: O(256^k × W×H) where k = num_thresholds
    pass
```

### Local Adaptive Thresholding
Use different thresholds for different regions (handles uneven lighting):

```python
def adaptive_threshold(img, window_size=15):
    """Apply Otsu's method to local windows."""
    # Divide image into overlapping windows
    # Apply Otsu to each window independently
    # Good for images with gradients, shadows
    pass
```

## References

- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms". *IEEE Transactions on Systems, Man, and Cybernetics*. 9 (1): 62–66.
- Used in: OpenCV (`cv2.threshold` with `cv2.THRESH_OTSU`), scikit-image (`threshold_otsu`)

## See Also

- [Main README](README.md) - Complete digit extractor documentation
- [simple_extractor.py](simple_extractor.py) - Implementation using fixed threshold
