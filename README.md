# PPM Digit Extractor

A simple, interview-friendly Python implementation that extracts digits from PPM images **without using any external libraries** (no PIL, OpenCV, or NumPy).

Perfect for technical interviews where you need to demonstrate image processing fundamentals from scratch.

## Design Choices

**Why PPM format?** To keep the solution simple and focused on algorithms rather than format parsing:
- PPM (Portable Pixmap) is the simplest image format — just ASCII text with RGB values
- No compression, no complex headers, no bit manipulation required
- Can be read with basic file I/O operations
- Easy to create test images programmatically or convert with tools like ImageMagick

**Why black text on white background?** To avoid additional image preprocessing:
- Simple thresholding (pixel < 128 → black) works reliably
- No need for edge detection, gradient analysis, or adaptive thresholding
- Inverted colors would require minimal code changes (just flip the threshold condition)
- These constraints weren't part of the requirements, so we optimized for clarity

**The core challenge** is the digit recognition algorithm, not format parsing — this keeps the interview focused on the interesting parts!

## Quick Start

```bash
# Extract digits from an image
python3 simple_extractor.py input.ppm

# Extract only digits (filter out letters)
python3 simple_extractor.py input2.ppm --digits-only
```

**Expected outputs:**
- `input.ppm` → `12`
- `input2.ppm --digits-only` → `596`

---

## How It Works: Step-by-Step Walkthrough

This section explains `simple_extractor.py` in interview-friendly terms, perfect for whiteboard discussions.

### Step 1: Load the PPM Image

**What is PPM?** P3 (ASCII) PPM is one of the simplest image formats:
```
P3
20 10        # width height
255          # max color value
255 255 255  # RGB triplets for each pixel
0 0 0
...
```

**Our approach:**
```python
def load_ppm(filename):
    # 1. Read file, skip comments (lines starting with #)
    # 2. Parse header: P3, width, height, max_val
    # 3. Read RGB triplets: [R, G, B, R, G, B, ...]
    # 4. Convert to grayscale: gray = (R + G + B) // 3
    # 5. Return 2D array of grayscale values
```

**Interview tip:** Mention you could extend this to handle P6 (binary) PPM for better performance.

---

### Step 2: Binarize (Convert to Black & White)

**Goal:** Simplify image to pure black (1) and white (0) pixels.

```python
def binarize(img, threshold=128):
    # If pixel < 128 → black (text)
    # If pixel >= 128 → white (background)
    return [[1 if pixel < 128 else 0 for pixel in row] for row in img]
```

**Why threshold=128?** Middle of 0-255 range works well for black text on white background.

**Interview tip:** Discuss adaptive thresholding (Otsu's method) as an improvement for varying lighting.

---

### Step 3: Find Connected Components (Characters)

**Problem:** Identify each separate character in the image.

**Solution:** Flood-fill algorithm (iterative DFS with a stack)

```python
def find_components(binary_img):
    visited = [[False] * width for _ in range(height)]
    
    def flood_fill(x, y, component):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            # Skip if out of bounds, visited, or white pixel
            if visited[cy][cx] or binary_img[cy][cx] == 0:
                continue
            visited[cy][cx] = True
            component.append((cx, cy))  # Add pixel to component
            # Add 4 neighbors to stack
            stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    
    # Scan image, start flood-fill at each unvisited black pixel
    components = []
    for y in range(height):
        for x in range(width):
            if binary_img[y][x] == 1 and not visited[y][x]:
                comp = []
                flood_fill(x, y, comp)
                components.append(comp)  # List of (x,y) coordinates
    return components
```

**Complexity:** O(width × height) — we visit each pixel once.

**Interview tip:** Explain why we use iterative DFS (avoid stack overflow) and 4-connectivity vs 8-connectivity.

---

### Step 4: Classify Each Component as a Digit

**Challenge:** Recognize which digit (0-9) each component represents.

**Strategy 1: Template Matching** (primary approach)

Store exact pixel patterns for 3×5 digit glyphs:

```python
digit_patterns = {
    '111\n1.1\n1.1\n1.1\n111': '0',  # Box shape
    '.1.\n.1.\n.1.\n.1.\n111': '1',  # Vertical line with base
    '111\n..1\n111\n1..\n111': '2',  # S-curve
    '111\n1..\n111\n..1\n111': '5',  # Reversed S
    # ... etc for 0-9
}
```

**How it works:**
1. Extract bounding box of component
2. Build mini-grid (relative coordinates)
3. Convert to string pattern (`'1'` = black, `'.'` = white)
4. Look up pattern in dictionary

**Interview tip:** Discuss trade-offs:
- **Pros:** Simple, fast lookup O(1), works great for fixed fonts
- **Cons:** Brittle to font changes, requires exact matches

---

**Strategy 2: Geometric Heuristics** (fallback)

When pattern doesn't match exactly, use shape analysis:

```python
def classify_digit(component):
    # Calculate bounding box
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    aspect_ratio = width / height
    
    # Feature 1: Narrow digits
    if aspect_ratio < 0.5 or width <= 2:
        return '1'  # Very narrow → likely a 1
    
    # Feature 2: Holes (enclosed white regions)
    holes = count_holes(grid)
    if holes == 1:
        return '0'  # or '6', '8', '9' based on position
    
    # Feature 3: Row analysis for 3×5 grids
    if width == 3 and height == 5:
        # Check which sides are filled in key rows
        row1_left = grid[1][0] == 1
        row1_right = grid[1][2] == 1
        row3_left = grid[3][0] == 1
        row3_right = grid[3][2] == 1
        
        # Pattern logic:
        # 5: left-only in row1, right-only in row3
        # 9: both sides row1, right-only row3
        # 6: left-only row1, both sides row3
        if row1_left and not row1_right and not row3_left and row3_right:
            return '5'
        # ... etc
```

**Key geometric features:**
- **Aspect ratio:** Distinguishes narrow '1' from square digits
- **Density:** Pixel count / area (high for '8', low for '7')
- **Holes:** Count enclosed white regions using flood-fill
- **Row patterns:** Which sides are filled at key heights

**Interview tip:** Explain this as a "decision tree" and discuss how ML (CNN) would learn these features automatically.

---

### Step 5: Handle Ambiguous Cases (Letters vs Digits)

**Problem:** Letters 'l' and 'o' look like digits '1' and '0'.

**Solution:** Context-aware filtering with `--digits-only` flag

```python
if digits_only:
    # Pass 1: Mark "confident" digits (2-9, never confused with letters)
    confident = set()
    for i, (_, _, digit) in enumerate(classified):
        if digit in '23456789':
            confident.add(i)
    
    # Pass 2: Filter ambiguous 0 and 1
    for i, (_, _, digit) in enumerate(classified):
        if digit == '1':
            # Keep '1' only if:
            # - Adjacent to another '1' (multi-stroke digit)
            # - Between two confident digits
            # - First character followed by confident digit
            if not (prev_is_one or next_is_one or 
                    (prev_confident and next_confident) or
                    (i == 0 and next_confident)):
                continue  # Reject as letter 'l'
        
        if digit == '0':
            # Keep '0' only if between two confident digits
            if not (prev_confident and next_confident):
                continue  # Reject as letter 'o'
```

**Interview tip:** This demonstrates **context modeling** — using surrounding information to resolve ambiguity, similar to how NLP models use context for word disambiguation.

---

### Step 6: Merge Multi-Part Digits

**Problem:** Some digits (like '1') may be drawn with multiple disconnected strokes.

**Solution:** Group adjacent narrow components

```python
# If two components are close (gap ≤ 2 pixels) and both narrow (width ≤ 3),
# and classified as same digit, merge them
if digit1 == digit2 and gap <= 2 and width1 <= 3 and width2 <= 3:
    merged_digits.append(digit1)  # Keep only one copy
    i += 2  # Skip both components
```

**Interview tip:** Explain this handles "serifs" and imperfect image quality where digits may have small gaps.

---

## Complete Algorithm Summary

```
INPUT: PPM image file
OUTPUT: String of digits

1. LOAD_PPM(filename)
   └─> Parse P3 format → RGB array → Grayscale

2. BINARIZE(grayscale, threshold=128)
   └─> Black (1) / White (0) array

3. FIND_COMPONENTS(binary)
   └─> Flood-fill DFS → List of pixel groups

4. SORT(components, by x-coordinate)
   └─> Left-to-right reading order

5. CLASSIFY(each component)
   ├─> Try template matching (exact patterns)
   ├─> Fall back to geometric heuristics
   └─> Return digit 0-9 or '?' for unknown

6. FILTER(if --digits-only)
   └─> Remove letter look-alikes using context

7. MERGE(adjacent identical narrow digits)
   └─> Combine multi-stroke characters

8. RETURN(joined digit string)
```

**Total Complexity:** O(W × H) where W=width, H=height
- Each pixel visited once in binarization
- Each pixel visited once in component finding
- Classification is O(component_size) per component

---

## Interview Discussion Points

### Design Decisions

**Q: Why no external libraries?**
- Demonstrates understanding of fundamentals
- Shows ability to implement algorithms from scratch
- Useful for embedded systems / constrained environments

**Q: Why template matching over ML?**
- Simple, fast, interpretable
- No training data needed
- Perfect for fixed fonts (OCR-A, digital displays)
- Can explain entire algorithm in 30 minutes

**Q: How would you improve this?**
- **Robustness:** Add rotation/skew correction, scale normalization
- **Fonts:** Train ML model (CNN) for arbitrary fonts
- **Speed:** Use spatial hashing for large images
- **Accuracy:** Ensemble multiple classifiers, use Tesseract OCR

### Extending the Solution

**1. Handle rotated text:**
```python
# Compute principal component of black pixels
# Rotate image to align text horizontally
```

**2. Multi-line text:**
```python
# Project pixels horizontally to find row separations
# Process each line independently
```

**3. Arbitrary fonts:**
```python
# Replace template matching with:
# - Feature extraction (HOG, SIFT)
# - SVM or CNN classifier
```

**4. Real-time processing:**
```python
# Use sliding window
# GPU acceleration for parallel flood-fill
```

---

## Usage Examples

```bash
# Basic usage
python3 simple_extractor.py input.ppm
# Output: 12

# Filter out letters, keep only digits
python3 simple_extractor.py input2.ppm --digits-only
# Output: 596 (from "H5el9lo6")

# No flag on mixed content
python3 simple_extractor.py input2.ppm
# Output: 519106 (includes letter-like shapes)
```

## Test Images

- **input.ppm** (20×10): Contains digits "12"
- **input2.ppm** (60×12): Contains "H5el9lo6" → digits-only output "596"
- Corresponding `.jpg` files for easy viewing

## Code Structure

```python
# simple_extractor.py (~180 lines, well-commented)

1. load_ppm()           # Parse P3 format → grayscale array
2. binarize()           # Threshold → binary image
3. find_components()    # Flood-fill → connected regions
4. classify_digit()     # Templates + heuristics → '0'-'9' or '?'
5. count_holes()        # Detect enclosed regions
6. extract_digits()     # Main pipeline with filtering
7. main()               # CLI argument parsing
```

## Requirements

- **Python 3.x** (no external libraries!)
- **Input:** P3 (ASCII) PPM format images
- **Assumption:** Black text on white background

---

## Time/Space Complexity

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Load PPM | O(W×H) | O(W×H) | Parse entire image |
| Binarize | O(W×H) | O(W×H) | Single pass threshold |
| Find Components | O(W×H) | O(W×H) | Each pixel visited once |
| Classify | O(C×K) | O(K) | C=components, K=avg size |
| **Total** | **O(W×H)** | **O(W×H)** | Linear in image size |

**Optimization opportunities:**
- Use run-length encoding for sparse images
- Parallel flood-fill on GPU
- Early termination in template matching

---

## License

MIT - Free to use in interviews and projects!
