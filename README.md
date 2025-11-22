# PPM Digit Extractors

Two implementations for extracting digits from P3 (ASCII) PPM images without external libraries.

## Files

### `ppm_digits_extractor.py` (Template-based)
Production-quality extractor using exact pixel pattern matching.

**Features:**
- Template dictionaries for digits 0-9 with variants
- Letter recognition (H, E, A, l, o) for disambiguation
- Context-aware classification
- Smart segment merging for multi-stroke characters
- Column-based segmentation

**Usage:**
```bash
# Extract all characters
python3 ppm_digits_extractor.py input.ppm

# Extract only digits
python3 ppm_digits_extractor.py input.ppm --digits-only

# Verbose output with patterns
python3 ppm_digits_extractor.py input.ppm --verbose
```

### `simple_extractor.py` (Heuristic-based)
Interview-friendly simplified version using geometric features and pattern templates.

**Features:**
- 3×5 digit pattern templates (0-9)
- Letter pattern filtering (H, E, A, l, o)
- Hole detection for 0, 6, 8, 9
- Aspect ratio and density fallbacks
- Context-aware filtering with `--digits-only`

**Usage:**
```bash
# Extract all characters
python3 simple_extractor.py input.ppm

# Extract only digits (filters ambiguous letter look-alikes)
python3 simple_extractor.py input2.ppm --digits-only
```

## Test Images

- **input.ppm** (20×10): Contains digits "12"
  - 4 connected components that merge into 2 characters
  
- **input2.ppm** (60×12): Contains "H5el9lo6"
  - Expected digits-only output: "596"
  - 8 components: letters H, E, l (×2), o mixed with digits 5, 9, 6

## Implementation Notes

### Template-based Approach (`ppm_digits_extractor.py`)
- **Pros:** Highly accurate, handles multi-stroke digits
- **Cons:** ~300 lines, complex merging logic
- **Best for:** Production use, accuracy-critical applications

### Heuristic Approach (`simple_extractor.py`)
- **Pros:** ~180 lines, easier to explain in interviews
- **Cons:** May struggle with unusual fonts or noise
- **Best for:** Technical interviews, simple datasets

## Algorithm Overview

Both extractors follow these steps:

1. **Load PPM**: Parse P3 format, convert RGB to grayscale
2. **Binarize**: Threshold at 128 (1=black, 0=white)
3. **Find Components**: Flood-fill DFS for connected pixels
4. **Sort**: Order components left-to-right by x-coordinate
5. **Classify**: Use templates or heuristics to identify digits
6. **Filter**: Optional removal of letter look-alikes (l→1, o→0)
7. **Merge**: Combine multi-part strokes into single digits

## Requirements

- Python 3.x (no external libraries)
- P3 (ASCII) PPM format images
- Black text on white background

## Testing

```bash
# Template-based extractor
python3 ppm_digits_extractor.py input.ppm --digits-only
# Output: 12

python3 ppm_digits_extractor.py input2.ppm --digits-only
# Output: 596

# Simple heuristic extractor
python3 simple_extractor.py input.ppm --digits-only
# Output: 12

python3 simple_extractor.py input2.ppm --digits-only
# Output: 596
```
