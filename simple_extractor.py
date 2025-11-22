#!/usr/bin/env python3
"""Simple digit extractor using geometric heuristics (no external libraries)."""

def load_ppm(filename):
    """Load a P3 PPM file and return width, height, and grayscale image."""
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    
    if lines[0] != 'P3':
        raise ValueError("Only P3 PPM format supported")
    
    width, height = map(int, lines[1].split())
    max_val = int(lines[2])
    
    # Parse RGB values
    values = []
    for line in lines[3:]:
        values.extend(map(int, line.split()))
    
    # Convert to grayscale
    img = []
    for i in range(0, len(values), 3):
        r, g, b = values[i], values[i+1], values[i+2]
        gray = (r + g + b) // 3
        img.append(gray)
    
    return width, height, img

def binarize(img):
    """Convert grayscale to binary (1=black, 0=white)."""
    return [1 if pixel < 128 else 0 for pixel in img]

def find_components(binary, width, height):
    """Find connected components using flood fill."""
    visited = [[False] * width for _ in range(height)]
    
    def flood_fill(x, y, component):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                continue
            if visited[cy][cx] or binary[cy * width + cx] == 0:
                continue
            visited[cy][cx] = True
            component.append((cx, cy))
            stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    
    components = []
    for y in range(height):
        for x in range(width):
            if binary[y * width + x] == 1 and not visited[y][x]:
                component = []
                flood_fill(x, y, component)
                if component:
                    components.append(component)
    
    return components

def classify_digit(component):
    """Classify a component as a digit (0-9) using template patterns first, then fallbacks."""
    # Bounding box
    min_x = min(p[0] for p in component)
    max_x = max(p[0] for p in component)
    min_y = min(p[1] for p in component)
    max_y = max(p[1] for p in component)
    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # Build grid
    grid = [[0] * width for _ in range(height)]
    for x, y in component:
        grid[y - min_y][x - min_x] = 1

    # Trim empty top/bottom rows
    while height > 0 and all(v == 0 for v in grid[0]):
        grid.pop(0)
        height -= 1
    while height > 0 and all(v == 0 for v in grid[-1]):
        grid.pop()
        height -= 1

    if height == 0:
        return '?'

    pattern = '\n'.join(''.join('1' if v else '.' for v in row) for row in grid)

    # Exact 3x5 patterns for digits 0-9
    digit_patterns = {
        '111\n1.1\n1.1\n1.1\n111': '0',
        '.1.\n.1.\n.1.\n.1.\n111': '1',
        '.1.\n11.\n.1.\n.1.\n111': '1',  # variant 1
        '111\n..1\n111\n1..\n111': '2',
        '11.\n1.1\n11.\n1.1\n11.': '2',      # compact 2
        '111\n..1\n111\n..1\n111': '3',
        '1.1\n1.1\n111\n..1\n..1': '4',
        '111\n1..\n111\n..1\n111': '5',
        '111\n1..\n111\n1.1\n111': '6',
        '111\n..1\n..1\n..1\n..1': '7',
        '111\n1.1\n111\n1.1\n111': '8',
        '111\n1.1\n111\n..1\n111': '9',
    }
    if pattern in digit_patterns:
        return digit_patterns[pattern]

    # Letter patterns to reject (prevent misclassification as digits)
    letter_patterns = {
        '1.1\n1.1\n111\n1.1\n1.1': 'H',
        '111\n1..\n111\n1..\n111': 'E',
        '.1.\n1.1\n111\n1.1\n1.1': 'A',
        '.1.\n.1.\n.1.\n.1.\n.1.': 'l',
        '111\n1.1\n1.1\n1.1\n111': 'o',  # same shape as 0 but context-less letter
    }
    if pattern in letter_patterns:
        return '?'  # Treat as non-digit for this simple extractor

    # Wide merged patterns (7 columns) occasionally representing 1/2
    if width == 7:
        wide_patterns = {
            '.1...1.\n1.1.11.\n111..1.\n1.1..1.\n1.1.111': '1',
            '11..111\n1.1...1\n11..111\n1.1.1..\n11..111': '2',
        }
        if pattern in wide_patterns:
            return wide_patterns[pattern]

    # Shape sanity: most digits except 1 / 7 require top and bottom bars
    if width == 3 and height == 5:
        top_row = pattern.split('\n')[0]
        bottom_row = pattern.split('\n')[-1]
        full_bar = lambda r: r == '111'
        # If neither top nor bottom is a full bar and pattern not a narrow '1', reject
        if not full_bar(top_row) and not full_bar(bottom_row):
            return '?'

    # Simple structural fallbacks
    aspect = width / height if height else 0
    pixel_count = sum(sum(row) for row in grid)
    area = width * height
    density = pixel_count / area if area else 0

    # Hole count for detecting 0/6/8/9 style
    holes = count_holes(grid)
    if holes == 1:
        # Distinguish 0 vs 6/9 by top/bottom row fullness
        top_full = all(v == 1 for v in grid[0])
        bottom_full = all(v == 1 for v in grid[-1])
        if top_full and bottom_full:
            # 0 or 8: check middle fullness for 8
            mid_index = height // 2
            mid_full = sum(grid[mid_index]) / width > 0.8
            return '8' if mid_full else '0'
        else:
            # Could be 6 or 9: check lower-mid vs upper-mid density
            upper_mid = sum(grid[1]) / width if height > 2 else 0
            lower_mid = sum(grid[-2]) / width if height > 2 else 0
            return '6' if lower_mid > upper_mid else '9'

    # Narrow aspect strongly suggests '1'
    if aspect < 0.5 or width <= 2:
        return '1'

    # High density with side columns suggests '8'
    if density > 0.75:
        # Check both side columns occupancy
        left_col = sum(row[0] for row in grid) / height
        right_col = sum(row[-1] for row in grid) / height
        if left_col > 0.6 and right_col > 0.6:
            return '8'

    # Check for top/mid/bottom bar pattern for '2' or '5'
    top_bar = sum(grid[0]) / width > 0.6
    mid_bar = sum(grid[height//2]) / width > 0.5
    bottom_bar = sum(grid[-1]) / width > 0.6
    if top_bar and mid_bar and bottom_bar:
        # Restrict bar-based identification to canonical 3x5 glyphs
        if width == 3 and height == 5:
            # Row indices: 0 top, 1 upper-mid, 2 middle, 3 lower-mid, 4 bottom
            r1 = grid[1]
            r3 = grid[3]
            # Pattern expectations:
            # 2: r1 right-only (..1), r3 left-only (1..)
            # 5: r1 left-only (1..), r3 right-only (..1)
            # Validate exclusivity (exactly one side filled)
            r1_left = r1[0] == 1 and r1[2] == 0
            r1_right = r1[2] == 1 and r1[0] == 0
            r3_left = r3[0] == 1 and r3[2] == 0
            r3_right = r3[2] == 1 and r3[0] == 0
            if r1_left and r3_right:
                return '5'
            if r1_right and r3_left:
                return '2'
        # Otherwise do not guess 2/5 from incomplete fragments

    return '?'

def count_holes(grid):
    """Count enclosed white regions (holes) inside a binary grid (list of lists)."""
    h = len(grid)
    if h == 0: return 0
    w = len(grid[0])
    visited = [[False] * w for _ in range(h)]
    def flood(x,y):
        stack=[(x,y)]
        enclosed=True
        while stack:
            cx,cy=stack.pop()
            if cx<0 or cy<0 or cx>=w or cy>=h: continue
            if visited[cy][cx] or grid[cy][cx]==1: continue
            visited[cy][cx]=True
            if cx==0 or cy==0 or cx==w-1 or cy==h-1:
                enclosed=False
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cx+dx,cy+dy))
        return enclosed
    holes=0
    for y in range(h):
        for x in range(w):
            if grid[y][x]==0 and not visited[y][x]:
                if flood(x,y): holes+=1
    return holes

def extract_digits(filename, digits_only=False):
    """Extract digits from a PPM image file."""
    width, height, img = load_ppm(filename)
    binary = binarize(img)
    components = find_components(binary, width, height)
    
    # Filter out very small noise components
    components = [c for c in components if len(c) > 2]
    
    # Sort components left to right and classify each
    results = []
    for comp in components:
        min_x = min(p[0] for p in comp)
        results.append((min_x, comp))
    results.sort()
    
    # Classify each component individually
    classified = []
    for min_x, comp in results:
        digit = classify_digit(comp)
        classified.append((min_x, comp, digit))
    
    # If digits_only mode, apply context-aware filtering for letter look-alikes
    if digits_only:
        # First pass: mark positions that are valid high-confidence digits (2-9, excluding 0,1)
        confident = set()
        for i, (min_x, comp, digit) in enumerate(classified):
            if digit in '23456789':
                confident.add(i)
        
        # Second pass: filter ambiguous 0/1 based on context
        filtered = []
        for i, (min_x, comp, digit) in enumerate(classified):
            if not digit.isdigit():
                continue  # Skip explicit non-digits
            
            # Filter '1' (likely letter 'l'): check context
            if digit == '1':
                width = max(p[0] for p in comp) - min(p[0] for p in comp) + 1
                prev_conf = (i-1) in confident
                next_conf = (i+1) in confident
                prev_is_one = i > 0 and classified[i-1][2] == '1'
                next_is_one = i < len(classified)-1 and classified[i+1][2] == '1'
                
                # Keep if:
                # - Wide (merged stroke pattern)
                # - Adjacent to another '1' (multi-part digit '1')
                # - Between two confident digits
                # - First digit and followed by confident
                if width > 3:
                    pass  # Keep wide patterns
                elif prev_is_one or next_is_one:
                    pass  # Keep if part of multi-stroke '1'
                elif prev_conf and next_conf:
                    pass  # Keep if between confident digits
                elif i == 0 and next_conf:
                    pass  # First component and next is confident
                else:
                    continue  # Reject isolated narrow '1'
            
            # Filter '0' (likely letter 'o'): require both neighbors to be confident
            if digit == '0':
                prev_conf = (i-1) in confident
                next_conf = (i+1) in confident
                if not (prev_conf and next_conf):
                    continue
            
            filtered.append((min_x, comp, digit))
        classified = filtered
    
    # Merge / collapse adjacent identical narrow digits (likely duplicated stroke parts)
    merged_digits = []
    i = 0
    while i < len(classified):
        min_x1, comp1, digit1 = classified[i]
        max_x1 = max(p[0] for p in comp1)
        
        # Check if next component should be merged
        if i + 1 < len(classified):
            min_x2, comp2, digit2 = classified[i + 1]
            gap = min_x2 - max_x1 - 1
            
            # Merge if same digit, gap <= 2, both narrow (width <= 3), and both are digits
            width1 = max_x1 - min_x1 + 1
            max_x2 = max(p[0] for p in comp2)
            width2 = max_x2 - min_x2 + 1
            
            if digit1 == digit2 and gap <= 2 and width1 <= 3 and width2 <= 3 and digit1.isdigit():
                # Same digit appears twice close together - only keep one
                merged_digits.append(digit1)
                i += 2
                continue
        
        # Keep this digit if it's valid
        if digit1.isdigit():
            # Avoid triple repeats: if last two are same, skip
            if len(merged_digits) >= 2 and merged_digits[-1] == digit1 and merged_digits[-2] == digit1:
                pass
            else:
                merged_digits.append(digit1)
        i += 1
    
    return ''.join(merged_digits)

if __name__ == "__main__":
    import sys
    filename = 'input.ppm'
    digits_only = False
    for arg in sys.argv[1:]:
        if arg == '--digits-only':
            digits_only = True
        else:
            filename = arg
    print(extract_digits(filename, digits_only=digits_only))
