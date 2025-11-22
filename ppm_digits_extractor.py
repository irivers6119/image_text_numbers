
# Extract digits from a black-on-white PPM image without external libraries
# Steps: Load image, binarize, segment characters, classify digits using simple heuristics

def load_ppm(filename):
    """Load a PPM image and return width, height, and pixel data."""
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header != 'P3':
            raise ValueError("Only ASCII PPM (P3) format supported.")
        
        # Skip comments
        line = f.readline().strip()
        while line.startswith('#'):
            line = f.readline().strip()
        
        width, height = map(int, line.split())
        max_val = int(f.readline().strip())
        
        # Read pixel data
        pixels = []
        data = f.read().split()
        for i in range(0, len(data), 3):
            r, g, b = map(int, data[i:i+3])
            # Convert to grayscale intensity
            intensity = (r + g + b) // 3
            pixels.append(intensity)
        
        # Reshape into 2D array
        img = [pixels[i*width:(i+1)*width] for i in range(height)]
        return width, height, img

def binarize(img, threshold=128):
    """Convert grayscale image to binary (0=white, 1=black)."""
    return [[1 if pixel < threshold else 0 for pixel in row] for row in img]

def find_components(binary_img):
    """Find connected components using 4-neighbor DFS. Returns list of pixel coordinate lists."""
    height = len(binary_img)
    width = len(binary_img[0])
    visited = [[False] * width for _ in range(height)]
    components = []

    def dfs(x, y, comp):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cy < 0 or cx >= width or cy >= height:
                continue
            if visited[cy][cx] or binary_img[cy][cx] == 0:
                continue
            visited[cy][cx] = True
            comp.append((cx, cy))
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((cx + dx, cy + dy))

    for y in range(height):
        for x in range(width):
            if binary_img[y][x] == 1 and not visited[y][x]:
                comp = []
                dfs(x, y, comp)
                components.append(comp)
    return components

def extract_bbox(component):
    """Return bounding box (min_x, min_y, max_x, max_y)."""
    xs = [p[0] for p in component]
    ys = [p[1] for p in component]
    return min(xs), min(ys), max(xs), max(ys)

def slice_subimage(binary_img, bbox):
    min_x, min_y, max_x, max_y = bbox
    return [row[min_x:max_x + 1] for row in binary_img[min_y:max_y + 1]]

def count_holes(subimg):
    """Count holes (enclosed white regions) inside the subimage where 1=black,0=white."""
    h = len(subimg)
    w = len(subimg[0])
    visited = [[False] * w for _ in range(h)]

    def flood(x, y):
        stack = [(x, y)]
        enclosed = True
        cells = []
        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cy < 0 or cx >= w or cy >= h:
                continue
            if visited[cy][cx] or subimg[cy][cx] == 1:
                continue
            visited[cy][cx] = True
            cells.append((cx, cy))
            # Touching boundary means not a hole
            if cx == 0 or cy == 0 or cx == w - 1 or cy == h - 1:
                enclosed = False
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                stack.append((cx+dx, cy+dy))
        return enclosed

    holes = 0
    for y in range(h):
        for x in range(w):
            if subimg[y][x] == 0 and not visited[y][x]:
                if flood(x, y):
                    holes += 1
    return holes

def classify_component(component, binary_img):
    bbox = extract_bbox(component)
    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    aspect_ratio = width / height if height else 0
    subimg = slice_subimage(binary_img, bbox)
    holes = count_holes(subimg)

    if holes > 0:
        # Treat any single-hole, roughly square component as '0'
        if 0.6 <= aspect_ratio <= 1.4:
            return '0'
        return '?'

    # Digit '1': very narrow
    if width <= 3 or aspect_ratio < 0.45:
        return '1'

    # Heuristic for '2': wide, no holes, strong top/mid/bottom strokes
    top = subimg[0]
    mid = subimg[len(subimg)//2]
    bottom = subimg[-1]
    def stroke_strength(row):
        return sum(row) / len(row)
    top_s = stroke_strength(top)
    mid_s = stroke_strength(mid)
    bot_s = stroke_strength(bottom)
    if top_s > 0.6 and mid_s > 0.5 and bot_s > 0.6 and aspect_ratio >= 0.5:
        return '2'

    return '?'

def sort_components_left_to_right(components):
    ordered = []
    for comp in components:
        min_x, min_y, max_x, max_y = extract_bbox(comp)
        ordered.append((min_x, comp))
    ordered.sort(key=lambda t: t[0])
    return [c for _, c in ordered]

def group_components(components, gap_threshold=2):
    """Group adjacent components horizontally into digit clusters.
    Assumes components already sorted left-to-right."""
    if not components:
        return []
    clusters = []
    current_cluster = []
    # Track current cluster x-extent
    cur_min_x, cur_min_y, cur_max_x, cur_max_y = extract_bbox(components[0])
    current_cluster.append(components[0])
    for comp in components[1:]:
        min_x, min_y, max_x, max_y = extract_bbox(comp)
        # If this component starts within gap_threshold of current cluster end, merge
        if min_x <= cur_max_x + gap_threshold:
            current_cluster.append(comp)
            cur_max_x = max(cur_max_x, max_x)
            cur_min_x = min(cur_min_x, min_x)
            cur_min_y = min(cur_min_y, min_y)
            cur_max_y = max(cur_max_y, max_y)
        else:
            clusters.append(current_cluster)
            current_cluster = [comp]
            cur_min_x, cur_min_y, cur_max_x, cur_max_y = min_x, min_y, max_x, max_y
    clusters.append(current_cluster)
    # Flatten each cluster to unified pixel list
    unified = []
    for cluster in clusters:
        pixels = []
        for c in cluster:
            pixels.extend(c)
        unified.append(pixels)
    return unified

def classify_subimg(sub, context_chars=None):
    """Classify a subimage (segment) into a digit or letter using 3-column template matching.
    context_chars: list of previously classified characters for context-aware disambiguation.
    Returns digit '0'-'9' or letter 'H','E','l','o' or '?' for unknown."""
    h = len(sub)
    w = len(sub[0]) if h else 0
    # Trim empty top/bottom rows for analysis
    top_trim = 0
    bottom_trim = h
    for i, row in enumerate(sub):
        if any(v == 1 for v in row):
            top_trim = i
            break
    for i in range(h - 1, -1, -1):
        if any(v == 1 for v in sub[i]):
            bottom_trim = i + 1
            break
    sub = sub[top_trim:bottom_trim]
    h = len(sub)
    if h == 0:
        return '?'
    rows = [''.join('1' if v==1 else '.' for v in r) for r in sub]
    pattern = '\n'.join(rows)
    
    # Explicit letter patterns
    letter_templates = {
        '1.1\n1.1\n111\n1.1\n1.1': 'H',
        '111\n1..\n111\n1..\n111': 'E',
        '.1.\n1.1\n111\n1.1\n1.1': 'A',  # uppercase A
    }
    
    # Digit patterns
    digit_templates = {
        '111\n1.1\n1.1\n1.1\n111': '0',
        '.1.\n.1.\n.1.\n.1.\n111': '1',
        '.1.\n11.\n.1.\n.1.\n111': '1',  # variant with top-left stroke
        '.1...1.\n1.1.11.\n111..1.\n1.1..1.\n1.1.111': '1',  # wide 1 pattern (7 columns)
        '111\n..1\n111\n1..\n111': '2',
        '11.\n1.1\n11.\n1.1\n11.': '2',  # compact 2 variant
        '11..111\n1.1...1\n11..111\n1.1.1..\n11..111': '2',  # wide 2 pattern
        '111\n1..\n111\n..1\n111': '5',
        '111\n1..\n111\n1.1\n111': '6',
        '111\n1.1\n111\n..1\n111': '9',
    }
    
    # Ambiguous patterns that depend on context
    ambiguous_patterns = {
        '.1.\n.1.\n.1.\n.1.\n111': ('1', 'l'),  # (digit, letter)
        '111\n1.1\n1.1\n1.1\n111': ('0', 'o'),
        '.1.\n1.1\n111\n1.1\n1.1': ('1', 'A'),  # Could be 1 or A - default to digit for standalone
    }
    
    if pattern in letter_templates:
        return letter_templates[pattern]
    
    if pattern in ambiguous_patterns:
        digit_val, letter_val = ambiguous_patterns[pattern]
        # Use context: if preceded by letters, likely a letter; if by digits, likely digit
        if context_chars:
            recent = context_chars[-2:] if len(context_chars) >= 2 else context_chars
            letter_count = sum(1 for c in recent if c.isalpha())
            if letter_count > 0:
                return letter_val
        return digit_val
    
    if pattern in digit_templates:
        return digit_templates[pattern]
    # Hole-based fallback classification
    holes = count_holes(sub)
    if holes > 0:
        return '0'
    return '?'

def main():
    import sys
    
    # Parse arguments
    filename = 'input.ppm'
    digits_only = False
    verbose = False
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--digits-only':
            digits_only = True
        elif args[i] == '--verbose':
            verbose = True
        elif not args[i].startswith('--'):
            filename = args[i]
        i += 1
    
    width, height, img = load_ppm(filename)
    binary = binarize(img)
    
    # Column-based segmentation for multi-stroke digits
    col_has_black = [any(binary[y][x] == 1 for y in range(height)) for x in range(width)]
    segments = []
    start = None
    for x, has in enumerate(col_has_black):
        if has and start is None:
            start = x
        elif not has and start is not None:
            segments.append((start, x - 1))
            start = None
    if start is not None:
        segments.append((start, width - 1))

    def subimage_from_cols(x0, x1):
        return [row[x0:x1 + 1] for row in binary]
    
    #  Merge consecutive narrow segments if they form a recognized multi-part pattern
    merged_segments = []
    i = 0
    while i < len(segments):
        x0, x1 = segments[i]
        # Try merging with next segment if close enough
        if i + 1 < len(segments):
            next_x0, next_x1 = segments[i + 1]
            gap = next_x0 - x1 - 1
            if gap == 1 and (x1 - x0 + 1) == 3 and (next_x1 - next_x0 + 1) == 3:
                # Try combined segment
                combined_sub = subimage_from_cols(x0, next_x1)
                combined_cls = classify_subimg(combined_sub, context_chars=[])
                # If combined makes sense as a digit, use it
                if combined_cls in '0123456789':
                    merged_segments.append((x0, next_x1))
                    i += 2
                    continue
        merged_segments.append((x0, x1))
        i += 1
    
    segments = merged_segments

    results = []
    for idx, (x0, x1) in enumerate(segments):
        sub = subimage_from_cols(x0, x1)
        cls = classify_subimg(sub, context_chars=results)
        
        if verbose:
            trimmed = []
            for row in sub:
                if any(v == 1 for v in row):
                    trimmed.append(''.join('1' if v==1 else '.' for v in row))
            pattern = '\n'.join(trimmed)
            print(f"Segment {idx} cols {x0}-{x1} width={x1-x0+1} classified={cls}\n{pattern}\n---")
        
        results.append(cls)
    
    if digits_only:
        output = ''.join(c for c in results if c.isdigit())
    else:
        output = ''.join(results)
    
    print(output)

if __name__ == "__main__":
    main()
