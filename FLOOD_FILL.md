# Flood-Fill Algorithm: Finding Connected Components

**Flood-fill** is a classic algorithm for identifying connected regions in a grid or image. You've seen it as the "bucket fill" tool in paint programs!

In our digit extractor, we use flood-fill to identify each separate character (connected component) in the binary image.

---

## The Problem

Given a binary image (1 = black/text, 0 = white/background), find all connected groups of black pixels.

**Example:**
```
Input (5×5 binary image):
  0 1 1 0 0
  0 1 0 0 1
  0 0 0 1 1
  1 1 0 1 0
  1 0 0 0 0

Output: 4 components
  Component 1: pixels (1,0), (1,1), (2,0)         [top-left group]
  Component 2: pixels (4,1), (3,2), (4,2), (3,3)  [right side]
  Component 3: pixels (0,3), (1,3), (0,4)         [bottom-left]
```

Each component represents a separate character or digit.

---

## What is DFS (Depth-First Search)?

**DFS** is a graph traversal algorithm that explores as far as possible along each branch before backtracking.

### Recursive DFS (Conceptual)

```python
def dfs_recursive(x, y):
    if out_of_bounds(x, y) or visited[y][x] or grid[y][x] == 0:
        return
    
    visited[y][x] = True
    component.append((x, y))
    
    # Recursively visit 4 neighbors
    dfs_recursive(x+1, y)  # right
    dfs_recursive(x-1, y)  # left
    dfs_recursive(x, y+1)  # down
    dfs_recursive(x, y-1)  # up
```

**Problem:** For large components (e.g., 100×100 digit), this can cause **stack overflow** with 10,000+ recursive calls!

### Iterative DFS (Our Solution)

Replace the call stack with an explicit stack data structure:

```python
def dfs_iterative(start_x, start_y):
    stack = [(start_x, start_y)]
    
    while stack:
        x, y = stack.pop()
        
        if out_of_bounds(x, y) or visited[y][x] or grid[y][x] == 0:
            continue
        
        visited[y][x] = True
        component.append((x, y))
        
        # Push 4 neighbors onto stack
        stack.append((x+1, y))
        stack.append((x-1, y))
        stack.append((x, y+1))
        stack.append((x, y-1))
```

**Benefits:**
- ✅ No recursion → no stack overflow
- ✅ Same traversal order as recursive DFS
- ✅ Easy to understand and debug
- ✅ O(1) space per pixel (just coordinates in stack)

---

## Complete Flood-Fill Implementation

Here's the full algorithm from `simple_extractor.py`:

```python
def find_components(binary_img, width, height):
    """Find all connected components using iterative DFS flood-fill."""
    visited = [[False] * width for _ in range(height)]
    components = []
    
    def flood_fill(start_x, start_y):
        """Flood-fill from starting pixel, return list of connected pixels."""
        component = []
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            # Boundary checks
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Skip if already visited or white pixel
            if visited[y][x] or binary_img[y][x] == 0:
                continue
            
            # Mark as visited and add to component
            visited[y][x] = True
            component.append((x, y))
            
            # Add 4-neighbors to stack (N, S, E, W)
            stack.extend([
                (x+1, y),   # East
                (x-1, y),   # West
                (x, y+1),   # South
                (x, y-1)    # North
            ])
        
        return component
    
    # Scan entire image for unvisited black pixels
    for y in range(height):
        for x in range(width):
            if binary_img[y][x] == 1 and not visited[y][x]:
                comp = flood_fill(x, y)
                if comp:  # Only keep non-empty components
                    components.append(comp)
    
    return components
```

### Step-by-Step Execution

**Example: 3×3 image with one component**
```
Initial:
  1 1 0
  1 1 0
  0 0 0

visited = all False
```

**Step 1:** Scan finds (0,0) — unvisited black pixel
- Start flood_fill(0, 0)
- Stack: `[(0,0)]`

**Step 2:** Pop (0,0)
- Mark visited[0][0] = True
- Add to component: `[(0,0)]`
- Push neighbors: Stack = `[(1,0), (-1,0), (0,1), (0,-1)]`

**Step 3:** Pop (0,-1) — out of bounds, skip

**Step 4:** Pop (0,1)
- Mark visited[0][1] = True
- Add to component: `[(0,0), (0,1)]`
- Push neighbors: Stack = `[(1,0), (-1,0), (1,1), (-1,1), (0,2), (0,0)]`

**Step 5:** Pop (0,0) — already visited, skip

**Step 6:** Pop (0,2) — white pixel, skip

... continues until stack is empty

**Result:** component = `[(0,0), (0,1), (1,0), (1,1)]`

---

## 4-Connectivity vs 8-Connectivity

Our implementation uses **4-connectivity** (N, S, E, W neighbors only).

### 4-Connectivity
```
    N
  W ■ E
    S
```
**Neighbors:** `(x±1, y)` and `(x, y±1)` — 4 neighbors

### 8-Connectivity
```
  NW N NE
  W  ■  E
  SW S SE
```
**Neighbors:** `(x±1, y±1)` too — 8 neighbors

### Example Where They Differ

```
Image:
  1 0 1
  0 1 0
  1 0 1

4-connectivity: 5 components (each '1' is separate)
8-connectivity: 1 component (diagonal connections count)
```

**For text/digits:** 4-connectivity is usually better because diagonal connections are rare and can cause false merging of nearby characters.

**To switch to 8-connectivity:**
```python
# Add diagonal neighbors
stack.extend([
    (x+1, y+1),   # SE
    (x+1, y-1),   # NE
    (x-1, y+1),   # SW
    (x-1, y-1)    # NW
])
```

---

## Complexity Analysis

### Time Complexity: O(W × H)
- **Outer loop:** Scans every pixel once → O(W × H)
- **Flood-fill:** Each pixel visited at most once (due to `visited` array)
  - Each pixel pushed to stack: 4 times (from 4 neighbors) worst case
  - But processed only once (continue if visited)
  - Total stack operations: O(W × H)
- **Total:** O(W × H)

### Space Complexity: O(W × H)
- `visited` array: W × H booleans
- Stack: Worst case all pixels in one component → O(W × H)
- Component storage: All pixels → O(W × H)
- **Total:** O(W × H)

### Practical Performance
For a 1920×1080 image:
- Pixels: ~2 million
- Time: < 100ms on modern CPU
- Memory: ~2 MB for visited array

---

## Alternative Approaches

### 1. Breadth-First Search (BFS)
```python
from collections import deque

def flood_fill_bfs(start_x, start_y):
    queue = deque([(start_x, start_y)])
    component = []
    
    while queue:
        x, y = queue.popleft()  # FIFO instead of LIFO
        # ... same logic as DFS
        queue.append((x+1, y))  # BFS explores layer by layer
```

**BFS vs DFS:**
- Same time/space complexity
- BFS finds pixels in "rings" around start (level-order)
- DFS explores deep paths first
- For connected components, **order doesn't matter** → DFS is simpler

### 2. Union-Find (Disjoint Set Union)
```python
# Two-pass algorithm
# Pass 1: Assign provisional labels, merge equivalences
# Pass 2: Relabel with canonical labels
```
**Pros:** More cache-friendly, can be parallelized
**Cons:** More complex implementation

### 3. Scan-Line Algorithm
```python
# Process image row by row
# Merge labels when pixels connect vertically
```
**Pros:** Single forward pass, very efficient
**Cons:** Harder to understand, requires label equivalence tracking

**For interviews:** Stick with DFS flood-fill — it's intuitive and correct!

---

## Common Pitfalls & Debugging

### 1. Infinite Loop (Forgetting `visited`)
```python
# BAD: No visited check
while stack:
    x, y = stack.pop()
    stack.extend([(x+1, y), (x-1, y), ...])  # x,y pushed back infinitely!
```
**Fix:** Always mark visited before pushing neighbors.

### 2. Stack Overflow (Recursive DFS)
```python
# BAD: Recursive on large images
def dfs(x, y):
    visited[y][x] = True
    dfs(x+1, y)  # Can recurse 10,000+ times!
```
**Fix:** Use iterative DFS with explicit stack.

### 3. Wrong Coordinate Order
```python
# BAD: Confusing x/y
if visited[x][y]:  # Should be visited[y][x]!
```
**Remember:** Arrays are row-major → `array[row][col]` = `array[y][x]`

### 4. Boundary Check After Mutation
```python
# BAD: Check bounds after accessing
x, y = stack.pop()
visited[y][x] = True  # Crash if out of bounds!
if x < 0 or x >= width:
    continue
```
**Fix:** Check bounds FIRST, then access arrays.

---

## Visualizing Flood-Fill

### Animation (Conceptual)

```
Step 0: Find start pixel
  ■ ■ □ □
  ■ ■ □ ■
  □ □ □ ■

Step 1: Mark (0,0), push neighbors
  ✓ ? □ □
  ? ? □ ■
  □ □ □ ■

Step 2: Process (1,0), push neighbors
  ✓ ✓ □ □
  ? ? □ ■
  □ □ □ ■

Step 3-5: Continue until all connected pixels visited
  ✓ ✓ □ □
  ✓ ✓ □ ?
  □ □ □ ?

Final: Component = {(0,0), (1,0), (0,1), (1,1)}
  ✓ ✓ □ □
  ✓ ✓ □ □
  □ □ □ □

Separate component at (3,1):
  □ □ □ □
  □ □ □ ✓
  □ □ □ ✓
```

---

## Real-World Applications

Flood-fill is used in:

1. **Paint bucket tool** (Photoshop, GIMP)
2. **Minesweeper** (clearing empty cells)
3. **Game pathfinding** (finding reachable areas)
4. **Computer vision:**
   - Object segmentation
   - Region growing
   - Blob detection
5. **Circuit board analysis** (finding connected traces)
6. **Geographic information systems** (watershed analysis)

---

## Interview Discussion Points

### Why Iterative Over Recursive?

**You:** "I chose iterative DFS to avoid stack overflow. For a 100×100 digit, recursive DFS would make 10,000 nested calls, exceeding Python's default recursion limit of 1,000. The iterative version uses an explicit stack — same algorithm, but we control the memory allocation."

### Optimization Ideas

**Interviewer:** "Can you optimize this further?"

**You:** "Several options:
1. **Early termination:** If we only need component count, stop after marking visited
2. **Bounding box tracking:** Update min/max x/y during flood-fill (one pass instead of two)
3. **Parallel processing:** Process multiple unconnected regions simultaneously
4. **Cache optimization:** Use row-major traversal to improve spatial locality

For this problem size (dozens of pixels per digit), the simple approach is fast enough."

---

## Code Variations

### With Bounding Box Calculation

```python
def flood_fill_with_bbox(start_x, start_y):
    """Return component and its bounding box."""
    component = []
    stack = [(start_x, start_y)]
    min_x = max_x = start_x
    min_y = max_y = start_y
    
    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if visited[y][x] or binary_img[y][x] == 0:
            continue
        
        visited[y][x] = True
        component.append((x, y))
        
        # Update bounding box
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        
        stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
    
    return component, (min_x, min_y, max_x, max_y)
```

### With Size Filtering

```python
def find_large_components(binary_img, min_size=10):
    """Only return components with at least min_size pixels."""
    # ... same flood_fill code ...
    
    for y in range(height):
        for x in range(width):
            if binary_img[y][x] == 1 and not visited[y][x]:
                comp = flood_fill(x, y)
                if len(comp) >= min_size:  # Filter small noise
                    components.append(comp)
```

---

## Summary

| Aspect | Details |
|--------|---------|
| **Purpose** | Find all connected components in binary image |
| **Algorithm** | Iterative DFS (depth-first search) |
| **Data Structure** | Explicit stack (list in Python) |
| **Time Complexity** | O(W × H) — each pixel visited once |
| **Space Complexity** | O(W × H) — visited array + stack |
| **Connectivity** | 4-neighbors (can extend to 8) |
| **Advantages** | Simple, robust, no stack overflow |

**Key Insight:** Flood-fill is just graph traversal (DFS/BFS) applied to a 2D grid where pixels are nodes and adjacency defines edges.

---

## References

- [Flood Fill Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Flood_fill)
- Sedgewick, R. & Wayne, K. (2011). *Algorithms, 4th Edition*. Chapter 4: Graphs
- Used in: OpenCV (`cv2.connectedComponents`), scikit-image (`label`), PIL (`ImageDraw.floodfill`)

## See Also

- [README.md](README.md) - Main digit extractor documentation
- [OTSU_METHOD.md](OTSU_METHOD.md) - Adaptive thresholding algorithm
- [simple_extractor.py](simple_extractor.py) - Implementation with flood-fill
