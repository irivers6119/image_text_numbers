# Decision Trees vs Machine Learning for Digit Classification

This document explains how our heuristic-based classifier works as a **decision tree** and contrasts it with how **machine learning** (specifically CNNs) would solve the same problem.

---

## Our Approach: Hand-Crafted Decision Tree

### What is a Decision Tree?

A decision tree makes classifications by asking a series of yes/no questions about features, following different branches based on the answers.

Our `classify_digit()` function is essentially a decision tree we designed by hand:

```
                    [Start: Unknown Component]
                              |
                    Is width == 3 and height == 5?
                         /            \
                       Yes             No
                        |               |
              [Check Row Patterns]  [Fallback Heuristics]
                        |               |
         Row1 left only? Row3 right only?
                /            \
              Yes             No
               |               |
           Return '5'    [Check other patterns...]
                               |
                         Row1 both? Row3 right?
                            /         \
                          Yes          No
                           |            |
                      Return '9'   [Continue...]
```

### Full Decision Tree for 3×5 Digits

```
classify_digit(component)
│
├─ width == 3 and height == 5?
│  │
│  ├─ YES → Check row patterns
│  │  │
│  │  ├─ row1_left && !row1_right && !row3_left && row3_right?
│  │  │  └─ Return '5'  ✓
│  │  │
│  │  ├─ row1_left && row1_right && !row3_left && row3_right?
│  │  │  └─ Return '9'  ✓
│  │  │
│  │  ├─ row1_left && !row1_right && row3_left && row3_right?
│  │  │  └─ Return '6'  ✓
│  │  │
│  │  └─ row1_left && row1_right && row3_left && row3_right?
│  │     │
│  │     └─ middle_row_count <= 1?
│  │        ├─ YES → Return '0'  ✓
│  │        └─ NO → Return '2'  ✓
│  │
│  └─ NO → Fallback heuristics
│     │
│     ├─ aspect_ratio < 0.5 or width <= 2?
│     │  └─ Return '1'  ✓
│     │
│     ├─ count_holes() == 1?
│     │  │
│     │  └─ Analyze hole position...
│     │     ├─ top_full && bottom_full?
│     │     │  ├─ middle_full? → Return '8'  ✓
│     │     │  └─ else → Return '0'  ✓
│     │     │
│     │     └─ lower_mid_density > upper_mid_density?
│     │        ├─ YES → Return '6'  ✓
│     │        └─ NO → Return '9'  ✓
│     │
│     ├─ density > 0.75 && left_col > 0.6 && right_col > 0.6?
│     │  └─ Return '8'  ✓
│     │
│     └─ top_bar && mid_bar && bottom_bar?
│        │
│        └─ Check row patterns...
│           ├─ left-only in mid, right in lower? → Return '5'  ✓
│           └─ else → Return '2'  ✓
│
└─ No matches → Return '?'  ✗
```

### Advantages of Hand-Crafted Decision Trees

✅ **Interpretable** — You can trace exactly why each decision was made
✅ **Fast** — O(depth) comparisons, typically < 20 operations
✅ **No training needed** — Works immediately on new images
✅ **Small code footprint** — ~100 lines vs megabytes of model weights
✅ **Debuggable** — Add print statements to see which branch was taken

### Disadvantages

❌ **Brittle** — Fails on fonts/styles not considered during design
❌ **Manual feature engineering** — You must identify discriminative features
❌ **Hard to extend** — Adding new digits requires rethinking entire tree
❌ **No generalization** — Doesn't learn from data
❌ **Expert knowledge required** — Need domain expertise to choose features

---

## Machine Learning Approach: Convolutional Neural Networks (CNNs)

### How CNNs Learn Features Automatically

Instead of hand-crafting rules, CNNs **learn the features and decision boundaries from training data**.

### Architecture for Digit Recognition

```
Input Image (28×28 grayscale)
        ↓
[Convolutional Layer 1]  ← Learns edge detectors
  32 filters, 3×3 kernel
  Output: 26×26×32
        ↓
[ReLU Activation]
        ↓
[Max Pooling 2×2]  ← Reduces size, adds translation invariance
  Output: 13×13×32
        ↓
[Convolutional Layer 2]  ← Learns stroke/curve detectors
  64 filters, 3×3 kernel
  Output: 11×11×64
        ↓
[ReLU Activation]
        ↓
[Max Pooling 2×2]
  Output: 5×5×64
        ↓
[Flatten]  ← Convert to 1D vector
  Output: 1600 features
        ↓
[Fully Connected Layer]  ← Combines features into digit concepts
  128 neurons
        ↓
[Dropout 0.5]  ← Prevents overfitting
        ↓
[Output Layer]  ← 10 neurons (one per digit 0-9)
  Softmax activation
        ↓
Probabilities: [0.01, 0.95, 0.02, ...]
                 ↑
          Predicted digit = 1
```

### What CNNs Learn Automatically

**Layer 1 (Low-level features):**
- Horizontal edges
- Vertical edges
- Diagonal lines
- Curves
- Corners

**Layer 2 (Mid-level features):**
- Loops (for 0, 6, 8, 9)
- Vertical strokes (for 1)
- Horizontal bars (for 2, 5, 7)
- S-curves (for 2, 5)
- Hooks and crosses

**Layer 3+ (High-level features):**
- Complete digit patterns
- Style variations
- Rotational invariance
- Scale invariance

### Training Process

```python
# Pseudocode for CNN training

# 1. Initialize random weights
model = CNN()

# 2. Load training data (e.g., 60,000 MNIST images)
train_images, train_labels = load_mnist()

# 3. Training loop
for epoch in range(10):
    for image, label in training_data:
        # Forward pass: compute prediction
        prediction = model.forward(image)
        
        # Compute loss (how wrong was the prediction?)
        loss = cross_entropy(prediction, label)
        
        # Backward pass: compute gradients
        gradients = model.backward(loss)
        
        # Update weights to reduce loss
        model.update_weights(gradients, learning_rate=0.01)

# 4. Evaluate on test data
accuracy = model.evaluate(test_images, test_labels)
print(f"Accuracy: {accuracy:.2%}")  # Typically 99%+ on MNIST
```

### Comparison: Decision Tree vs CNN

| Aspect | Hand-Crafted Decision Tree | CNN |
|--------|---------------------------|-----|
| **Features** | Manual (aspect ratio, holes, densities) | Automatic (learned from data) |
| **Training** | None (rules coded by human) | Requires labeled training data |
| **Accuracy** | 70-90% (depends on font) | 99%+ (generalizes well) |
| **Interpretability** | High (can trace decision path) | Low ("black box") |
| **Speed (inference)** | Very fast (~1ms) | Fast (~10ms on CPU, ~1ms on GPU) |
| **Model Size** | Tiny (~1KB code) | Large (~5MB weights) |
| **Robustness** | Brittle (breaks on new fonts) | Robust (handles variations) |
| **Development Time** | Days (design + debug rules) | Hours (collect data + train) |
| **Maintenance** | Hard (rewrite rules for changes) | Easy (retrain on new data) |

---

## Example: What Each Approach "Sees"

### Input: Handwritten '5'
```
  ███
  █
  ███
    █
  ███
```

### Decision Tree Analysis:
```
1. width=3, height=5? ✓
2. row[1] = "█  " → left=True, right=False ✓
3. row[3] = "  █" → left=False, right=True ✓
4. Match pattern: '5' ✓
```
**Prediction:** '5' ✅

### CNN Analysis (Visualized):

**Layer 1 activations:**
```
Filter 1 (horizontal edges):     Filter 7 (vertical left):
  ───                              █
  ───                              █
  ───                              █
```

**Layer 2 activations:**
```
Filter 23 (top bar + vertical):   Filter 45 (S-curve):
  ███                               ███
  █                                  ▓▓
  █                                  ▓▓█
                                      █
                                     ███
```

**Fully Connected Layer:**
```
Neuron 47 fires (detects "top bar")
Neuron 82 fires (detects "left vertical stroke")
Neuron 101 fires (detects "bottom bar")
...
```

**Output Layer:**
```
0: 0.002
1: 0.001
2: 0.034
3: 0.003
4: 0.012
5: 0.943  ← Highest confidence
6: 0.003
7: 0.001
8: 0.000
9: 0.001
```
**Prediction:** '5' (94.3% confidence) ✅

---

## Hybrid Approach: Combining Both

In practice, you might combine rule-based and ML approaches:

### Ensemble Strategy

```python
def classify_digit_ensemble(component):
    # Get predictions from both methods
    rule_pred = decision_tree_classify(component)
    cnn_pred, cnn_confidence = cnn_classify(component)
    
    # Use rules for high-confidence cases
    if rule_pred != '?' and matches_template(component, rule_pred):
        return rule_pred  # Fast path: exact template match
    
    # Fall back to CNN for ambiguous cases
    if cnn_confidence > 0.9:
        return cnn_pred
    
    # Low confidence: try both and use voting
    if rule_pred == cnn_pred:
        return rule_pred
    
    return '?'  # Uncertain
```

**Benefits:**
- Fast path for easy cases (templates)
- ML handles edge cases and new fonts
- Interpretable for debugging
- High accuracy overall

---

## When to Use Each Approach

### Use Decision Trees (Rule-Based) When:

✅ **Fixed font/format** — OCR-A, LCD displays, digital meters
✅ **Small dataset** — Not enough training data for ML
✅ **Interpretability required** — Medical/legal applications
✅ **Embedded systems** — Limited memory/compute
✅ **Quick prototype** — Need something working today
✅ **Expert knowledge available** — You understand the domain well

**Example domains:**
- Barcode readers
- License plate recognition (fixed format)
- Digital meter readings
- Calculator displays

### Use CNNs (Machine Learning) When:

✅ **Variable fonts** — Handwriting, multiple typefaces
✅ **Large dataset** — 10,000+ labeled examples available
✅ **Accuracy critical** — Need 99%+ performance
✅ **Handling noise** — Blur, rotation, occlusion
✅ **Continuous improvement** — Can retrain as more data arrives
✅ **Unknown variations** — Can't anticipate all edge cases

**Example domains:**
- Handwritten digit recognition (MNIST, check processing)
- Document OCR (arbitrary fonts)
- Street sign recognition
- Captcha solving

---

## Evolution: From Rules to Learning

### Historical Timeline

**1950s-1980s: Rule-Based Systems**
- Expert systems with hand-coded rules
- Template matching
- Requires domain experts to design features

**1990s: Traditional Machine Learning**
- Support Vector Machines (SVMs)
- Decision tree ensembles (Random Forests)
- Still requires manual feature engineering (HOG, SIFT)

**2012+: Deep Learning Revolution**
- CNNs learn features automatically
- End-to-end learning from raw pixels
- Achieves human-level accuracy

**2020s: Hybrid Systems**
- Combine interpretable rules with neural networks
- Explainable AI (XAI) techniques
- Neural-guided rule extraction

---

## Practical Example: Converting Rules to ML

### Step 1: Start with Rules (Our Current Approach)

```python
def classify_digit(component):
    if aspect_ratio < 0.5:
        return '1'
    if has_hole() and top_full and bottom_full:
        return '0'
    # ... 20 more rules
```

**Works well for:** Fixed font, 90% accuracy

### Step 2: Collect Training Data

```python
# Create labeled dataset from your rules
training_data = []
for image_file in glob("digits/*.ppm"):
    component = extract_component(image_file)
    label = rule_based_classify(component)  # Your rules as labels
    training_data.append((component, label))

# Add edge cases with manual labels
manual_labels = load_human_verified_labels("hard_cases.json")
training_data.extend(manual_labels)
```

### Step 3: Train Simple Neural Network

```python
import tensorflow as tf

# Simple fully-connected network (no CNN yet)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(5, 3)),  # 3×5 pixels
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 digits
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
```

**Result:** 95% accuracy, handles more fonts

### Step 4: Upgrade to CNN for Better Performance

```python
# Add convolutional layers
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(5,3,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

**Result:** 99% accuracy, very robust

---

## Interview Discussion Points

### "Why not use ML for this problem?"

**Good answer:**
"For a technical interview focused on algorithms, the rule-based approach demonstrates:
- Understanding of image processing fundamentals
- Ability to design discriminative features
- Clean, debuggable code without dependencies
- Reasoning about complexity and edge cases

In production, I'd evaluate ML (likely a lightweight CNN) if we needed to handle multiple fonts or achieve higher accuracy. The decision tree approach works well for controlled environments like digital displays or fixed-format documents."

### "How would you transition this to ML?"

**Good answer:**
"Three-phase approach:
1. **Data collection:** Use the rule-based classifier to auto-label a large dataset, then manually verify ambiguous cases
2. **Baseline ML model:** Train a simple fully-connected network on extracted features (aspect ratio, hole count, etc.) — similar features but learned weights
3. **End-to-end CNN:** If accuracy isn't sufficient, train a CNN on raw pixels to learn better features automatically

This progressive approach minimizes risk and allows comparison at each stage."

---

## Further Reading

### Classical Computer Vision (Rule-Based)
- Gonzalez & Woods, *Digital Image Processing* (2018)
- Szeliski, *Computer Vision: Algorithms and Applications* (2022)

### Machine Learning for Vision
- Goodfellow et al., *Deep Learning* (2016) — [Free online](https://www.deeplearningbook.org/)
- Stanford CS231n: Convolutional Neural Networks — [Course notes](http://cs231n.github.io/)

### Practical Implementations
- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- TensorFlow Tutorial: [Handwritten Digit Recognition](https://www.tensorflow.org/tutorials/quickstart/beginner)
- PyTorch Tutorial: [Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

### Hybrid Approaches
- Ribeiro et al., *"Why Should I Trust You?" Explaining Predictions of ML Models* (2016)
- Chen et al., *Neural-Backed Decision Trees* (2020)

---

## Code Example: Training on Your Data

```python
# train_cnn.py - Convert your digit extractor to ML

import tensorflow as tf
import numpy as np
from simple_extractor import load_ppm, binarize, find_components

def load_training_data(folder):
    """Load all PPM files and their labels."""
    X, y = [], []
    for filename in glob(f"{folder}/*.ppm"):
        # Extract digit from filename: "digit_5_sample1.ppm" → 5
        label = int(filename.split('_')[1])
        
        # Load and preprocess
        width, height, img = load_ppm(filename)
        binary = binarize(img)
        components = find_components(binary, width, height)
        
        # Assume one digit per image
        if len(components) == 1:
            # Extract 28×28 patch around component
            patch = extract_patch(components[0], target_size=28)
            X.append(patch)
            y.append(label)
    
    return np.array(X), np.array(y)

# Load data
X_train, y_train = load_training_data("training_digits")
X_test, y_test = load_training_data("test_digits")

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                   epochs=10, 
                   validation_data=(X_test, y_test))

# Save
model.save('digit_classifier.h5')

print(f"Test accuracy: {model.evaluate(X_test, y_test)[1]:.2%}")
```

---

## Summary

| Approach | Our Implementation | Machine Learning (CNN) |
|----------|-------------------|------------------------|
| **Method** | Hand-crafted decision tree | Learned feature hierarchy |
| **Features** | aspect_ratio, holes, row patterns | Edges → strokes → digit shapes |
| **Design Time** | Days | Hours (mostly data collection) |
| **Training Time** | 0 (no training) | Minutes to hours |
| **Inference Speed** | ~1ms | ~1-10ms |
| **Accuracy** | 70-95% (font-dependent) | 99%+ (generalizes) |
| **Code Complexity** | Medium (~200 lines) | Low (framework handles details) |
| **Interpretability** | High (trace decision path) | Low (black box) |
| **Best Use Case** | Fixed fonts, embedded systems | Variable fonts, high accuracy needs |

**Key Insight:** Decision trees make the feature engineering explicit and debuggable. CNNs learn the same kinds of features (edges, strokes, patterns) but discover them automatically from data. Both approaches ultimately ask similar questions — CNNs just learned which questions to ask!

---

## See Also

- [README.md](README.md) - Main documentation
- [FLOOD_FILL.md](FLOOD_FILL.md) - Connected component analysis
- [OTSU_METHOD.md](OTSU_METHOD.md) - Adaptive thresholding
- [simple_extractor.py](simple_extractor.py) - Decision tree implementation
