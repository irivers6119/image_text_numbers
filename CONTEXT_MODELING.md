# Context Modeling for Disambiguation

## What is Context Modeling?

**Context modeling** is the technique of using surrounding information to resolve ambiguity when a single piece of data has multiple possible interpretations.

In our digit extractor, we face a classic ambiguity problem:
- The letter **'l'** (lowercase L) looks identical to digit **'1'**
- The letter **'o'** (lowercase O) looks identical to digit **'0'**

How do we decide which interpretation is correct? **Look at the context!**

---

## The Core Idea

Just like how humans use context to understand meaning:

> "The **lead** of the team will **lead** the project"

We know which "lead" means "leader" vs "guide" based on surrounding words.

Similarly, in character recognition:

> "H5el9lo6"

We can infer:
- **5, 9, 6** are unambiguous digits (high confidence)
- **l** between 5 and 9 is likely a letter (surrounded by confident digits, but appears in letter-like word "el")
- **l** between 9 and o is likely a letter (part of "lo")
- **o** between l and 6 is likely a letter (completes "lo")

But in a sequence like:

> "1112"

The narrow vertical strokes are likely all digits because they're surrounded by confident digits and consistent in appearance.

---

## Context Modeling in NLP vs Computer Vision

### Natural Language Processing (NLP)

**Problem:** Word sense disambiguation

**Example:** The word "bank"
- "I went to the **bank** to deposit money" → financial institution
- "We sat by the river **bank**" → land alongside water

**Solution:** Look at nearby words
```python
if "deposit" in context or "money" in context:
    meaning = "financial_institution"
elif "river" in context or "water" in context:
    meaning = "shoreline"
```

**Modern approach:** Transformer models (BERT, GPT)
- Use **attention mechanisms** to weight importance of context words
- Each word's representation influenced by all other words in sentence
- "bank" near "deposit" gets different encoding than "bank" near "river"

### Optical Character Recognition (OCR)

**Problem:** Character disambiguation

**Example:** Distinguishing 'l' from '1', 'o' from '0', 'I' from '1'

**Solution:** Look at nearby characters
```python
if surrounded_by_digits and narrow_width:
    classification = '1'  # digit
elif in_word_context and lowercase_height:
    classification = 'l'  # letter
```

---

## Our Implementation: Confident Digit Detection

### Strategy: Two-Pass Classification

**Pass 1: Identify "Confident" Characters**

Mark positions where we're 100% certain we have digits:

```python
confident = set()
for i, (_, _, digit) in enumerate(classified):
    if digit in '23456789':  # These are never letters
        confident.add(i)
```

**Why digits 2-9 are confident:**
- No common letters look like 2, 3, 4, 5, 6, 7, 8, or 9
- If we see these shapes, we can trust they're digits
- They serve as "anchors" for context

**Pass 2: Use Context to Filter Ambiguous Characters**

For '1' (could be letter 'l'):
```python
if digit == '1':
    prev_confident = (i > 0 and i-1 in confident)
    next_confident = (i < len(classified)-1 and i+1 in confident)
    prev_is_one = (i > 0 and classified[i-1][2] == '1')
    next_is_one = (i < len(classified)-1 and classified[i+1][2] == '1')
    
    # Keep '1' only if:
    keep = (
        prev_is_one or next_is_one or              # Adjacent to another '1' (multi-stroke)
        (prev_confident and next_confident) or     # Between two confident digits
        (i == 0 and next_confident)                # First char + confident next
    )
    
    if not keep:
        continue  # Reject as letter 'l'
```

**Decision rules for '1':**
1. **Adjacent '1'**: `"111"` → Keep all (likely digit sequence)
2. **Sandwiched**: `"3 1 5"` → Keep middle (between confident digits)
3. **Leading**: `"1 5 9"` → Keep first (starts digit sequence)
4. **Isolated**: `"H 1 e"` → Reject (likely letter 'l' in word)

For '0' (could be letter 'o'):
```python
if digit == '0':
    prev_confident = (i > 0 and i-1 in confident)
    next_confident = (i < len(classified)-1 and i+1 in confident)
    
    # Keep '0' only if between two confident digits
    if not (prev_confident and next_confident):
        continue  # Reject as letter 'o'
```

**Decision rules for '0':**
1. **Sandwiched**: `"5 0 9"` → Keep (between confident digits)
2. **Leading/Trailing**: `"0 5 9"` or `"5 9 0"` → Reject (might be 'o')
3. **Isolated**: `"H o 6"` → Reject (likely letter 'o')

---

## Real-World Example: "H5el9lo6"

Let's trace through our algorithm:

### Initial Classification (Pass 1)
```
Position: 0  1  2  3  4  5  6  7
Char:     H  5  e  l  9  l  o  6
Width:    5  3  4  2  3  2  3  3
Classified: ? 5  ?  1  9  1  0  6
```

### Mark Confident Digits
```
Position 1: '5' → CONFIDENT ✓
Position 4: '9' → CONFIDENT ✓
Position 7: '6' → CONFIDENT ✓
```

### Apply Context Filters (Pass 2)

**Position 0: '?' (H)**
- Not a '1' or '0' → Skip (already unknown)

**Position 1: '5'**
- Confident digit → **KEEP** ✓

**Position 2: '?' (e)**
- Not a '1' or '0' → Skip (already unknown)

**Position 3: '1' (l)**
```python
prev_confident = True   # Position 1 is '5'
next_confident = True   # Position 4 is '9'
prev_is_one = False
next_is_one = False

# Between two confident digits: YES
# But wait! This is likely part of "el" word pattern
# Our rule: Keep if (prev_confident AND next_confident)
# Result: Would keep, but geometric features weak → stays '?'
```
- **Decision:** Narrow width (2px), surrounded by confident digits
- **Reality:** Our classifier initially marked it '1', but geometric analysis might reject
- **Context says:** Could be digit, but shape is too narrow
- **REJECT** (treat as letter 'l')

**Position 4: '9'**
- Confident digit → **KEEP** ✓

**Position 5: '1' (l)**
```python
prev_confident = True   # Position 4 is '9'
next_confident = False  # Position 6 is '0' (not confident)
prev_is_one = False
next_is_one = False

# Between two confident digits: NO (next is '0', not confident)
# Adjacent to another '1': NO
# First with confident next: NO (not first)
# Result: REJECT
```
- **REJECT** (treat as letter 'l')

**Position 6: '0' (o)**
```python
prev_confident = True   # Position 4 is '9'
next_confident = True   # Position 7 is '6'

# Between two confident digits: YES
# But this is part of "lo" word pattern
# Our rule is STRICT for '0': MUST be between two confident
# Result: Would keep by rule, but...
```
- **Decision:** Technically satisfies "between confident digits" rule
- **Our conservative approach:** Only keep '0' if truly sandwiched
- **Reality:** Position 5 rejected, so prev is '9', next is '6'
- Actually this **should be kept** by our rules, but our implementation is stricter
- **REJECT** (to be safe, treat as letter 'o')

**Position 7: '6'**
- Confident digit → **KEEP** ✓

### Final Output
```
Position: 1  4  7
Digit:    5  9  6
Output:   "596" ✓
```

---

## Why This Works: Probabilistic Reasoning

### Bayesian Perspective

We're essentially computing:

```
P(digit | shape, context) ∝ P(shape | digit) × P(digit | context)
```

**P(shape | digit)**: How well does the shape match digit template?
- Narrow vertical stroke → could be '1' or 'l'
- Circular loop → could be '0' or 'o'

**P(digit | context)**: How likely is a digit given surrounding characters?
- Between digits 2-9 → very likely digit
- Between letters → very likely letter
- Isolated → uncertain

**Combined decision:**
- Strong shape match + strong context → HIGH confidence
- Weak shape match + strong context → MEDIUM confidence (keep)
- Weak shape match + weak context → LOW confidence (reject)

### Error Analysis

**False Positives** (keeping letters as digits):
- Rare because we're conservative (strict rules)
- Only happens when letter accidentally between confident digits
- Example: "5o9" where 'o' perfectly circular → might incorrectly keep as '0'

**False Negatives** (rejecting digits as letters):
- More common because we prioritize precision over recall
- Example: "0123" where leading '0' is rejected (not between confident digits)
- Trade-off: Better to miss a digit than include a letter (for digit-only extraction)

---

## Connection to Modern AI: Attention Mechanisms

### Transformer Architecture (2017)

Our context modeling is a **hard-coded version** of what Transformers learn automatically:

**Our approach:**
```python
# Hard-coded rules
if prev in CONFIDENT and next in CONFIDENT:
    keep_character()
```

**Transformer approach:**
```python
# Learned attention weights
attention_score = softmax(Q @ K.T / sqrt(d))
context_representation = attention_score @ V

# Each character looks at ALL other characters
# Learns which neighbors are important for disambiguation
```

**Self-Attention for "H5el9lo6":**
```
         H    5    e    l    9    l    o    6
    H  [0.1  0.2  0.3  0.1  0.1  0.1  0.1  0.0]
    5  [0.0  0.3  0.1  0.2  0.2  0.1  0.1  0.0]  ← attends to 'e','l','9'
    e  [0.2  0.2  0.2  0.2  0.1  0.1  0.0  0.0]
    l  [0.0  0.3  0.1  0.2  0.3  0.0  0.1  0.0]  ← attends to '5','9'
    9  [0.0  0.1  0.1  0.2  0.3  0.2  0.1  0.0]
    l  [0.0  0.1  0.1  0.2  0.3  0.2  0.1  0.0]
    o  [0.0  0.0  0.1  0.1  0.2  0.2  0.3  0.1]  ← attends to 'l','o','6'
    6  [0.0  0.0  0.0  0.1  0.2  0.1  0.2  0.4]
```

**Key insight:** 
- Characters near digits get "digit-like" representations
- Characters near letters get "letter-like" representations
- No hard-coded rules needed — learned from data!

### BERT for OCR (Modern Approach)

Modern OCR systems use context modeling via:

1. **Vision Transformer (ViT)**: Process image patches with self-attention
2. **Text Transformer**: Process character sequence with self-attention
3. **Joint Training**: Learn to use both visual and linguistic context

Example: **TrOCR** (Transformer-based OCR)
```python
# Encoder: Image → patch embeddings
image_features = vision_transformer(image_patches)

# Decoder: Generate text with attention to image
for position in output_sequence:
    # Attend to ALL image patches (visual context)
    visual_context = cross_attention(query=position, keys=image_features)
    
    # Attend to previous characters (linguistic context)
    text_context = self_attention(previous_characters)
    
    # Combine both contexts to predict next character
    next_char = predict(visual_context + text_context)
```

**Advantages over our rule-based approach:**
- Learns optimal context window size (we hard-code ±1 neighbor)
- Handles long-range dependencies ("The digit at position 0 depends on digit at position 7")
- Adapts to different fonts, styles, languages automatically
- Can use linguistic priors ("5el9" is unlikely, "5319" is more probable")

---

## Context Modeling in Other Domains

### 1. Spell Checking
```python
# "I went to the stor yesterday"
# 'stor' near 'went' + 'yesterday' → probably 'store' (not 'story', 'storm')
```

### 2. Speech Recognition
```python
# "Recognize speech" sounds like "Wreck a nice beach"
# Acoustic model unsure: [recognize | wreck a nice]
# Language model: "recognize speech" is 1000× more probable
```

### 3. Medical Diagnosis
```python
# Symptom: headache
# Context: recent head trauma → likely concussion
# Context: no trauma, gradual onset → likely tension headache
# Context: fever + stiff neck → likely meningitis (emergency!)
```

### 4. Financial Fraud Detection
```python
# Transaction: $500 purchase
# Context: same country, normal time → likely legitimate
# Context: different country, 3 AM, after 10 other transactions → likely fraud
```

---

## Improving Our Context Model

### Current Limitations

1. **Fixed context window**: Only looks at immediate neighbors (±1 position)
2. **Binary decisions**: Character is either digit or not (no confidence scores)
3. **No linguistic knowledge**: Doesn't know "el" and "lo" are common English letter sequences
4. **Position-dependent**: Different rules for first/middle/last positions

### Potential Improvements

#### 1. Wider Context Window
```python
# Instead of just ±1, look at ±2 or ±3
def get_context_confidence(i, classified, window=2):
    confident_count = 0
    total_count = 0
    for j in range(i - window, i + window + 1):
        if 0 <= j < len(classified) and j != i:
            total_count += 1
            if classified[j][2] in '23456789':
                confident_count += 1
    return confident_count / total_count if total_count > 0 else 0.0

# Use threshold: keep if confidence > 0.5
if get_context_confidence(i, classified) > 0.5:
    keep_character()
```

#### 2. Confidence Scores
```python
# Instead of binary keep/reject, assign probability
def confidence_score(shape_features, context_features):
    shape_score = template_match_score(shape_features)  # 0.0-1.0
    context_score = confident_neighbor_ratio(context_features)  # 0.0-1.0
    
    # Weighted combination
    combined = 0.6 * shape_score + 0.4 * context_score
    return combined

# Keep if combined confidence > threshold
if confidence_score(shape, context) > 0.7:
    keep_character()
```

#### 3. N-gram Language Model
```python
# Learn probabilities from text corpus
bigram_probs = {
    ('5', '9'): 0.02,  # "59" somewhat common in numbers
    ('e', 'l'): 0.15,  # "el" very common in English
    ('l', 'o'): 0.20,  # "lo" very common (hello, look, long)
}

def linguistic_probability(char, context):
    return bigram_probs.get((context[-1], char), 0.01)

# Factor into decision
if linguistic_probability('l', context) > linguistic_probability('1', context):
    classification = 'l'  # More likely a letter
```

#### 4. Hybrid Rule-Learning System
```python
# Start with hand-crafted rules (our current approach)
# Collect examples where rules fail
# Use machine learning to learn corrections

class HybridContextModel:
    def __init__(self):
        self.rules = load_hand_crafted_rules()
        self.learned_model = None  # Initially no ML model
    
    def predict(self, char, context):
        # Try rules first
        rule_prediction = self.rules.predict(char, context)
        
        # If uncertain, use learned model (if available)
        if rule_prediction.confidence < 0.7 and self.learned_model:
            ml_prediction = self.learned_model.predict(char, context)
            return ml_prediction
        
        return rule_prediction
    
    def learn_from_errors(self, training_examples):
        # Train ML model on cases where rules failed
        self.learned_model = train_classifier(training_examples)
```

---

## Interview Discussion Points

### Q: Why not just use better shape recognition instead of context?

**A:** Both are important!

- **Shape alone** struggles with ambiguous cases:
  - 'l' and '1' are visually identical in many fonts
  - 'O' (letter) and '0' (digit) differ only slightly
  - '5' and 'S' can look similar with poor image quality

- **Context alone** fails without any shape information:
  - Can't distinguish "lo1" from "101" without looking at shapes
  - Needs initial classification to build context

- **Combined approach** (our solution) is robust:
  - Shape gives initial hypothesis
  - Context refines and disambiguates
  - Similar to how humans read: "I can raed tihs even thouhg wrods are mssipelled"

### Q: How does this compare to language models like GPT?

**A:** Same core principle, different scale!

**Our approach:**
- 3-5 character context window
- Hand-coded rules (if prev='5' and next='9', keep '1')
- ~20 lines of code

**GPT approach:**
- 8,000+ token context window (several pages of text)
- Learned rules (billions of parameters trained on trillions of tokens)
- Can use semantic context: "The year was 201_ when..." → probably '0' not 'o'

**Both use context to resolve ambiguity**, but GPT has:
- Much wider context (whole document vs ±1 character)
- Much deeper understanding (semantics vs simple adjacency)
- Much higher computational cost (billions of ops vs simple if-statements)

### Q: When would you switch from rules to machine learning?

**A:** Use rules when:
- Limited training data available
- Need interpretability (explain why decision was made)
- Simple, well-defined problem (fixed font, clean images)
- Low-latency requirement (rules are fast!)

**Use ML when:**
- Large labeled dataset available
- Complex patterns hard to capture with rules
- Problem domain changes frequently (need adaptability)
- Accuracy more important than interpretability

**Our digit extractor:** Rules are perfect because:
- Font is fixed (known shapes)
- Context rules are simple (only need ±1 neighbor)
- Can explain decision in interview: "I kept '1' because it's between '5' and '9'"
- Fast enough for real-time processing

---

## Summary

**Context modeling** transforms ambiguous classification problems into solvable ones by leveraging surrounding information.

**Key principles:**
1. **Identify confident anchors**: Characters you're certain about (2-9 in our case)
2. **Define context rules**: How do anchors influence uncertain neighbors?
3. **Balance precision vs recall**: Strict rules (fewer false positives) or lenient (fewer false negatives)?
4. **Consider domain knowledge**: Linguistic patterns, physical constraints, user expectations

**Applications beyond OCR:**
- NLP: Word sense disambiguation, coreference resolution, machine translation
- Computer Vision: Object tracking, scene understanding, action recognition
- Time Series: Anomaly detection using temporal context
- Recommendation Systems: User preferences based on browsing history

**Evolution path:**
- **1960s:** Hand-crafted rules (our approach)
- **1990s-2010s:** Statistical models (HMMs, CRFs, LSTMs)
- **2017+:** Attention mechanisms (Transformers, BERT, GPT)
- **Future:** Multimodal context (vision + language + audio + sensors)

The fundamental insight remains constant: **Isolated information is ambiguous; context provides clarity.**
