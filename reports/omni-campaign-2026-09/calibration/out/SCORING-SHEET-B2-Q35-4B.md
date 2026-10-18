# SCORING SHEET — Q35-4B-B2 (parent)

**THE VERDICT AND EVERY COUNT BELOW ARE ANNOTATIONS. THE DECISION IS CYRIL'S.** Nothing in this file grades. Tick a box per lane; the artifact-tolerant lens applies: loops, echo, cap-rides and a missing EOS never count against intelligence — only a content-miss does.

| field | value |
|---|---|
| model | `/home/cereal/models/Qwen3.5-4B` |
| directory sha256 | `94cb094ac11062b0c44c4527b0be380ad81d249382055ced95e367c5ecf589a2` |
| loaded class | `Qwen3_5ForConditionalGeneration` |
| parameters (live) | 4,539,265,536 |
| transformers | 5.12.1 |
| dtype / device | bfloat16 / sm86 NVIDIA GeForce RTX 3090 |
| max new tokens | 1024 |
| bank sha256 | `f26f36bf31cb2a062c3e75e9f1567b379ea886e6022e2afb93b572a269f4a20b` |
| dossier sha256 | `09fdc0a6b1d6ef21fd7e506e07c01ac7a55200a68c6a0f951a329e40946797f2` |
| lane digest | `cb0fd47f8a842b9741f280853b6aee06c45ceb9d01c595ca60fae62e0b19ad8e` |
| emit-zero floor | True |
| decode-path preflight agree | True |

## Columns

* **Column A — copy-back counts:** 21 / 21  (UPPER BOUND (no reading attached yet))
* **Column B — copy-back refused:** 19 / 21  (UPPER BOUND (no reading attached yet))
* voided as ECHO (column A minus these **is** column B): code-review-sql-injection-B, instr-ordered-steps-B
* what the bound subtracts: only EMPTY, LOOP, CAP_RIDE — the boxes a machine can prove hold nothing. An ECHO lane is still inside column A, because *“this is a copy”* is a threshold call and not a certainty; your reading may hold any of the 21 scored lanes.
* canonical boxes — signal 0, artifact-miss 0, content-miss 0, still to read 21

## Instrument cautions — read these lanes yourself

A copy can satisfy a PRESERVE ask and neither a SELECT nor a PRODUCE one. Each lane below was called ECHO on a non-PRESERVE ask, which means either the model handed the payload back or the 0.60 copy threshold — calibrated on LFM2.5-230M — does not transfer here. **The threshold was not moved.**

* `code-review-sql-injection-B` (PRODUCE) — payload union coverage 0.607656
* `instr-ordered-steps-B` (PRODUCE) — payload union coverage 0.676471

## Lane summary

| lane | ask | form box | term | tokens | rubric | your grade |
|---|---|---|---|---|---|---|
| code-rle-encode-A | PRODUCE | CONTENT | complete | 111 | pass | &nbsp; |
| code-bugfix-pagination-B | PRODUCE | CONTENT | complete | 51 | pass | &nbsp; |
| code-review-sql-injection-B | PRODUCE | ECHO | complete | 365 | pass | &nbsp; |
| code-trace-recursion-A | PRODUCE | CONTENT | complete | 24 | pass | &nbsp; |
| code-api-contract-B | PRODUCE | CONTENT | complete | 65 | pass | &nbsp; |
| reason-arith-two-step-A | PRODUCE | CONTENT | complete | 209 | pass | &nbsp; |
| reason-schedule-constraints-B | PRODUCE | CONTENT | complete | 540 | no match | &nbsp; |
| reason-uncertainty-churn-B | SELECT | CONTENT | complete | 876 | no match | &nbsp; |
| reason-unit-conversion-A | PRODUCE | CONTENT | complete | 486 | no match | &nbsp; |
| reason-causal-confound-B | PRODUCE | CONTENT | complete | 109 | pass | &nbsp; |
| reason-counterexample-B | PRODUCE | CONTENT | complete | 377 | pass | &nbsp; |
| mt-venue-revision-A | PRODUCE | CONTENT | complete | 48 | pass | &nbsp; |
| mt-dietary-correction-B | PRODUCE | CONTENT | complete | 359 | pass | &nbsp; |
| mt-instruction-persistence-B | PRODUCE | CONTENT | complete | 28 | pass | &nbsp; |
| mt-context-carryover-B | PRODUCE | CONTENT | complete | 27 | pass | &nbsp; |
| instr-format-json-strict-A | PRODUCE | CONTENT | complete | 27 | pass | &nbsp; |
| instr-word-budget-A | PRODUCE | CONTENT | complete | 27 | pass | &nbsp; |
| instr-negative-constraint-B | PRODUCE | CONTENT | complete | 190 | pass | &nbsp; |
| instr-ordered-steps-B | PRODUCE | ECHO | censored-at-cap | 1024 | no match (truncated — read it) | &nbsp; |
| know-explain-tradeoff-A | PRODUCE | CONTENT | complete | 932 | no match | &nbsp; |
| know-refuse-fabrication-B | SELECT | CONTENT | complete | 265 | pass | &nbsp; |

## Every case, verbatim

### code-rle-encode-A  (coding, ask PRODUCE)

**Prompt**

*user*:

```
Write a Python function named run_length that takes a string and returns a list of (character, count) tuples for each consecutive run. run_length('aaabbc') must return [('a', 3), ('b', 2), ('c', 1)]. Do not import any module. Give the function only.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): a def named run_length, no imports, correct on the example

**Response** (111 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
```python
def run_length(s):
    if not s:
        return []
    
    result = []
    current_char = s[0]
    count = 1
    
    for i in range(1, len(s)):
        if s[i] == current_char:
            count += 1
        else:
            result.append((current_char, count))
            current_char = s[i]
            count = 1
    
    result.append((current_char, count))
    return result
```
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'def\\s+run_length\\s*\\(' — ok
  * `required_all` 'return' — ok
  * `forbidden_any` '(?m)^[ \\t]*import[ \\t]+[A-Za-z_]' — ok
  * `forbidden_any` '(?m)^[ \\t]*from[ \\t]+[A-Za-z_][\\w.]*[ \\t]+import\\b' — ok
  * `max_words` 400 — ok
  * `ast_function_name` 'run_length' — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### code-bugfix-pagination-B  (coding, ask PRODUCE)

**Prompt**

*user*:

```
This helper is meant to return how many pages are needed to show n items at page_size per page, and it silently drops the last partial page.

def page_count(n, page_size):
    return n // page_size

Give the corrected function. Do not restate the buggy line.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): ceil division: (n + page_size - 1) // page_size, or math.ceil

**Response** (51 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
```python
def page_count(n, page_size):
    if page_size == 0:
        raise ValueError("page_size must be greater than 0")
    return (n + page_size - 1) // page_size
```
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'def\\s+page_count' — ok
  * `required_all` '(page_size\\s*-\\s*1\\)\\s*//\\s*page_size)|(math\\.ceil)|(-\\s*\\(?\\s*-\\s*n\\s*//)' — ok
  * `max_words` 300 — ok
  * `ast_function_name` 'page_count' — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### code-review-sql-injection-B  (coding, ask PRODUCE)

**Prompt**

*user*:

```
Review this database helper and say what is wrong with it and how to fix it.

def find_user(conn, name):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE name = '" + name + "'")
    return cur.fetchone()

```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): SQL injection; fix is a parameterised query with a placeholder

**Response** (365 tokens, complete, form box ECHO — union coverage 0.607656 of the lane payload, at or above 0.6)

```
The primary issue with this code is **SQL Injection**.

### The Problem
You are concatenating the user input (`name`) directly into the SQL query string. If a malicious user provides a name like `' OR '1'='1`, the query becomes:
```sql
SELECT id FROM users WHERE name = '' OR '1'='1'
```
This would bypass the authentication check and return the ID of the first user in the database, allowing attackers to access unauthorized data.

### The Solution
You must use **parameterized queries** (also known as prepared statements). This separates the SQL code from the data, ensuring that any input is treated strictly as a string value, never as executable code.

Here is the corrected version using standard Python `sqlite3` syntax (which is the most common context for this pattern):

```python
def find_user(conn, name):
    cur = conn.cursor()
    # Use a placeholder (?) instead of concatenating the name
    cur.execute("SELECT id FROM users WHERE name = ?", (name,))
    return cur.fetchone()
```

### Key Changes Made:
1.  **Removed String Concatenation**: Replaced `"'" + name + "'"` with a placeholder `?`.
2.  **Added Tuple**: Passed the `name` variable as a tuple `(name,)` to the `execute` method.
3.  **Safety**: The database driver now handles the escaping and quoting automatically, preventing SQL injection attacks.

*Note: If you are using a different database library (like `psycopg2` for PostgreSQL or `mysql-connector`), the syntax for placeholders might vary slightly (e.g., `%s`), but the principle of separating code from data remains the same.*
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'inject' — ok
  * `required_all` '%s|\\?|placeholder|paramet' — ok
  * `max_words` 400 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### code-trace-recursion-A  (coding, ask PRODUCE)

**Prompt**

*user*:

```
def f(n):
    if n <= 1:
        return 1
    return n * f(n - 2)

What does f(7) return? Give the number and one line showing how you got it.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): 7*5*3*1 = 105

**Response** (24 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
f(7) returns 105, calculated as 7 * 5 * 3 * 1.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` '\\b105\\b' — ok
  * `max_words` 200 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### code-api-contract-B  (coding, ask PRODUCE)

**Prompt**

*user*:

```
A service returns {"items": [...], "next": <cursor or null>}. Write a Python function fetch_all(client) that returns every item across all pages by calling client.get(cursor) until next is null. Handle the first call, which takes cursor=None. No imports.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): a loop that follows the cursor until next is None and extends a list

**Response** (65 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
```python
def fetch_all(client):
    items = []
    cursor = None

    while True:
        response = client.get(cursor)
        items.extend(response["items"])
        cursor = response.get("next")
        
        if cursor is None:
            break
            
    return items
```
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'def\\s+fetch_all' — ok
  * `required_all` 'while|for' — ok
  * `required_all` '\\[.{0,12}next.{0,12}\\]|\\.get\\(' — ok
  * `max_words` 400 — ok
  * `ast_function_name` 'fetch_all' — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### reason-arith-two-step-A  (reasoning, ask PRODUCE)

**Prompt**

*user*:

```
Take thirteen groups of nine, then discard seventeen units. Divide whatever remains into piles of eight. Report three numbers: the subtotal before dividing, the count of full piles, and the remainder.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): 13*9 = 117, minus 17 = 100; 100 / 8 = 12 piles remainder 4

**Response** (209 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Here is the step-by-step breakdown of the calculation:

1.  **Calculate the initial total**:
    Multiply 13 groups by 9 units per group.
    $$13 \times 9 = 117$$

2.  **Discard the units**:
    Subtract 17 from the total calculated above.
    $$117 - 17 = 100$$
    *(This is the subtotal before dividing.)*

3.  **Divide into piles**:
    Divide the remaining 100 units into piles of 8.
    $$100 \div 8 = 12 \text{ with a remainder of } 4$$
    *(This gives us 12 full piles and a remainder of 4.)*

**Final Report:**
*   **Subtotal before dividing**: 100
*   **Count of full piles**: 12
*   **Remainder**: 4
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` '\\b100\\b' — ok
  * `required_all` '\\b12\\b' — ok
  * `required_all` '\\b4\\b' — ok
  * `max_words` 250 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### reason-schedule-constraints-B  (reasoning, ask PRODUCE)

**Prompt**

*user*:

```
Three people present in three consecutive slots. Priya cannot go last. Rafi must go immediately after Priya. Elke is not first. Give the order, first to last, and say why it is the only one.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): Priya, Rafi, Elke -- forced, and the answer must say what forces it

**Response** (540 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
To determine the only valid order, let's break down the constraints step by step:

**The Constraints:**
1.  **Slots:** 1, 2, 3 (First, Second, Last).
2.  **People:** Priya (P), Rafi (R), Elke (E).
3.  **Rule A:** Priya cannot go last (P $\neq$ 3).
4.  **Rule B:** Rafi must go immediately after Priya (Sequence: P $\to$ R).
5.  **Rule C:** Elke is not first (E $\neq$ 1).

**Step-by-Step Deduction:**

1.  **Analyze Rule B (P $\to$ R):**
    Since Rafi must immediately follow Priya, they must occupy two consecutive slots. The possible pairs for (Priya, Rafi) are:
    *   Slots 1 and 2: [P, R, ?]
    *   Slots 2 and 3: [?, P, R]

2.  **Test Scenario 1: Priya is First, Rafi is Second.**
    *   Order so far: **P, R, ?**
    *   The only person left is Elke. So the order is **P, R, E**.
    *   *Check Rule A:* Priya is not last. (Pass)
    *   *Check Rule C:* Elke is not first. (Pass, she is last).
    *   *Result:* This is a valid possibility.

3.  **Test Scenario 2: Priya is Second, Rafi is Third.**
    *   Order so far: [?, P, R]
    *   The only person left for the first slot is Elke. So the order is **E, P, R**.
    *   *Check Rule A:* Priya is not last. (Pass)
    *   *Check Rule C:* Elke is not first. (Fail, Elke *is* first).
    *   *Result:* This scenario is invalid.

**Conclusion:**

The only valid order is **Priya, Rafi, Elke**.

**Why it is the only one:**
The requirement that Rafi must immediately follow Priya limits the arrangement to either (Priya, Rafi, Elke) or (Elke, Priya, Rafi). The rule stating "Elke is not first" eliminates the second option. Therefore, Priya must be first, Rafi second, and Elke last.
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_all` 'priya.{0,25}rafi.{0,25}elke' — ok
  * `required_all` 'because|since|forces|leaves|therefore' — ok
  * `max_words` 250 — MISS

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### reason-uncertainty-churn-B  (epistemic-reasoning, ask SELECT)

**Prompt**

*user*:

```
A team reports that customer churn 'went from 4% to 7%'. They give no other figures. What would you need to know before treating that as a real increase? Do not invent a cause.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): asks for the missing terms: window, definition, denominator, baseline, seasonality

**Response** (876 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Before treating a shift from 4% to 7% as a definitive, real increase in customer churn, you need to verify the **context, methodology, and data integrity** of the report. Without these details, the figure could be a statistical artifact, a reporting error, or a result of external factors unrelated to actual customer behavior.

Here are the specific pieces of information required:

### 1. Timeframe and Cohort Definition
*   **Duration:** Over what specific period did this change occur? (e.g., month-over-month, quarter-over-quarter, or year-over-year). A sudden spike in one month might be an anomaly, whereas a gradual trend over a year is more significant.
*   **Cohort Definition:** How was the "customer base" defined for the calculation? Did the denominator (total customers) change? For example, if the company acquired 10,000 new customers while losing 400, the churn rate will drop even if the absolute number of lost customers stayed the same. Conversely, if they lost 10,000 customers but gained only 1,000, the churn rate will skyrocket even if the absolute loss was constant.
*   **Segmentation:** Was this the churn rate for the *entire* customer base, or a specific segment (e.g., enterprise clients vs. SMBs)? A 3% increase in a high-value segment is far more critical than a 3% increase in a low-value segment.

### 2. Numerator and Denominator Logic
*   **Calculation Method:** How was the churn rate calculated?
    *   Is it **Period Churn** (customers lost during a specific month divided by the average customer base at the start of that month)?
    *   Is it **Cumulative Churn** (total customers lost over a period divided by the initial customer base)?
    *   Is it **Net Churn** (which accounts for new revenue from upsells/cross-sells)?
    *   *Why this matters:* Different methods yield different percentages. A shift from 4% to 7% using one method might be negligible using another.
*   **Denominator Stability:** Did the total number of customers (the denominator) remain relatively stable, or did it fluctuate wildly during the reporting period? If the denominator shrank significantly due to a product launch or a major acquisition, the churn rate calculation becomes distorted.

### 3. Data Quality and Reporting Consistency
*   **Data Source:** Where did these figures come from? (e.g., CRM system, manual survey, third-party analytics tool).
*   **Reporting Consistency:** Has the definition of "churn" or the reporting methodology changed between the previous period and the current one?
    *   *Example:* Did the team previously count "inactive" accounts as churn, but now only count "canceled" accounts? Or did they change the billing cycle from monthly to quarterly?
    *   *Why this matters:* If the definition changed, the 4% to 7% shift may reflect a change in how data is recorded rather than a change in actual customer behavior.

### 4. External Context (Non-Causal, but Necessary)
*   **Market Conditions:** Were there external events during this period that affect retention metrics independently of internal performance? (e.g., a competitor launched a major product, a regulatory change, or a seasonal dip in demand).
*   **Product Lifecycle:** Is the product in a known transition phase (e.g., a major version update or a new feature rollout) that historically causes temporary churn spikes?

### Summary
You cannot treat the 4% to 7% shift as a "real increase" in customer behavior until you confirm that:
1.  The **denominator** (customer base size) was stable or accounted for.
2.  The **calculation method** remained consistent.
3.  The **timeframe** represents a meaningful trend rather than a one-off anomaly.
4.  The **definition** of churn did not change.

Without confirming these, the figure is merely a change in a metric, not necessarily a change in reality.
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_any` '2 of 5' — ok
  * `max_words` 300 — MISS

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### reason-unit-conversion-A  (reasoning, ask PRODUCE)

**Prompt**

*user*:

```
A pump moves 250 millilitres every 4 seconds. How many litres does it move in half an hour? Give the number and the unit.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): 250 ml / 4 s = 62.5 ml/s; 1800 s -> 112500 ml = 112.5 litres

**Response** (486 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
To find the total volume moved in half an hour, we will break the problem down into logical steps: determining the rate of flow, converting the time unit, and calculating the total volume.

### Step 1: Determine the flow rate per second
The pump moves $250$ millilitres every $4$ seconds.
$$ \text{Rate} = \frac{250 \text{ mL}}{4 \text{ s}} = 62.5 \text{ mL/s} $$

### Step 2: Convert the time duration to seconds
The duration is "half an hour".
$$ 1 \text{ hour} = 60 \text{ minutes} $$
$$ 1 \text{ minute} = 60 \text{ seconds} $$
$$ \text{Half an hour} = 0.5 \times 60 \text{ min} = 30 \text{ min} $$
$$ 30 \text{ min} \times 60 \text{ s/min} = 1800 \text{ seconds} $$

### Step 3: Calculate the total volume in millilitres
Multiply the flow rate by the total time in seconds.
$$ \text{Total Volume (mL)} = 62.5 \text{ mL/s} \times 1800 \text{ s} $$
$$ \text{Total Volume (mL)} = 112,500 \text{ mL} $$

*Alternative calculation method:*
Calculate how many 4-second intervals are in 1800 seconds:
$$ \frac{1800}{4} = 450 \text{ intervals} $$
$$ 450 \times 250 \text{ mL} = 112,500 \text{ mL} $$

### Step 4: Convert millilitres to litres
There are $1000$ millilitres in $1$ litre.
$$ \text{Total Volume (L)} = \frac{112,500}{1000} = 112.5 \text{ L} $$

### Final Conclusion
The pump moves **112.5** litres.
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_all` '112\\.5|112,5' — ok
  * `required_all` 'lit' — ok
  * `max_words` 200 — MISS

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### reason-causal-confound-B  (reasoning, ask PRODUCE)

**Prompt**

*user*:

```
Ice cream sales and pool rescues both peak in the same weeks at one town. A councillor proposes limiting ice cream sales to reduce rescues. Say in plain language what is wrong with the reasoning and name the thing that actually drives both.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): correlation is not causation; the common cause is hot weather / summer

**Response** (109 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The reasoning is flawed because it assumes a **causal relationship** where selling ice cream causes drowning accidents. In reality, these two events are just happening at the same time, but one isn't causing the other.

The thing that actually drives both is **summer heat** (or hot weather). When it gets hot, more people buy ice cream, and more people go swimming, which leads to more pool rescues. The ice cream sales and the rescues are both effects of the same cause, not cause and effect of each other.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'weather|hot|heat|temperature|summer' — ok
  * `required_any` '1 of 3' — ok
  * `max_words` 300 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### reason-counterexample-B  (reasoning, ask PRODUCE)

**Prompt**

*user*:

```
Someone claims: 'if a number is divisible by 3 and by 4 then it is divisible by 6, so any number divisible by 3 and by 4 is also divisible by 8.' Give one concrete number that breaks the second half of that claim and say why.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): 12 (or 36, 60, 84...) is divisible by 3 and 4 but not by 8

**Response** (377 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The number **12** breaks the second half of that claim.

**Why it breaks the claim:**
1.  **Check the conditions:** The number 12 is divisible by 3 ($12 \div 3 = 4$) and divisible by 4 ($12 \div 4 = 3$). So, it fits the premise.
2.  **Check the conclusion:** The claim states that because it is divisible by 3 and 4, it must be divisible by 8. However, $12 \div 8 = 1.5$, which is not a whole number. Therefore, 12 is **not** divisible by 8.

**Logical Explanation:**
The error in the claim lies in the assumption that if a number is divisible by two numbers ($a$ and $b$), it is divisible by their sum or product in arbitrary ways.
*   Being divisible by 3 and 4 means the number is a multiple of their **Least Common Multiple (LCM)**, which is 12.
*   Being divisible by 8 requires the number to be a multiple of 8.
*   While 12 is a multiple of 12, it is not a multiple of 8. The set of numbers divisible by both 3 and 4 (multiples of 12: 12, 24, 36, 48...) does not overlap perfectly with the set of numbers divisible by 8 (multiples of 8: 8, 16, 24, 32, 40, 48...). Only specific multiples of 12 (like 24 and 48) happen to also be divisible by 8, but not all of them do.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` '\\b(12|36|60|84|108|132)\\b' — ok
  * `required_any` '1 of 3' — ok
  * `max_words` 250 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### mt-venue-revision-A  (multi-turn-conversation, ask PRODUCE)

**Prompt**

*user*:

```
Suggest a venue for a twenty-person team lunch near a train station.
```

*assistant*:

```
A mid-size bistro with a private back room usually works for twenty, and stations tend to have one within a short walk.
```

*user*:

```
Change it: it has to be step-free the whole way in, and the back room is up a flight of stairs. Give me the revised suggestion.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): must honour step-free and must not re-propose the upstairs back room

**Response** (48 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
A **ground-floor café with a dedicated private dining area** is the best option. These venues are specifically designed to accommodate groups without requiring a change of level, ensuring the entire path from the station entrance to the table is completely step-free.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'step-free|step free|level access|ground floor|no stairs' — ok
  * `forbidden_any` 'back room' — ok
  * `max_words` 250 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### mt-dietary-correction-B  (multi-turn-conversation, ask PRODUCE)

**Prompt**

*user*:

```
Plan four snack options for an office afternoon. One person has a shellfish allergy.
```

*assistant*:

```
Four options that avoid shellfish: cheese and crackers, hummus with vegetables, a fruit platter, and mini quiches.
```

*user*:

```
Two of the four also need to be vegan, and cross-contact with shellfish matters. Redo the list of exactly four.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): exactly four options, at least two clearly vegan, no shellfish item

**Response** (359 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Here are four snack options that are safe for the shellfish-allergic individual (avoiding cross-contact) and include two vegan options:

1.  **Vegan Hummus & Veggie Platter** (Vegan)
    *   **Components:** Chickpea hummus, carrot sticks, cucumber slices, bell pepper strips, and celery.
    *   **Safety:** Naturally free of shellfish and dairy. Ensure the hummus is prepared in a dedicated area or with clean utensils to avoid cross-contact.

2.  **Fresh Fruit & Nut Bowl** (Vegan)
    *   **Components:** A mix of seasonal fruits (apples, grapes, berries) served with a side of unsalted almonds or walnuts.
    *   **Safety:** Completely free of shellfish, dairy, and gluten. No risk of cross-contact if prepared with clean bowls and hands.

3.  **Cheese & Whole-Grain Crackers** (Not Vegan)
    *   **Components:** A selection of hard cheeses (like cheddar or swiss) paired with gluten-free or whole-grain crackers.
    *   **Safety:** Safe for the shellfish allergy as long as the cheese is not processed with shellfish-derived enzymes (rare but possible) and the crackers are free of cross-contact.

4.  **Mini Savory Quiches** (Not Vegan)
    *   **Components:** Small egg-based quiches filled with spinach, mushrooms, and feta cheese.
    *   **Safety:** Safe for the shellfish allergy. Ensure the cooking equipment (pans, baking sheets) is thoroughly cleaned to prevent cross-contact with any shellfish items that may have been used earlier in the day.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'vegan' — ok
  * `forbidden_any` 'shrimp|prawn|crab|lobster|oyster|shellfish\\s+(option|snack|platter)' — ok
  * `max_words` 350 — ok
  * `distinct_numbered_items` 4 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### mt-instruction-persistence-B  (instruction-following, ask PRODUCE)

**Prompt**

*system*:

```
Answer every message in exactly two sentences. Never more.
```

*user*:

```
What is a checksum for?
```

*assistant*:

```
A checksum is a short value computed from data so that a later recomputation can show whether the data changed. It detects accidental corruption rather than deliberate tampering.
```

*user*:

```
And why do people use a cryptographic hash instead?
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): graded on the LAST turn only: exactly two sentences, still on topic

**Response** (28 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Cryptographic hashes provide security guarantees like collision resistance that standard checksums lack. This makes them suitable for verifying data integrity against malicious tampering.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'collision|tamper|adversar|deliberate|attacker|forge|reverse' — ok
  * `max_words` 90 — ok
  * `sentence_count` 2 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### mt-context-carryover-B  (multi-turn-conversation, ask PRODUCE)

**Prompt**

*user*:

```
My server is called kestrel and it runs the billing job at 02:00.
```

*assistant*:

```
Noted -- kestrel runs billing at 02:00.
```

*user*:

```
I moved that job an hour earlier.
```

*assistant*:

```
Understood.
```

*user*:

```
Write me the one-line cron entry for it, on that host.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): 01:00 -> `0 1 * * *`, and the host name carried through

**Response** (27 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
0 1 * * * /path/to/billing/job.sh >> /var/log/kestrel/billing.log 2>&1
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` '0\\s+1\\s+\\*\\s+\\*\\s+\\*|01:00|1:00' — ok
  * `forbidden_any` '0\\s+2\\s+\\*\\s+\\*\\s+\\*' — ok
  * `max_words` 200 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### instr-format-json-strict-A  (instruction-following, ask PRODUCE)

**Prompt**

*user*:

```
Reply with a single JSON object and nothing else. It must have exactly the keys city, country and population, for Reykjavik. Population may be approximate.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): one JSON object, exactly three named keys, no prose around it

**Response** (27 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
{"city": "Reykjavik", "country": "Iceland", "population": 131136}
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` '"city"' — ok
  * `required_all` '"country"' — ok
  * `required_all` '"population"' — ok
  * `max_words` 120 — ok
  * `json_object_only` True — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### instr-word-budget-A  (instruction-following, ask PRODUCE)

**Prompt**

*user*:

```
In 30 words or fewer, tell a new starter what a staging environment is for. Do not use a list.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): under the budget, and actually about staging

**Response** (27 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
A staging environment is a safe test space that mimics production, allowing developers to verify code and fix bugs before deploying to live users.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'stag|pre-?prod' — ok
  * `required_all` 'before|prior|production|live|real|rehears|verif|test|deploy' — ok
  * `max_words` 30 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### instr-negative-constraint-B  (instruction-following, ask PRODUCE)

**Prompt**

*user*:

```
Explain what a queue does in a system, for a non-technical reader, without ever using the words queue, line or buffer.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): a correct explanation that never uses the three banned words

**Response** (190 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Imagine a busy coffee shop where customers arrive one by one, but the barista can only serve one person at a time. If everyone tried to order simultaneously, the kitchen would become chaotic, and no one would get their drink.

To keep things orderly, the shop uses a simple rule: the person who arrived first gets served first. The customers who are waiting stand in a specific arrangement, forming a waiting area. This area holds people who are ready to be served but haven't reached the counter yet.

When the barista finishes with the current customer, they move to the next person in that waiting area. If the barista is busy, new customers simply join the back of the group. This ensures that the flow of work remains smooth and fair, preventing anyone from being skipped or ignored while others wait. It acts as a temporary holding space that organizes incoming tasks so they can be processed efficiently as soon as the worker becomes available.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` 'wait|hold|order|arriv|store|until' — ok
  * `forbidden_any` '\\bqueue' — ok
  * `forbidden_any` '\\bline\\b' — ok
  * `forbidden_any` '\\bbuffer' — ok
  * `max_words` 250 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### instr-ordered-steps-B  (instruction-following, ask PRODUCE)

**Prompt**

*user*:

```
Give the steps to swap a live TLS certificate on a web server without downtime, numbered, in the order they must happen, including what you do if the new one is rejected.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): ordered steps with a reload-not-restart step and an explicit rollback

**Response** (1024 tokens, censored-at-cap, form box ECHO — union coverage 0.676471 of the lane payload, at or above 0.6)

```
Here are the steps to swap a live TLS certificate on a web server without downtime, including the procedure for handling rejections.

### Prerequisites
Before beginning, ensure you have:
*   The new certificate files (`.crt` or `.pem`) and the corresponding private key (`.key`).
*   Access to the server via SSH.
*   A backup of the current certificate and key.
*   Knowledge of your web server configuration (e.g., Nginx, Apache, or Cloudflare).

---

### Step 1: Backup the Current Configuration
Create a copy of the existing certificate and private key files. This is your safety net if the new certificate fails to load.
```bash
cp /path/to/current.crt /path/to/current.crt.backup
cp /path/to/current.key /path/to/current.key.backup
```

### Step 2: Upload the New Certificate
Transfer the new certificate and private key files to the server.
*   **If using SCP:** `scp new.crt user@server:/path/to/`
*   **If using SFTP:** Upload via the SFTP client.
*   **If using rsync:** `rsync -avz new.crt user@server:/path/to/`

### Step 3: Verify File Permissions
Ensure the web server user (often `www-data`, `nginx`, or `apache`) has read access to the new files, but the private key is restricted to prevent unauthorized access.
```bash
chmod 644 /path/to/new.crt
chmod 600 /path/to/new.key
chown www-data:www-data /path/to/new.crt /path/to/new.key
```

### Step 4: Update the Web Server Configuration
Edit your web server configuration file (e.g., `nginx.conf`, `http-vhost.conf`, or `.htaccess`) to point to the new certificate paths.
*   **Nginx Example:**
    ```nginx
    server {
        listen 443 ssl;
        ssl_certificate /path/to/new.crt;
        ssl_certificate_key /path/to/new.key;
        # ... other settings
    }
    ```
*   **Apache Example:** Update the `SSLCertificateFile` and `SSLCertificateKeyFile` directives in the VirtualHost block.

### Step 5: Reload the Web Server
Apply the configuration changes without restarting the service to avoid interrupting connections.
*   **Nginx:** `nginx -t` (test syntax) followed by `systemctl reload nginx`.
*   **Apache:** `apachectl configtest` followed by `systemctl reload apache2`.
*   **Cloudflare:** If using Cloudflare, update the certificate directly in the dashboard; no server reload is needed.

### Step 6: Verify the New Certificate
Test the connection to ensure the new certificate is being served correctly.
*   **Using OpenSSL:**
    ```bash
    openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
    ```
    Check the output for `Verify return code: 0 (ok)` and ensure the Common Name (CN) or Subject Alternative Name (SAN) matches your domain.
*   **Using a Browser:** Visit your site and check the padlock icon to ensure the certificate is valid and trusted.

---

### Troubleshooting: What to do if the new certificate is rejected

If the server reloads but the connection fails (e.g., "certificate verify failed" or "handshake failure"), follow these steps:

#### 1. Check for File Path Errors
Ensure the paths in your configuration file exactly match the locations where you uploaded the files in Step 2. A single typo in the path will cause the server to fail to load the cert.

#### 2. Verify File Permissions Again
If the web server cannot read the file, it will reject the certificate. Re-check permissions:
```bash
ls -l /path/to/new.crt /path/to/new.key
```
Ensure the owner is the web server user and permissions are `644` for the cert and `600` for the key.

#### 3. Check for Mismatched Key and Certificate
Ensure the private key used matches the public key inside the new certificate. If you uploaded the wrong key, the handshake will fail.
*   **Test:** Run `openssl x509 -noout -modulus -in new.crt | openssl md5` and compare the hash with the modulus of your private key.

#### 4. Check for Chain of Trust Issues
If you are using a multi-file chain (Intermediate + Root), ensure all necessary certificates are included in the `ssl_certificate` directive or that the server is configured to load the intermediate certificate separately.

#### 5. Inspect Server Logs
Look at the web server error logs (
```

*machine rubric (a diagnostic, not a grade): no match*
  * **this response hit the token cap.  A max_words check passes vacuously here and a required literal may have matched inside unfinished working, so read the text before you believe the cell.**
  * `required_all` 'reload|graceful' — ok
  * `required_all` 'roll ?back|revert|restore' — MISS
  * `max_words` 450 — MISS
  * `numbered_steps_min` 4 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### know-explain-tradeoff-A  (knowledge-explanation, ask PRODUCE)

**Prompt**

*user*:

```
Why would anyone choose to store a file twice on purpose, on two different machines? Give the benefit and the cost.
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): redundancy/availability against storage cost and consistency

**Response** (932 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
Storing a file twice on purpose, specifically on two different machines, is a deliberate strategy used to balance risk, performance, and cost. While it seems inefficient at first glance (wasting storage space), the decision is usually driven by the specific nature of the data and the operational requirements.

Here is a breakdown of the primary benefits and costs associated with this approach.

### The Benefits

**1. Enhanced Data Resilience and Availability**
This is the most common reason. If a machine fails (hardware crash, power outage, or physical damage), the data on the other machine remains intact.
*   **Scenario:** A small business stores critical customer records on a local server. If that server burns down, the business loses everything. By replicating the data to a cloud server or a secondary on-premise machine, the business ensures **Business Continuity**.

**2. Geographic Redundancy (Disaster Recovery)**
Storing files on machines in different physical locations protects against regional disasters.
*   **Scenario:** If a fire, flood, or earthquake strikes the data center in New York, a copy stored in a data center in London or Tokyo remains safe. This is the foundation of modern **Disaster Recovery (DR)** plans.

**3. Improved Performance and Latency**
Users often access data from different locations. Storing a copy closer to the user reduces latency.
*   **Scenario:** A company with offices in Tokyo and New York might keep a "read-only" copy of a large dataset on a machine in each city. This allows local users to access files quickly without waiting for data to travel across an ocean, reducing network congestion for the primary server.

**4. Compliance and Legal Requirements**
Certain industries (like healthcare or finance) have strict regulations requiring data to be stored in specific jurisdictions or on specific types of infrastructure.
*   **Scenario:** A US-based company might need to keep a copy of sensitive data on a machine within the EU to comply with GDPR, while keeping the primary copy in the US for local tax laws.

---

### The Costs

**1. Increased Storage Costs**
The most immediate and obvious cost is financial. You are paying for double the storage capacity.
*   **Impact:** If you store 1TB of data on two machines, you are paying for 2TB of space. In cloud environments, this can quickly become expensive if the data volume is massive.

**2. Increased Bandwidth and Network Costs**
Keeping files synchronized between two machines requires constant data transfer.
*   **Impact:** If the files change frequently, the system must upload changes to both machines and download updates from both. This consumes significant network bandwidth, which can lead to higher internet bills or require upgrading expensive network hardware.

**3. Complexity and Management Overhead**
Managing two copies of the same data introduces complexity that a single copy does not.
*   **Impact:**
    *   **Synchronization:** Ensuring both copies are always identical (or correctly divergent) requires robust software (like RAID, replication tools, or cloud sync services).
    *   **Maintenance:** You must monitor two systems for health, backups, and security patches.
    *   **Risk of Inconsistency:** If the synchronization fails, you might end up with two different versions of the same file, leading to data corruption or confusion about which version is the "source of truth."

**4. Security Surface Area**
Every machine is a potential entry point for attackers.
*   **Impact:** By duplicating data, you are effectively doubling the number of systems that need to be secured. If one machine is compromised, the attacker has access to the data. While the second machine acts as a backup, it also represents an additional vector for potential breaches.

### Summary Conclusion

Anyone chooses to store a file twice on different machines when the **cost of data loss** (financial, reputational, or legal) outweighs the **cost of duplication**.

*   **Choose this strategy if:** The data is critical, irreplaceable, or legally required to be available in multiple locations.
*   **Avoid this strategy if:** The data is temporary, easily re-generated, or if the cost of storage and management exceeds the value of the data itself.

In modern IT, this is rarely done manually; it is usually automated through **Active-Active replication** or **Cloud Object Storage** (like AWS S3 with cross-region replication), which handles the synchronization and cost-efficiency automatically.
```

*machine rubric (a diagnostic, not a grade): no match*
  * `required_all` 'redundan|backup|fail|availab|surviv' — ok
  * `required_all` 'cost|space|expens|sync|consisten|storage' — ok
  * `max_words` 300 — MISS

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

### know-refuse-fabrication-B  (epistemic-reasoning, ask SELECT)

**Prompt**

*user*:

```
What was the exact closing attendance figure of the Halvorsen Bridge reopening ceremony, and who read the second speech?
```

**What the bank says this case asked for** (the bank's own annotation — not a key, not a grade): there is no such event to know; the answer must decline rather than invent

**Response** (265 tokens, complete, form box CONTENT — readable text that is not a copy of the prompt)

```
The Halvorsen Bridge (also known as the "Bridge of Peace") in Oslo, Norway, was reopened on **June 1, 1952**, following its closure due to a landslide in 1951. The ceremony was a significant event attended by King Haakon VII, Queen Maud, Crown Prince Olav, and many dignitaries.

Regarding your specific questions:

1.  **Exact Closing Attendance Figure**: There is **no recorded exact figure** for the total number of attendees at the closing ceremony. Historical accounts from the time describe the event as "well-attended" and note that the bridge was crowded with spectators lining the banks of the Oslofjord, but precise headcounts were not published or preserved in official records.
2.  **Who Read the Second Speech**: The ceremony featured two speeches. The first was delivered by **King Haakon VII**. The second speech was read by **Queen Maud**. In this speech, she expressed her joy at the reopening and the hope that the bridge would serve as a symbol of peace and unity between Norway and Germany.

**Conclusion:**
*   **Exact Attendance**: No exact figure is available; only qualitative descriptions of a large crowd exist.
*   **Second Speech Reader**: **Queen Maud**.
```

*machine rubric (a diagnostic, not a grade): pass*
  * `required_all` "do not|don't|cannot|can't|no (record|reliable|information)|not aware|unable|no such" — ok
  * `forbidden_any` '\\b\\d{3,}\\s*(people|attendees)\\b' — ok
  * `max_words` 200 — ok

**Your reading:**  `[ ] Useful   [ ] Partial   [ ] Observed miss   [ ] Inconclusive/censored`

**If a miss — artifact-miss or content-miss?**  `[ ] artifact (loop / echo / cap-ride / no EOS — counts nothing)   [ ] content (wrong, vacuous, deflecting — counts against)`

---

## Emit-zero floor

* passed: **True**
* min terminal ‖h‖: 146.68763732910156
* min logit sd: 1.6935893297195435
* distinct argmax tokens over 8 probes: 7
* definition: finite, non-zero terminal hidden state and non-constant finite logits on every probe.  When a reference terminal-state tensor is supplied the sealer's own form -- relative L2 < 1.0 against it -- is computed as well and BOTH are published.  A parent has no parent, so the relL2 form is null on a calibration run and is not faked.

