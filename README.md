# TOPSIS Decision Support System

This repository contains a comprehensive implementation of the *Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)*. The project covers three main components: a command-line interface (CLI) tool, a published Python package (PyPI), and a web-based service that emails results to users.

## 📸 Web Service Implementation (Local Host)

As part of the assignment's final stage, a web service was developed to provide a graphical user interface for the TOPSIS algorithm.


*Features shown in the screenshot:*
•⁠  ⁠*File Upload:* Accepts ⁠ .csv ⁠ files containing numeric data.
•⁠  ⁠*Parameter Input:* text fields for Weights (e.g., ⁠ 1,1,1,1 ⁠) and Impacts (e.g., ⁠ +,+,-,+ ⁠).
•⁠  ⁠*Email Integration:* A field to capture the user's email address to send the resultant ranked file.
•⁠  ⁠*Status Indicators:* Success messages confirming the processing and email delivery.

---
DEPLOYED WEBSITE

https://topsis2-omega.vercel.app/

## 🛠️ Project Workflow & Methodology

This project was executed in three distinct phases, following the assignment specifications:

### Phase 1: Core Logic & CLI Implementation
The fundamental logic of TOPSIS was implemented in Python.
•⁠  ⁠[cite_start]*Input Handling:* The script reads an input CSV file, validating that it contains three or more columns and that all value columns contain numeric data[cite: 13, 14].
•⁠  ⁠[cite_start]*Validation:* It ensures the number of weights and impacts matches the number of columns and handles "File Not Found" exceptions[cite: 10, 12, 15].
•⁠  ⁠[cite_start]*Execution:* The script processes the data and outputs a new CSV file containing the original data with appended 'Topsis Score' and 'Rank' columns[cite: 6, 8].

### Phase 2: Python Package Creation (PyPI)
To make the tool accessible, the code was modularized into a Python package.
•⁠  ⁠*Packaging:* The project was structured with necessary ⁠ setup.py ⁠ and configuration files.
•⁠  ⁠[cite_start]*Naming Convention:* The package follows the format ⁠ Topsis-FirstName-RollNumber ⁠[cite: 22].
•⁠  ⁠[cite_start]*Distribution:* The package was uploaded to PyPI, allowing users to install it via pip and run it from the command line[cite: 21, 27].

### Phase 3: Web Service Development
The final phase involved wrapping the logic in a web framework (e.g., Flask).
•⁠  ⁠[cite_start]*Frontend:* A clean HTML interface was designed to accept inputs (File, Weights, Impacts, Email) [cite: 30-36].
•⁠  ⁠*Backend Processing:* The server processes the uploaded file using the TOPSIS logic.
•⁠  ⁠[cite_start]*SMTP Integration:* The final result file is automatically emailed to the address provided by the user[cite: 39].

---

## 🧮 Mathematics of TOPSIS

[cite_start]TOPSIS is based on the concept that the chosen alternative should have the shortest geometric distance from the *positive ideal solution* (best possible values) and the longest geometric distance from the *negative ideal solution* (worst possible values)[cite: 2].

### Step 1: Normalization
The vector normalization technique is used to transform different scales and units into a common measurable scale.
For a value $x_{ij}$, the normalized value $r_{ij}$ is calculated as:

$$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}$$

### Step 2: Weight Assignment
We calculate the weighted normalized decision matrix $v_{ij}$ by multiplying the normalized value by its associated weight $w_j$:

$$v_{ij} = w_j \times r_{ij}$$

### Step 3: Determine Ideal Solutions
[cite_start]We identify the Ideal Best ($V^+$) and Ideal Worst ($V^-$) values for each criterion based on the impact (positive ⁠ + ⁠ or negative ⁠ - ⁠)[cite: 16].

•⁠  ⁠*Positive Impact (+):* Maximize benefit.
    * $V_j^+ = \max(v_{ij})$
    * $V_j^- = \min(v_{ij})$
•⁠  ⁠*Negative Impact (-):* Minimize cost.
    * $V_j^+ = \min(v_{ij})$
    * $V_j^- = \max(v_{ij})$

### Step 4: Calculate Euclidean Distances
Calculate the separation of each alternative from the ideal solutions.

*Distance from Positive Ideal ($S_i^+$):*
$$S_i^+ = \sqrt{\sum_{j=1}^{n} (v_{ij} - V_j^+)^2}$$

*Distance from Negative Ideal ($S_i^-$):*
$$S_i^- = \sqrt{\sum_{j=1}^{n} (v_{ij} - V_j^-)^2}$$

### Step 5: Calculate Performance Score
The relative closeness to the ideal solution is calculated as:

$$P_i = \frac{S_i^-}{S_i^+ + S_i^-}$$

### Step 6: Ranking
The alternatives are ranked based on $P_i$ values in descending order. The highest score represents the best alternative.

---

## 🚀 How to Run

### Command Line
```bash
python <program.py> <InputDataFile> <Weights> <Impacts> <OutputResultFileName>


