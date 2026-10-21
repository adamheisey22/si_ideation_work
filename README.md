# Suicidal Ideation (SI) Detection in Clinical Notes

## Overview

Suicidal ideation and behavior are often recorded only in free-text clinical notes, not in structured fields. These notes are full of negations ("denies SI"), historical mentions ("hx of suicide attempt in 2015"), mentions about other people ("father died by suicide"), and boilerplate (crisis-line disclaimers, screening-form headers). A plain keyword search over-flags them.

This repository compares three rule-based NLP approaches for flagging suicidal ideation (SI) and suicidal behavior (SB) in notes:

1. **C-SSRS-aligned rules (`CSSRS_script.py`).** These are NegEx-based rules organized around [Columbia Suicide Severity Rating Scale](https://cssrs.columbia.edu/) concepts. They separately flag **current SI**, **historical SI**, **current SB**, and **historical SB**, plus SI with a **plan** or **intent**. They use the lexicons in `SI_Files/` and the trigger sets in `NegExRules2/`.
2. **WCM NegEx-based rules (`rb_script.py`).** This is the rule-based approach from the [Weill Cornell Medicine SI_Ideation repository](https://github.com/wcmc-research-informatics/SI_Ideation). Each SI mention is classified as positive or negated, using separate NegEx trigger sets for natural-language sentences and for structured "question: answer" text (`NegexRules/`).
3. **pyConText (`pycontext_script.py`).** It uses [pyConTextNLP](https://github.com/chapmanbe/pyConTextNLP) with SI targets and negation modifiers (`pycontext/`) to label each sentence. Sentence labels are then combined into a document label, using either "any positive" or a majority rule.

All three approaches are transparent, rule-based baselines. You can inspect and change the lexicons and triggers as plain text files.

> **Data:** This repository contains no clinical notes or annotations. You must supply your own note data. Do not commit real patient text, identifiers, or local database and export files.

## Repository layout

| Path | Purpose |
|---|---|
| `CSSRS_script.py` | C-SSRS-aligned SI/SB detection (current, historical, plan, intent) |
| `run_algorithm_notebook.ipynb` | Notebook version of the C-SSRS pipeline, plus document-level evaluation cells |
| `rb_script.py` | WCM NegEx-based current-SI classifier |
| `pycontext_script.py` | pyConText-based SI classifier |
| `negex.py` | NegEx implementation, adapted from Peter Kang's negex |
| `support_functions.py` | Preprocessing (HTML stripping, date/year masking, disclaimer scrubbing) and tagger helpers |
| `note_extraction.py` | Loads a folder of `.txt` notes into a CSV with `ID` and `NOTE_TEXT` columns |
| `data_prep.py` | Cleans raw notes from a SQLite table (HTML removal, disclaimer scrub, separating out the C-SSRS form) |
| `results_evaluation.py` | Example patient-level evaluation against a known case cohort |
| `SI_Files/` | SI, SB, plan, and intent lexicons; screening-assessment phrases to ignore |
| `NegexRules/` | Natural and structured NegEx triggers for the WCM script (`*_a.txt` variants drop historical triggers) |
| `NegExRules2/` | SI/SB, historical, and intent triggers for the C-SSRS script |
| `pycontext/` | pyConText target and modifier definitions |
| `BRAT/annotation.conf` | [BRAT](https://brat.nlplab.org/) annotation schema used to label SI/SB, negation, plan, intent, and history |
| `rq.txt` | Python requirements |

## Setup

These scripts target Python 3.7 and the pinned package versions in `rq.txt`.

```bash
git clone https://github.com/adamheisey22/si_ideation_work.git
cd si_ideation_work
python3 -m venv SI_env
source SI_env/bin/activate
pip install -r rq.txt
pip install beautifulsoup4 openpyxl   # used by support_functions.py and for Excel I/O
```

For pyConText, also install the spaCy English model and apply one patch:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
```

With newer Python versions, you need to comment out line 489 of `pyConTextNLP/ConTextMarkup.py` inside your virtual environment (for example `SI_env/lib/python3.7/site-packages/pyConTextNLP/ConTextMarkup.py`). Otherwise you get `AttributeError: 'dict_keyiterator' object has no attribute 'sort'`.

Run every script from the repository root, because the lexicon and rule files are loaded by relative path.

## Usage

### 1. C-SSRS rules

`CSSRS_script.py` reads notes from a local SQLite database. It expects a file named `notes_DB_test.db`, a table named `si_data_all_clean`, and a `NOTE_TEXT` column. It appends its results to a table named `si_data_results` in the same database. The command-line arguments are currently commented out in the script, so change the database name, query, or `note_c` near the bottom of the file to point it at your data.

```bash
python CSSRS_script.py
```

Or open `run_algorithm_notebook.ipynb`, set the input data in the first cell, and run the notebook.

Output columns include `predSI`, `predhistSI`, `predSB`, `predhistSB`, `predSI_plan`, `predSI_intent`, and `not_relevant`. Each has a matching set of the text spans that triggered it (for example `pos_SI_men` and `neg_SI_men`).

### 2. WCM NegEx-based rules

```bash
python rb_script.py -d "datapath.xlsx" -nc "note_text" -s "save_path.xlsx" -c 1 -a 1 -tc "SI Label"
```

| Arg | Meaning |
|---|---|
| `-d` | Input Excel file |
| `-nc` | Note text column |
| `-s` | Output Excel file |
| `-c` | `1` = current-SI triggers (`NegexRules/*_rules.txt`, which treat historical mentions as negated); `0` = the `*_a.txt` variants |
| `-a` | `1` = print accuracy and classification report; `0` = skip (the flag must be passed) |
| `-tc` | Ground-truth label column (needed when `-a 1`) |

Output columns: `pos`, `neg`, `pos_men`, `neg_men`, `pred_SI`.

You can customize the rules in these files:
- Add SI keywords to `SI_Files/SI_phrases.txt`.
- Add screening-assessment text to ignore to `SI_Files/SI_assessments.txt`.
- Add negation triggers to `NegexRules/natural_rules.txt` (natural sentences) or `NegexRules/structured_rules.txt` (structured "question: answer" text).

### 3. pyConText

```bash
python pycontext_script.py -d "datapath.csv" -nc "NOTE_TEXT" -s "save_path.csv" -m 0 -a 1 -tc "SI Label"
```

| Arg | Meaning |
|---|---|
| `-d` | Input CSV file (the file must have a `NOTE_TEXT` column) |
| `-nc` | Note text column |
| `-s` | Output CSV file |
| `-m` | `0` = the document is positive if any sentence is positive; `1` = majority rule |
| `-a` | `1` = print accuracy and classification report; `0` = skip |
| `-tc` | Ground-truth label column (needed when `-a 1`) |

The modifier and target YAML files are downloaded at runtime from the WCM repository. Local copies are in `pycontext/`. Output column: `pycontext_label`.

### Optional preprocessing and evaluation

- `note_extraction.py` turns a folder of `.txt` notes into a CSV. Set `path` in the script first.
- `data_prep.py` strips HTML, removes crisis-line disclaimer text, and moves the embedded C-SSRS form into a separate `CSSRS_FORM` column. Set the database folder in the script first.
- `results_evaluation.py` evaluates at the patient level: it checks whether each known case has at least one relevant note. It expects local cohort files and the `si_data_results` table produced by the C-SSRS script.

## Citations and credit

- **Original SI detection code:** this code was adapted and modified from the Weill Cornell Medicine Research Informatics repository: https://github.com/wcmc-research-informatics/SI_Ideation
- **NegEx:** the negex code was downloaded from the following repository, with credit to Peter Kang: https://github.com/mongoose54/negex/tree/master/negex.python. It was downloaded and altered for this classification problem.
  - Chapman WW, Bridewell W, Hanbury P, Cooper GF, Buchanan BG. A simple algorithm for identifying negated findings and diseases in discharge summaries. *J Biomed Inform.* 2001;34(5):301–310.
- **pyConText / ConText:** https://github.com/chapmanbe/pyConTextNLP
  - Harkema H, Dowling JN, Thornblade T, Chapman WW. ConText: an algorithm for determining negation, experiencer, and temporal status from clinical reports. *J Biomed Inform.* 2009;42(5):839–851.
- **C-SSRS:** Columbia Suicide Severity Rating Scale: https://cssrs.columbia.edu/
- **BRAT annotation tool:** https://brat.nlplab.org/

## License

GPL-3.0. See [LICENSE](LICENSE).
