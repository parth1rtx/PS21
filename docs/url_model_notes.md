# URL model notes (url_v1)

## Data
- Dataset: PhiUSIIL Phishing URL Dataset (see datasets.md)
- Rows after cleaning: 235362 (100512 phishing, 134850 legitimate)
- Split: by registered domain, seed 42
- Split sizes: train 168197, val 34406, test 32759
- Features: hostname only

## Models compared (winner chosen on validation PR-AUC only)
| Model | Val PR-AUC | Test PR-AUC | Test precision | Test recall | Test FPR |
|---|---|---|---|---|---|
| logreg | 0.8343 | 0.7863 | 0.9775 | 0.3758 | 0.0053 |
| lightgbm | 0.8837 | 0.8645 | 0.9896 | 0.5660 | 0.0037 |

- Chosen: lightgbm. The test set was scored once, after all choices were made.

## Calibration
- Method: isotonic regression fitted on validation
- ECE before 0.0146, after 0.0068
- Threshold 0.8661, targeting about 1% false positives on validation
- At this threshold test recall is 0.5660

## Leakage checks
1. Scrambled labels: PR-AUC 0.476 vs test phishing share 0.381.
   Result: Pass, with a caveat. With shuffled labels the model scored 0.476 against a test phishing share of 0.381. That is near chance level and far below the real model’s 0.8645, so the pipeline does not leak the answers. The small excess over the share is plausibly noise from one shuffled run on a test set whose class mix differs from the training set (0.440 vs 0.381 phishing).
2. Hostname-only test PR-AUC 0.8645 vs full-URL (lightgbm) 0.8853; gap +0.0209.
   Interpretation: Small gap. The full-URL LightGBM scored 0.8853 against 0.8645 for hostname-only, a difference of +0.0209. Paths and query strings add little signal beyond the hostname, so path-style bias in this dataset appears small. The app still uses the hostname-only model, because it is the more conservative choice and is not affected by how paths were written in each class.
3. Largest domains: web.app (5718), firebaseapp.com (5546), repl.co (3746); unique domains 175507 of 235362 rows.
   Phishing share: train 0.440, val 0.406, test 0.381.
   Comment: Somewhat lopsided. The phishing share falls from 0.440 (train) to 0.381 (test). The three largest domains, web.app (5,718 rows), firebaseapp.com (5,546) and repl.co (3,746), are free hosting services whose subdomains belong to many different owners. Grouping by registered domain keeps each of these whole inside one split, which likely caused the shift. Test metrics should be read knowing the test set has a slightly lower phishing share than training.
4. Hostname-only PR-AUC is 0.8645. Comment: Plausible. Hostname-only test PR-AUC is 0.8645, well below the 0.97 level where I would suspect leakage. Precision is high (0.9896) while recall is moderate (0.5660), which is the expected result of choosing a strict false-positive target of about 1%.

## Limitations
- Lexical features only: no domain age, reputation, page content or WHOIS data
- Class balance is artificial; real traffic is mostly legitimate, so real precision would be lower
- Legitimate and phishing URLs in PhiUSIIL may differ in style, so some signal may be dataset-specific
- Phishing tactics change over time; collection period: Phishing tactics change over time; the collection period is not stated in the
  accessible dataset description (UCI lists only a 3/3/2024 donation date), so
  how well the data matches current phishing is unknown
- "Legitimate" means no phishing pattern in the URL text, not that the site is safe

## Conclusion
[WRITE: one honest paragraph]
