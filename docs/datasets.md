## PhiUSIIL Phishing URL Dataset
- Used for: phishing and legitimate URLs
- Source: UCI Machine Learning Repository, dataset 967
- Authors: Arvind Prasad and Shalini Chandra
- License: CC BY 4.0 (attribution required)
- Citation: Prasad, A., & Chandra, S. (2023). PhiUSIIL: A diverse security profile
  empowered phishing URL detection framework based on similarity index and
  incremental learning. Computers & Security, 103545.
- Date downloaded: 17 Nov 2026
- SHA-256 of phiusiil.csv: [paste hash]
- Rows in original file: 235,795
- After cleaning: 235362 rows, 100512 phishing, 134850 legitimate,
  8 dropped as invalid
- Label direction verified: original label 1 = legitimate (134,850 rows),
  label 0 = phishing (100,945 rows), matching the dataset page; remapped so 1 = phishing
- Columns used: URL and label only. The other columns were ignored on purpose
  because some were computed with knowledge of the label.
- Limitations: balance between classes is artificial; legitimate and phishing URLs
  may differ in style (for example "www." and https habits), so a model could learn
  the source instead of phishing; collection period should be checked in the paper;
  no domain age or reputation data