# Dataset-Audit Results

## Audit run

- Dataset: BMW_25 / Viertel BMW gefiltert
- Audit status: erfolgreich lokal ausgefuehrt
- Images total: 4607
- Readable images: 4607
- Corrupt/unreadable images: 0
- Number of classes: 4

## Class distribution

| Class | Number of images |
| --- | ---: |
| Erste Bearbeitungsstufe Viertel | 614 |
| Zweite Bearbeitungsstufe Viertel | 1340 |
| Fraeszustand Viertel | 1198 |
| Finaler Zustand Viertel | 1455 |
| Total | 4607 |

## Duplicate and similarity note

The audit found 10 identical MD5 duplicate pairs and 6947
perceptual-hash-similar pairs. These findings do not automatically indicate
invalid data, but they are relevant for leakage-safe split planning. Similar or
duplicated images must be inspected before defining Train/Validation/Test
splits, especially if visually related images could otherwise be distributed
across different splits.

## Methodological consequence

Train/Validation/Test splits must be defined before patch generation. If images
belong to the same part, acquisition series, source image, or quartered
original image, those related images should remain within the same split.

## Privacy note

Detailed inventories, image paths, hashes, duplicate-pair files and sample
image grids are kept local and are not committed to the public repository.
