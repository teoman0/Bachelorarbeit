# Datensatz-Audit fuer Kapitel 3.1

Dieses Skript dient der fruehen strukturellen Pruefung der Datensatzeignung. Es
trainiert kein Modell, erzeugt keine Patches und bewertet kein Testset fuer
Modellentscheidungen. Ziel ist ein reproduzierbares Inventar der lokalen
Klassenordner, bevor Split-Strategie und spaetere Trainingsschritte festgelegt
werden.

## Erwartete lokale Struktur

```text
data/raw/
  schleifgrad_1/
  schleifgrad_2/
  schleifgrad_3/
  schleifgrad_4/
```

`data/raw/` bleibt lokal und ist durch `.gitignore` vom Repository
ausgeschlossen. Das Skript speichert nur kleine CSV-Tabellen und kuratierte
PNG-Abbildungen in `reports/`.

## Aufruf

```bash
python scripts/audit_dataset.py --data-root data/raw --output-dir reports
```

Falls `python` unter Windows nicht im PATH liegt, kann die lokale virtuelle
Umgebung direkt genutzt werden:

```powershell
.\.venv\Scripts\python.exe scripts\audit_dataset.py --data-root data/raw --output-dir reports
```

Nuetzliche Optionen:

```bash
python scripts/audit_dataset.py \
  --data-root data/raw \
  --output-dir reports \
  --seed 42 \
  --samples-per-class 4 \
  --group-regex "^(teil_[0-9]+)"
```

`--group-regex` ist optional. Wenn Dateinamen Bauteil-, Proben- oder
Aufnahmeserien-IDs enthalten, sollte eine passende Regex mit einer Capture
Group angegeben werden. Das Ergebnis landet in `possible_group_id` und kann
spaeter helfen, Leakage-Risiken und eine korrekte Split-Ebene zu diskutieren.
Ohne Regex nutzt das Skript nur eine einfache Dateinamensheuristik.

## Ausgaben

Das Skript erzeugt:

- `reports/tables/dataset_summary.csv`: Anzahl Dateien je Klasse,
  Bildgroessenbereiche, Dateiendungen und korrupt/unlesbare Dateien.
- `reports/tables/image_inventory.csv`: eine Zeile pro Bilddatei mit relativem
  Pfad, Label, Bildgroesse, Kanalzahl, Dateigroesse, MD5-Hash, einfachem
  perceptual hash und optionaler Gruppen-ID aus dem Dateinamen.
- `reports/tables/potential_duplicates.csv`: Paare mit identischem MD5-Hash
  oder sehr aehnlichem perceptual hash.
- `reports/tables/dataset_audit_metadata.json`: Aufrufparameter, Seed,
  Git-Commit, Dirty-Status und Paketversionen.
- `reports/figures/class_distribution.png`: Balkendiagramm der Klassenverteilung.
- `reports/figures/image_size_distribution.png`: Streudiagramm der Bildgroessen
  und Histogramm der Bildflaechen.
- `reports/figures/sample_grid_per_class.png`: kleines Thumbnail-Raster mit
  Beispielbildern pro Klasse.

## Datenschutz und Versionierung

`sample_grid_per_class.png` enthaelt reale Beispielbilder aus dem lokalen
Datensatz. Diese Abbildung ist fuer die lokale Sichtpruefung gedacht und soll
nur nach ausdruecklicher Freigabe versioniert oder oeffentlich geteilt werden.

`image_inventory.csv`, `potential_duplicates.csv` und
`dataset_audit_metadata.json` koennen Dateinamen, lokale Pfade, Hashes oder
Rueckschluesse auf den Datensatz enthalten. Diese Dateien bleiben
standardmaessig lokal und werden nicht ins oeffentliche Repository committed.

## Methodische Einordnung

`num_images` in `dataset_summary.csv` zaehlt alle gefundenen Bilddateien mit
passender Dateiendung. Korrupt oder unlesbare Dateien werden zusaetzlich in
`num_corrupt_images` ausgewiesen; Groessenstatistiken werden nur aus lesbaren
Bildern berechnet.

Der perceptual hash ist ein einfacher horizontaler Difference-Hash. Er ist nur
ein Hinweis auf moegliche Duplikate oder nahezu gleiche Aufnahmen und ersetzt
keine manuelle Pruefung. Ein Treffer in `potential_duplicates.csv` sollte
besonders dann untersucht werden, wenn die betroffenen Dateien in verschiedenen
Klassen liegen oder spaeter in verschiedene Splits geraten koennten.

Die Ergebnisse duerfen fuer Kapitel 3.1 genutzt werden, um Klassenbalance,
Bildgroessen, korrupte Dateien, moegliche Wiederholungsaufnahmen und
Leakage-Risiken zu beschreiben. Sie duerfen nicht zur Modellauswahl,
Hyperparameterwahl, Schwellenwertwahl oder finalen Testset-Interpretation
verwendet werden.
