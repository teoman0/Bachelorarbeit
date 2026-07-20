# CVAT-Regionenanalyse-Plan

Stand: 2026-07-20

## Ziel

Diese Notiz beschreibt eine lokale Regionenpipeline fuer die
Schleifgradanalyse. Die vorhandenen manuellen CVAT-Annotationen werden als
rechteckige Regionen interpretiert und in eine tabellarische Form ueberfuehrt.
Die Pipeline ist keine semantische Segmentierung und trainiert kein Modell.

Die Regionentabelle soll spaeter genutzt werden, um lokale Bildbereiche mit
dem besten DINOv3-Partial-Fine-Tuning-Modell auszuwerten. Das Testset bleibt
fuer die finale Evaluation reserviert und wird nicht fuer Modellwahl,
Schwellenwerte oder qualitative Validierungsentscheidungen verwendet.

## Eingaben

Die lokalen CVAT-Dateien liegen ausserhalb des Repositorys im
`manual_all`-Verzeichnis. Der lokale Pfad wird nicht versioniert, sondern per
Umgebungsvariable oder CLI-Parameter uebergeben.

Erwartete Dateien innerhalb dieses lokalen Verzeichnisses:

- `annotations/cvat_annotations.json`
- `annotations/frame_meta.json`
- `annotations/labels.json`
- `manifest.csv`
- `images/`

Das bestehende gruppierte Split-Manifest bleibt verbindlich:

```text
data/splits/bmw25_grouped_split_manifest.csv
```

Es werden keine neuen Splits erzeugt.

## Label-Mapping

| CVAT-Label | Ziel-Label |
| --- | --- |
| `Fraeszustand` | `Fräszustand Viertel` |
| `Stufe_1` | `Erste Bearbeitungssufe Viertel` |
| `Stufe_2` | `Zweite Bearbeitungsstufe Viertel` |
| `Final` | `Finaler Zustand Viertel` |
| `Nicht_bewertbar` | Sonderklasse, nicht Teil der vier globalen Klassen |

`Nicht_bewertbar` wird als Sonderklasse erhalten. Standardmaessig wird diese
Sonderklasse aus der exportierten Regionentabelle ausgeschlossen; mit
`--include-nicht-bewertbar` kann sie fuer eine lokale Bestandsaufnahme
mitgeschrieben werden.

## Tabellenlogik

Das Skript extrahiert ausschliesslich CVAT-Objekte vom Typ `rectangle`. Die
Koordinaten werden an die Bildgrenzen geclippt. Jede Region erhaelt eine
deterministische `region_id`.

Die Bildzuordnung erfolgt ueber den aus dem Originaldateinamen abgeleiteten
`group_id`. Dazu werden lokale manuelle Praefixe wie `manual_v1___` entfernt
und q1- bis q4- beziehungsweise Viertel-Suffixe entfernt. Anschliessend wird
gegen `group_id` im bestehenden Split-Manifest gematcht. Nicht zuordenbare
Bilder bleiben als `matched_manifest=false` markiert, sofern
`--exclude-unmatched` nicht gesetzt wird.

Die Regionentabelle enthaelt mindestens:

```text
region_id, source_image, original_image_name, group_id, split, original_label,
mapped_label, is_global_class, image_width, image_height, x_min, y_min, x_max,
y_max, bbox_width, bbox_height, bbox_area, bbox_area_ratio, clipped,
matched_manifest, exclude_reason
```

## Lokale Befehle

Dry-Run ohne Dateiausgabe:

```powershell
python scripts/prepare_cvat_region_annotations.py `
  --manual-root <lokales_manual_all_verzeichnis> `
  --dry-run `
  --split all `
  --include-nicht-bewertbar
```

Lokaler Export der Regionentabelle und Summary:

```powershell
python scripts/prepare_cvat_region_annotations.py `
  --manual-root <lokales_manual_all_verzeichnis> `
  --allow-export `
  --split train `
  --exclude-unmatched
```

Crops werden nur mit expliziter Freigabe geschrieben:

```powershell
python scripts/prepare_cvat_region_annotations.py `
  --manual-root <lokales_manual_all_verzeichnis> `
  --allow-export `
  --allow-export-crops `
  --split train `
  --exclude-unmatched
```

## Lokale Artefakte

Standard-Ausgabeort:

```text
outputs/cvat_region_analysis/manual_all/
```

Moegliche lokale Dateien:

- `region_annotations.csv`
- `region_annotations_summary.json`
- `crops/`, nur bei `--allow-export-crops`

Diese Artefakte bleiben lokal und werden nicht committed.

## Empfehlung fuer die naechste Auswertung

Die CVAT-Rechtecke sind fuer eine lokale rechteckbasierte Regionenanalyse
geeignet, aber nicht fuer eine echte semantische Segmentierung. Sinnvoll ist
als naechster Schritt eine reine Train/Validation-Auswertung mit dem bereits
gewaehlten DINOv3-Partial-Fine-Tuning-Modell. Testregionen duerfen erst in der
spaeteren finalen Evaluation verwendet werden.
