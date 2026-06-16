# Datensatzrichtlinien

In diesem Ordner werden keine Rohdaten gespeichert. Rohbilder, generierte Patches, Feature-Caches, Modellgewichte und grosse Zwischenprodukte bleiben lokal und sind durch `.gitignore` ausgeschlossen.

## Geplante lokale Struktur

Die folgende Struktur kann lokal genutzt werden, ohne Rohdaten in GitHub zu versionieren:

```text
data/
  raw/          # originale Bilddaten, nicht in Git
  interim/      # temporäre Zwischendaten, nicht in Git
  processed/    # abgeleitete Datensaetze, nicht in Git
  patches/      # generierte Patches, nicht in Git
  cache/        # Feature-Caches, nicht in Git
  splits/       # kleine Split-Manifeste, falls datenschutzrechtlich und praktisch ok
```

## Split-Grundsatz

Train/Val/Test-Splits muessen vor der Patch-Erzeugung auf Originalbild-, Bauteil- oder Aufnahmeebene erfolgen. Wenn Bauteil-, Proben- oder Aufnahme-IDs verfuegbar sind, sollen diese fuer gruppierte Splits genutzt werden.

Nicht erlaubt:

- Patches eines Originalbildes ueber Train, Val und Test verteilen.
- Aufnahmen desselben Bauteils zufaellig in verschiedene Splits legen, wenn dadurch visuelle Wiedererkennung moeglich ist.
- Das Testset zur Modell-, Hyperparameter-, Augmentierungs- oder Patchgroessenauswahl verwenden.

## Empfohlenes Split-Manifest

Ein versionierbares Split-Manifest kann zum Beispiel als CSV unter `data/splits/` liegen, sofern es keine sensiblen Daten enthaelt:

```text
image_id,relative_path,label,split,group_id,part_id,acquisition_id,notes
```

Mindestspalten:

- `image_id`: stabile Bild-ID,
- `relative_path`: Pfad relativ zu einem lokalen Datenwurzelverzeichnis,
- `label`: Schleifgradklasse,
- `split`: `train`, `val` oder `test`,
- `group_id`: Gruppierung fuer leakage-sichere Splits, z. B. Bauteil oder Aufnahmeserie.

## Datenintegritaet

Fuer reproduzierbare Experimente sollen spaeter Checksummen oder eine Datensatzversion dokumentiert werden. Die eigentlichen Bilddaten bleiben lokal, aber ihre Struktur, Anzahl pro Klasse und Split-Logik muessen nachvollziehbar sein.

## Patchdaten

Patchdaten werden ausschliesslich aus bereits gesplitteten Originalbildern erzeugt. Die Patchdateien selbst werden nicht versioniert. Stattdessen sollen spaeter die Patch-Parameter in einer Config dokumentiert werden, zum Beispiel:

- Patchgroesse,
- Stride oder Ueberlappung,
- Randbehandlung,
- Filterregeln fuer leere oder unbrauchbare Patches,
- Zuordnung zum Originalbild,
- Koordinaten im Originalbild.
