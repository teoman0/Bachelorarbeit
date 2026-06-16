# Projektkontext

## Thema

Die Bachelorarbeit untersucht Vision-Transformer-Modelle zur bildbasierten Bewertung des Schleifgrades metallischer Oberflaechen.

## Problemstellung

Metallische Oberflaechen koennen je nach Bearbeitungs- und Schleifprozess unterschiedliche visuelle Merkmale aufweisen. Ziel ist es, aus Bilddaten den Schleifgrad automatisiert vorherzusagen. Die geplanten Experimente vergleichen klassische globale Bildklassifikation mit einer patchbasierten lokalen Auswertung.

## Methodischer Fokus

Die Arbeit soll nicht nur Modellmetriken berichten, sondern eine saubere experimentelle Methodik zeigen:

- reproduzierbare Datensplits,
- klare Trennung von Training, Validierung und Test,
- dokumentierte Modell- und Lizenzentscheidungen,
- nachvollziehbare Configs,
- keine unbeabsichtigte Informationsuebertragung zwischen Splits.

## Globale Klassifikation

Bei der globalen Klassifikation erhaelt ein Modell ein Bild und sagt einen Schleifgrad fuer das gesamte Bild vorher. Diese Sichtweise passt, wenn jedes Bild genau ein globales Label besitzt und dieses Label fuer den sichtbaren Bildinhalt plausibel ist.

## Warum keine echte semantische Segmentierung?

Eine echte semantische Segmentierung wuerde pixelgenaue Zielmasken benoetigen. Wenn nur Bildlabels fuer den Schleifgrad vorhanden sind, waere ein Segmentierungstraining methodisch nicht sauber, weil dem Modell keine verlaesslichen Pixel- oder Regionenlabels bereitgestellt werden. Ein solches Vorgehen wuerde den Eindruck einer raeumlich exakten Ground Truth erzeugen, die im Datensatz nicht existiert.

## Patchbasierte lokale Klassifikation als Annaeherung

Die patchbasierte Variante zerlegt ein Bild in kleinere Bildausschnitte und klassifiziert diese Patches. Die Patchvorhersagen koennen anschliessend wieder auf die Bildpositionen projiziert und als Heatmap visualisiert werden.

Diese Methode ist keine echte Segmentierung. Sie ist eine Annaeherung, weil jedes Patchlabel aus dem uebergeordneten Bildlabel abgeleitet wird. Dadurch entsteht Labelrauschen, besonders wenn ein Bild lokal heterogene Oberflaechenbereiche enthaelt. Die Methode kann trotzdem sinnvoll sein, um zu untersuchen, ob das Modell raeumlich konsistente Hinweise auf den Schleifgrad lernt.

## Data-Leakage-Risiken

Bei Oberflaechenbildern koennen Leakage-Risiken besonders leicht entstehen:

- mehrere Bilder desselben Bauteils,
- mehrere Aufnahmen derselben Oberflaechenstelle,
- ueberlappende Bildausschnitte,
- Patches aus demselben Originalbild,
- nahezu identische Aufnahmebedingungen ueber verschiedene Dateien hinweg.

Deshalb muessen Splits vor der Patch-Erzeugung auf der passenden Ebene erfolgen. Wenn Bauteil- oder Aufnahme-IDs verfuegbar sind, ist ein Split auf dieser Ebene einem rein zufaelligen Bildsplit vorzuziehen.

## Testset-Regel

Das Testset ist fuer die finale Bewertung reserviert. Es darf nicht zur Auswahl von Modellarchitektur, Lernrate, Augmentierung, Patchgroesse, Aggregationsmethode, Schwellenwerten oder Checkpoints genutzt werden. Diese Entscheidungen werden ausschliesslich ueber Training und Validierung getroffen.
