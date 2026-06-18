# DINOv3-Prototyp-Heatmaps fuer BMW_25

## Zweck

Dieser qualitative Test prueft, ob DINOv3-Patchfeatures auf dem lokalen
BMW_25-Datensatz raeumlich interpretierbare Hinweise auf die vier
Bearbeitungs-/Schleifgrade liefern koennen.

Der Test ist kein Training einer Segmentierung und keine finale Bewertung. Er
nutzt Bildlabels aus den vier Klassenordnern nur, um je Klasse einen einfachen
Feature-Prototyp aus Referenzbildern zu bilden. Preview-Bilder werden danach
patchweise mit diesen Prototypen verglichen.

## Methode

1. Bilder werden aus den vier Klassenordnern gelesen.
2. Aus Dateinamen wie `_DSC0162_q1.JPG` wird die Gruppe `_DSC0162`
   abgeleitet.
3. Referenz- und Preview-Auswahl erfolgen gruppensicher, damit Viertel
   desselben Ursprungsbildes nicht auf beiden Seiten landen.
4. Gruppen, die in mehreren Klassenordnern vorkommen, werden nicht fuer die
   Referenz-Prototypen genutzt. Fuer Preview-Bilder werden solche Gruppen
   bevorzugt, weil sie besonders interessant fuer lokale Unterschiede sind.
5. DINOv3 bleibt eingefroren. Es werden nur Patch-Embeddings extrahiert.
6. Pro Klasse wird aus normalisierten Referenz-Patchfeatures ein
   normalisierter Klassenprototyp gebildet.
7. Jedes Preview-Patch wird per Kosinus-Aehnlichkeit mit den vier Prototypen
   verglichen.
8. Die Visualisierung zeigt:
   - die patchweise naechste Klasse,
   - ein ordinal gewichtetes Farbbild ueber die vier Bearbeitungsstufen,
   - direkte transparente Overlays in Originalbildgroesse,
   - Patchscores als CSV.

## Interpretation

Eine Heatmap aus diesem Verfahren ist eine schwach ueberwachte
Aehnlichkeitskarte, keine Ground-Truth-Segmentierung. Wenn ein Bild lokal
abweichende Oberflaechenbereiche hat, kann die Karte anzeigen, welche Patches
DINOv3 eher einem anderen Schleifgrad-Prototyp zuordnet. Das ist ein plausibler
Hinweis fuer weitere Untersuchung, aber kein Beweis fuer lokal korrekte
Schleifgradlabels.

## Erster lokaler Testlauf

Der erste Lauf am 2026-06-18 nutzte den lokalen Ordner
`BMW_25/Viertel BMW gefiltert` mit 4.607 Bildern:

- Fraeszustand: 1.198 Bilder
- Stufe 1: 614 Bilder
- Stufe 2: 1.340 Bilder
- Final: 1.455 Bilder

Aus den Dateinamen wurden 1.547 Gruppen abgeleitet. Davon kamen 19 Gruppen in
mehr als einem Klassenordner vor. Diese Gruppen sind fuer qualitative
Heatmaps besonders relevant, weil sie auf lokale Unterschiede,
Grenzfaelle oder uneinheitliche Viertel desselben Ursprungsbildes hinweisen
koennen.

Fuer den Test wurden pro Klasse vier reine Referenzgruppen ausgewaehlt. Mit
Patchgroesse und Stride `224` entstanden je Klasse 648 Referenz-Patches. Fuer
die Preview wurden pro Klasse ein Bild und bevorzugt Mehrklassen-Gruppen
genutzt, soweit vorhanden.

## Ausfuehrung

Alle Befehle werden aus dem Repository-Root ausgefuehrt:

```powershell
cd C:\path\to\Bachelorarbeit
```

Eine passende Minimalumgebung kann mit der pinbaren Requirements-Datei
installiert werden:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dinov3-prototype.txt
```

Der Datensatzpfad wird nicht versioniert, sondern lokal als Umgebungsvariable
gesetzt:

```powershell
$env:BMW25_DATA_ROOT = "C:\path\to\BMW_25\Viertel BMW gefiltert"
```

```powershell
.\.venv\Scripts\python.exe scripts\dinov3_prototype_heatmaps.py --config configs\dinov3_bmw25_prototype_heatmaps.yaml
```

Alle Ergebnisse werden lokal unter
`outputs/dinov3_bmw25_prototype_heatmaps/` gespeichert und nicht versioniert.

Die wichtigsten Bilddateien pro Preview-Bild sind:

- `*_class_direct.png`: direkte Klassen-Heatmap auf dem Originalbild.
- `*_ordinal_direct.png`: weichere ordinale Mischung auf dem Originalbild.
- `*_class_compare.png`: Originalbild links, direkte Klassen-Heatmap rechts.
- `*_class_overlay.png`: Variante mit Legende am rechten Rand.
