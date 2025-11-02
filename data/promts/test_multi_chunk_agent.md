# Test Query Generator - Multi Chunk

## Kontext
Du generierst Testfragen für ein RAG-System (Retrieval-Augmented Generation), das Informationen über medizinische Kühl- und Gefriergeräte bereitstellt.

## Aufgabe
Erstelle **1-2 komplexere Fragen**, die ein Nutzer stellen könnte und die **nur durch Kombination ALLER gegebenen Chunks** vollständig beantwortet werden können.

## Regeln

1. **Jede Frage muss ALLE Chunks benötigen** - keine Frage, die nur mit einem Chunk beantwortbar ist
2. **Verschiedene Fragetypen verwenden:**
   - Zusammenhänge: "Welche Features hat das Gerät und welche Abmessungen?"
   - Kombinationen: "Was sind Temperaturbereich und Energieverbrauch...?"
   - Übersichten: "Welche technischen Daten und Sicherheitsfeatures hat...?"
3. **Nutze natürliche Sprache** - so wie ein Labormitarbeiter oder Einkäufer fragen würde
4. **Produktname und/oder Hersteller in der Frage nennen**!
5. **Konkret und spezifisch** - die Frage soll klar zeigen, dass mehrere Aspekte gefragt sind

## Zielgruppe der Fragen
- Labormitarbeiter, die umfassende Informationen suchen
- Einkäufer, die mehrere Eigenschaften auf einmal prüfen
- Techniker, die Gesamtüberblick benötigen

---

## Eingabe (Chunks):
{chunks_text}

---

## Ausgabeformat (JSON-Array):
[
  "Frage 1: Komplexe Frage, die alle Chunks benötigt",
  "Frage 2: Optional weitere Frage"
]


## Beispiele guter Multi-Chunk-Fragen:
- "Welche Abmessungen, Temperaturbereich und Sicherheitsfeatures hat der Kirsch LABO-288?"
- "Was sind die technischen Daten und welche Alarmfunktionen bietet der Liebherr SFFsg-5501?"
- "Welche Ausstattungsmerkmale und welchen Energieverbrauch hat der Haier HYC-85GD?"

## Wichtig:
- Alle Fragen müssen **mehrere Chunks kombinieren**
- Keine Fragen, die nur mit einem Chunk beantwortbar wären
- Die Chunks stammen vom gleichen Produkt - nutze das für zusammenhängende Fragen
