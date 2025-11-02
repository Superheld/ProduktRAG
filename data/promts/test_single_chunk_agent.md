# Test Query Generator - Single Chunk

## Kontext
Du generierst Testfragen für ein RAG-System (Retrieval-Augmented Generation), das Informationen über medizinische Kühl- und Gefriergeräte bereitstellt.

## Aufgabe
Erstelle **2-3 präzise Fragen**, die ein Nutzer stellen könnte und die **genau durch den gegebenen Chunk beantwortet** werden können.

## Regeln

1. **Jede Frage muss durch den Chunk vollständig beantwortbar sein** - keine Informationen, die nicht im Chunk stehen
2. **Verschiedene Fragetypen verwenden:**
   - Faktenfragen: "Welche Temperatur...?"
   - Spezifikationsfragen: "Was sind die Abmessungen...?"
   - Feature-Fragen: "Hat das Gerät...?"
   - Ja/Nein-Fragen: "Verfügt der X über...?"
3. **Nutze natürliche Sprache** - so wie ein Labormitarbeiter oder Einkäufer fragen würde
4. **Produktname und/oder Hersteller in der Frage nennen**!
5. **Konkret und spezifisch** - keine vagen oder allgemeinen Fragen

## Zielgruppe der Fragen
- Labormitarbeiter, die technische Details suchen
- Einkäufer, die Produkte vergleichen möchten
- Techniker, die Spezifikationen prüfen

---

## Eingabe (Chunk):
{chunk_text}

---

## Ausgabeformat (JSON-Array):
[
  "Frage 1: Konkrete Frage zum Chunk",
  "Frage 2: Weitere Frage zum Chunk",
  "Frage 3: Optional dritte Frage"
]


## Beispiele guter Fragen:
- "Welchen Temperaturbereich hat der Kirsch LABO-288?"
- "Verfügt der Liebherr SFFsg-5501 über eine automatische Abtauung?"
- "Was sind die Außenmaße des Haier HYC-85GD?"
- "Hat der Kirsch FROSTER LABEX-530 einen explosionsgeschützten Innenraum?"

## Wichtig:
- Alle Fragen müssen mit Informationen **nur aus diesem einen Chunk** beantwortbar sein
- Keine Fragen, die mehrere Chunks oder Vergleiche benötigen
