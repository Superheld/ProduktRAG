# Test Query Generator - Single Chunk

## Kontext
Du generierst Testfragen für ein RAG-System (Retrieval-Augmented Generation), das Informationen über medizinische Kühl- und Gefriergeräte bereitstellt.

## Aufgabe
Erstelle **1-3 Fragen** basierend auf der Komplexität des Chunks:
- Enthält der Chunk nur EINE Information? → Generiere 1-2 Frage
- Enthält er mehrere verschiedene Informationen? → Generiere 2-3 unterschiedliche Fragen zu verschiedenen Aspekten

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


## Beispiele für Fragestrukturen (NICHT kopieren - eigene Fragen generieren!):
- Format: "Welchen [SPEZIFIKATION] hat der/das [PRODUKT]?"
- Format: "Verfügt der/das [PRODUKT] über [FEATURE]?"
- Format: "Was sind die [TECHNISCHE_ANGABE] des [PRODUKT]?"
- Format: "Hat der/das [PRODUKT] einen/eine [EIGENSCHAFT]?"

## Wichtig:
- **Generiere NEUE Fragen basierend NUR auf dem gegebenen Chunk-Inhalt**
- **Kopiere NICHT die obigen Format-Beispiele - nutze sie nur als Inspiration**
- Alle Fragen müssen mit Informationen **nur aus diesem einen Chunk** beantwortbar sein
- Keine Fragen, die mehrere Chunks oder Vergleiche benötigen
