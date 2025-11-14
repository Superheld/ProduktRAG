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
   - **Produktname = NUR Hersteller + Modellbezeichnung**
   - **Entferne SEO-Erweiterungen** wie "Laborkühlschrank", "Gefrierschrank", "Medikamentenkühlschrank" aus dem Produktnamen
   - ✅ RICHTIG: "Welche Temperatur hat der [Hersteller] [Modell]?"
   - ❌ FALSCH: "Welche Temperatur hat der [Hersteller] [Modell] [Kategorie]?"
5. **Konkret und spezifisch** - keine vagen oder allgemeinen Fragen

## Zielgruppe der Fragen
- Labormitarbeiter, die technische Details suchen
- Einkäufer, die Produkte vergleichen möchten
- Techniker, die Spezifikationen prüfen

---

## Eingabe (Chunk):
{chunk_text}

---

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
