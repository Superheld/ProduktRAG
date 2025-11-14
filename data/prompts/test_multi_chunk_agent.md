# Test Query Generator - Multi Chunk

## Kontext
Du generierst Testfragen für ein RAG-System (Retrieval-Augmented Generation), das Informationen über medizinische Kühl- und Gefriergeräte bereitstellt.

## Aufgabe
Erstelle **2 komplexere Fragen**, die ein Nutzer stellen könnte und die **nur durch Kombination ALLER gegebenen Chunks** vollständig beantwortet werden können.

## Regeln

1. **Jede Frage muss ALLE Chunks benötigen** - keine Frage, die nur mit einem Chunk beantwortbar ist
2. **Verschiedene Fragetypen verwenden:**
   - Zusammenhänge: "Welche Features hat das Gerät und welche Abmessungen?"
   - Kombinationen: "Was sind Temperaturbereich und Energieverbrauch...?"
   - Übersichten: "Welche technischen Daten und Sicherheitsfeatures hat...?"
3. **Nutze natürliche Sprache** - so wie ein Labormitarbeiter oder Einkäufer fragen würde
4. **Produktname und/oder Hersteller in der Frage nennen**!
   - **Produktname = NUR Hersteller + Modellbezeichnung**
   - **Entferne SEO-Erweiterungen** wie "Laborkühlschrank", "Gefrierschrank", "Medikamentenkühlschrank" aus dem Produktnamen
   - ✅ RICHTIG: "Welche Features und Abmessungen hat der [Hersteller] [Modell]?"
   - ❌ FALSCH: "Welche Features hat der [Hersteller] [Modell] [Kategorie]?"
5. **Konkret und spezifisch** - die Frage soll klar zeigen, dass mehrere Aspekte gefragt sind

## Zielgruppe der Fragen
- Labormitarbeiter, die umfassende Informationen suchen
- Einkäufer, die mehrere Eigenschaften auf einmal prüfen
- Techniker, die Gesamtüberblick benötigen

---

## Eingabe (Chunks):
["[Chunk 1]", "[Chunk 2]"]
---

## Beispiele für Fragestrukturen (NICHT kopieren - eigene Fragen generieren!):
- Format: "Welche [ASPEKT1], [ASPEKT2] und [ASPEKT3] hat der/das [PRODUKT]?"
- Format: "Was sind die [TECHNISCHE_DATEN] und welche [FEATURES] bietet der/das [PRODUKT]?"
- Format: "Welche [EIGENSCHAFTEN] und welchen [SPEZIFIKATION] hat der/das [PRODUKT]?"

## Wichtig:
- **Generiere NEUE Fragen basierend NUR auf dem gegebenen Chunk-Inhalt**
- **Kopiere NICHT die obigen Format-Beispiele - nutze sie nur als Inspiration**
- Alle Fragen müssen **mehrere Chunks kombinieren**
- Keine Fragen, die nur mit einem Chunk beantwortbar wären
- Die Chunks stammen vom gleichen Produkt - nutze das für zusammenhängende Fragen
