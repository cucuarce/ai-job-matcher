"""
Paso 1: Extraer las ofertas de trabajo desde los .docx a un dataset JSON limpio.
Esto es el input real para el paso de embeddings.
"""

import docx
import json
import re

# Colocá tus .docx originales en data/raw/ (esa carpeta está en .gitignore,
# nunca se sube al repo público)
ARCHIVOS = [
    "data/raw/Ofertas_V1.docx",
    "data/raw/Ofertas_V2.docx",
]

DELIMITADOR = "Acerca del empleo"


def extraer_ofertas(path: str) -> list[str]:
    """Parte el documento en bloques, uno por oferta, usando el delimitador."""
    doc = docx.Document(path)
    textos = [p.text for p in doc.paragraphs if p.text.strip()]

    indices_inicio = [i for i, t in enumerate(textos) if DELIMITADOR in t]
    ofertas = []
    for idx, inicio in enumerate(indices_inicio):
        fin = indices_inicio[idx + 1] if idx + 1 < len(indices_inicio) else len(textos)
        bloque = textos[inicio:fin]
        ofertas.append("\n".join(bloque).strip())

    return ofertas


def main():
    dataset = []
    id_global = 1

    for path in ARCHIVOS:
        ofertas = extraer_ofertas(path)
        origen = path.split("/")[-1]
        print(f"{origen}: {len(ofertas)} ofertas extraídas")

        for texto in ofertas:
            # Título aproximado: primera línea con contenido real después del delimitador
            lineas = [l.strip() for l in texto.split("\n") if l.strip()]
            titulo = lineas[1] if len(lineas) > 1 else "Sin título"

            dataset.append({
                "id": id_global,
                "origen": origen,
                "titulo": titulo[:150],
                "texto_completo": texto,
                "n_caracteres": len(texto)
            })
            id_global += 1

    with open("data/raw/ofertas.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(dataset)} ofertas guardadas en data/raw/ofertas.json")
    print("\nEjemplo (oferta #1):")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2)[:500])


if __name__ == "__main__":
    main()
