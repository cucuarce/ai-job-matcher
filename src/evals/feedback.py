"""
Feedback humano sobre el último resultado de evals, y comparación contra los checks automáticos.

Para cada caso mostrás la respuesta del agente y la puntuás de 1 (mala) a 5 (muy buena).
Al final se muestra dónde coinciden y dónde NO coinciden humano y checks automáticos:
los desacuerdos son la información más valiosa (qué mide mal el check, o qué se le escapa).

Uso:
    python -m src.evals.feedback [ruta_resultados.json]
"""

import json
import sys

from src.evals.run_evals import DIR_RESULTADOS

UMBRAL_HUMANO = 4  # >= 4 se considera "respuesta buena"


def pedir_puntaje() -> tuple[int | None, str]:
    while True:
        raw = input("Puntaje 1-5 (Enter para saltear): ").strip()
        if raw == "":
            return None, ""
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw), input("Nota (opcional): ").strip()
        print("  Ingresá un número del 1 al 5.")


def comparar(casos: list[dict]) -> None:
    evaluados = [c for c in casos if c["feedback_humano"]]
    if not evaluados:
        print("Sin feedback cargado.")
        return
    print("\n=== Humano vs. checks automáticos ===")
    coinciden = 0
    for c in evaluados:
        humano_ok = c["feedback_humano"]["puntaje"] >= UMBRAL_HUMANO
        auto_ok = c["aprobado_auto"]
        coinciden += humano_ok == auto_ok
        estado = "coinciden" if humano_ok == auto_ok else "DESACUERDO"
        print(f"  {c['id']:<28} humano={c['feedback_humano']['puntaje']}/5  auto={'OK' if auto_ok else 'FALLA'}  -> {estado}")
    print(f"\nAcuerdo: {coinciden}/{len(evaluados)}")


def main() -> None:
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        archivos = sorted(DIR_RESULTADOS.glob("*.json"))
        if not archivos:
            sys.exit("No hay resultados. Corré primero: python -m src.evals.run_evals")
        ruta = archivos[-1]

    datos = json.loads(open(ruta, encoding="utf-8").read())
    print(f"Archivo: {ruta}\nModelo: {datos['provider']}/{datos['model']}")

    for c in datos["casos"]:
        print(f"\n{'=' * 70}\nCaso: {c['id']}\nPregunta: {c['pregunta']}\n\nRespuesta del agente:\n{c['respuesta']}\n")
        print(f"Checks automáticos: {c['checks']}")
        puntaje, nota = pedir_puntaje()
        if puntaje:
            c["feedback_humano"] = {"puntaje": puntaje, "nota": nota}

    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)
    comparar(datos["casos"])


if __name__ == "__main__":
    main()
