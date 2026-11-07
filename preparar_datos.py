"""Partición por escena y recortes reproducibles. Ejecutar desde la raíz del proyecto."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml

CLASES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]
ROOT = Path(__file__).resolve().parent


def leer_etiquetas(path):
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Fila inválida en {path}")
        values = np.asarray(fields, dtype=float)
        c, x, y, w, h = values
        if (not np.isfinite(values).all() or c != int(c) or not 0 <= c < 5
                or min(x, y) < 0 or max(x, y) > 1 or min(w, h) <= 0
                or max(w, h) > 1 or min(x-w/2, y-h/2) < -2e-6
                or max(x+w/2, y+h/2) > 1+2e-6):
            raise ValueError(f"Caja inválida en {path}")
        rows.append([int(c), x, y, w, h])
    return rows


def auditar(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Imagen ilegible {path}")
    label = path.parents[2] / "labels" / "train" / (path.stem + ".txt")
    rows = leer_etiquetas(label)  # Un TXT faltante debe fallar.
    counts = Counter(int(row[0]) for row in rows)
    return dict(image=f"images/train/{path.name}", label=f"labels/train/{path.stem}.txt",
                scene="__".join(path.stem.split("__")[:2]), source=path.stem.split("__")[0],
                width=image.shape[1], height=image.shape[0],
                counts=[counts[c] for c in range(5)],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def posiciones(length, size, overlap):
    if size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Tamaño o solapamiento inválido")
    last = max(0, length-size)
    return sorted(set(range(0, last+1, max(1, int(size*(1-overlap))))) | {last})


def etiquetas_recorte(rows, width, height, x, y, size, visible=0.7):
    """Rechazar la ventana si deja un fragmento de un barco sin etiquetar."""
    crop_w, crop_h = min(size, width-x), min(size, height-y)
    output = []
    for c, xc, yc, w, h in rows:
        bx1, by1 = (xc-w/2)*width, (yc-h/2)*height
        bx2, by2 = (xc+w/2)*width, (yc+h/2)*height
        ix1, iy1 = max(x, bx1), max(y, by1)
        ix2, iy2 = min(x+crop_w, bx2), min(y+crop_h, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2-ix1)*(iy2-iy1)/((bx2-bx1)*(by2-by1)) < visible:
            return None
        output.append([c, ((ix1+ix2)/2-x)/crop_w, ((iy1+iy2)/2-y)/crop_h,
                       (ix2-ix1)/crop_w, (iy2-iy1)/crop_h])
    return output


def elegir_split(records, seed):
    groups = defaultdict(list)
    for row in records:
        groups[row["scene"]].append(row)
    names = sorted(groups)
    sources = sorted({row["source"] for row in records})
    # Balancear cajas por clase, fuentes e imágenes, manteniendo escenas completas.
    features = np.array([
        [sum(r["counts"][c] for r in groups[g]) for c in range(5)]
        + [sum(r["source"] == s for r in groups[g]) for s in sources]
        + [len(groups[g])] for g in names], dtype=float)
    totals = features.sum(0)
    rng = np.random.default_rng(seed)
    best, error = None, float("inf")
    for _ in range(200):
        selected = rng.choice(len(names), max(1, round(len(names)*0.15)), replace=False)
        val = features[selected].sum(0)
        if np.any(val[:5] == 0) or np.any(totals[:5]-val[:5] == 0):
            continue
        loss = np.square(val / np.maximum(totals, 1) - 0.15).mean()
        if loss < error:
            best, error = selected, loss
    if best is None:
        raise ValueError("No fue posible crear validación con cinco clases")
    val_groups = {names[i] for i in best}
    for row in records:
        row["split"] = "val_dev" if row["scene"] in val_groups else "train_dev"
    hash_splits = defaultdict(set)
    for row in records:
        hash_splits[row["sha256"]].add(row["split"])
    if any(len(s) > 1 for s in hash_splits.values()):
        raise ValueError("Imágenes idénticas cruzan el split")


def muestra(records, count, seed):
    # Incluir cinco clases y la escena más densa para la prueba de memoria.
    rng = random.Random(seed)
    shuffled = sorted(records, key=lambda r: r["image"])
    rng.shuffle(shuffled)
    chosen = {max(records, key=lambda r: sum(r["counts"]))["image"]}
    for c in range(5):
        chosen.add(next(r["image"] for r in shuffled if r["counts"][c]))
    for row in shuffled:
        if len(chosen) >= count:
            break
        chosen.add(row["image"])
    return [r for r in records if r["image"] in chosen]


def preparar(args):
    data, target = args.data_root.resolve(), args.output_root.resolve()
    if target.exists():
        raise FileExistsError(f"Ya existe {target}. Usar sus manifiestos o una salida nueva.")
    stage = target.with_name(target.name + "_building")
    stage.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    images = sorted((data/"images/train").glob("*.jpg"))
    if not images:
        raise ValueError("No se encontraron imágenes de entrenamiento")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(auditar, images))
    if args.manifest:
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        by_name = {r["image"]: r for r in previous}
        if set(by_name) != {r["image"] for r in records}:
            raise ValueError("El manifiesto y el dataset tienen imágenes distintas")
        for row in records:
            old = by_name[row["image"]]
            if old["sha256"] != row["sha256"] or old["split"] not in {"train_dev", "val_dev"}:
                raise ValueError(f"Manifiesto incompatible: {row['image']}")
            row["split"] = old["split"]
        grouped = defaultdict(set)
        for row in records:
            grouped[row["scene"]].add(row["split"])
            grouped[row["sha256"]].add(row["split"])
        if any(len(s) > 1 for s in grouped.values()):
            raise ValueError("El manifiesto cruza escenas o hashes entre splits")
    else:
        elegir_split(records, args.seed)
    subsets = {s: [r for r in records if r["split"] == s] for s in ["train_dev", "val_dev"]}
    for split, rows in subsets.items():
        for kind in ["images", "labels"]:
            (stage/kind/split).mkdir(parents=True)
        for row in rows:
            for key, kind in [("image", "images"), ("label", "labels")]:
                shutil.copy2(data/row[key], stage/kind/split/Path(row[key]).name)
    (stage/"manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    for split in subsets:
        small = muestra(subsets[split], 128 if split == "train_dev" else 64, args.seed)
        paths = [str(target/"images"/split/Path(r["image"]).name) for r in small]
        (stage/f"smoke_{split}.txt").write_text("\n".join(paths), encoding="utf-8")
    for name, train, val in [
        ("data", "images/train_dev", "images/val_dev"),
        ("smoke", "smoke_train_dev.txt", "smoke_val_dev.txt"),
    ]:
        (stage/f"{name}.yaml").write_text(yaml.safe_dump(
            dict(path=str(target), train=train, val=val, names=CLASES)), encoding="utf-8")
    summary = {s: dict(images=len(rows), objects=sum(sum(r["counts"]) for r in rows),
               classes=np.sum([r["counts"] for r in rows], axis=0).tolist())
               for s, rows in subsets.items()}
    (stage/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stage.rename(target)
    print(json.dumps(summary), flush=True)


def cortar(args):
    dev, target = args.development.resolve(), args.output_root.resolve()
    if target.exists():
        raise FileExistsError(f"Ya existe {target}")
    stage = target.with_name(target.name+"_building")
    stage.mkdir(parents=True, exist_ok=False)
    for kind in ["images", "labels"]:
        (stage/kind/"train").mkdir(parents=True)
    records = json.loads((dev/"manifest.json").read_text(encoding="utf-8"))
    records = [r for r in records if r["split"] == "train_dev"]
    if args.limit:
        records = muestra(records, args.limit, args.seed)
    manifest, rejected = [], 0
    cv2.setNumThreads(max(1, args.workers))
    for i, row in enumerate(records):
        source = dev/"images/train_dev"/Path(row["image"]).name
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"No se puede leer {source}")
        height, width = image.shape[:2]
        rows = leer_etiquetas(dev/"labels/train_dev"/(source.stem+".txt"))
        for y in posiciones(height, args.size, args.overlap):
            for x in posiciones(width, args.size, args.overlap):
                boxes = etiquetas_recorte(rows, width, height, x, y, args.size, args.visible)
                if boxes is None:
                    rejected += 1
                    continue
                key = f"{args.seed}_{source.stem}_{x}_{y}"
                chance = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)/2**32
                if not boxes and chance >= args.background:
                    continue
                name = f"{source.stem}__crop{x}_{y}"
                tile = image[y:y+args.size, x:x+args.size]
                if not cv2.imwrite(str(stage/"images/train"/(name+".jpg")), tile):
                    raise OSError(f"Fallo al guardar {name}")
                lines = [" ".join([str(b[0])] + [f"{v:.6f}" for v in b[1:]]) for b in boxes]
                (stage/"labels/train"/(name+".txt")).write_text("\n".join(lines), encoding="utf-8")
                manifest.append(dict(tile=name, source=row["image"], scene=row["scene"],
                                     x=x, y=y, width=tile.shape[1], height=tile.shape[0], boxes=len(boxes)))
        if i % 200 == 0:
            print(f"Tiles: {i}/{len(records)} imágenes, {len(manifest)} recortes", flush=True)
    if not manifest:
        raise ValueError("No se generaron recortes válidos")
    # Mezcla explícita de imágenes completas y tiles, sin duplicar clase 4.
    full = sorted((dev/"images/train_dev").glob("*.jpg"))
    paths = [str(p) for p in full] + [str(target/"images/train"/(r["tile"]+".jpg")) for r in manifest]
    (stage/"train_mixed.txt").write_text("\n".join(paths), encoding="utf-8")
    (stage/"mixed.yaml").write_text(yaml.safe_dump(dict(
        path=str(target), train="train_mixed.txt", val=str(dev/"images/val_dev"), names=CLASES)), encoding="utf-8")
    (stage/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (stage/"summary.json").write_text(json.dumps(dict(tiles=len(manifest),
        rejected_partial=rejected, sources=len(records), seed=args.seed, size=args.size,
        overlap=args.overlap, min_visible=args.visible, background=args.background), indent=2), encoding="utf-8")
    stage.rename(target)
    print(f"Guardados {len(manifest)} tiles en {target}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["split", "tiles"])
    parser.add_argument("--data-root", type=Path, default=ROOT/"spacelab_dataset")
    parser.add_argument("--development", type=Path, default=ROOT/"derived/development")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--visible", type=float, default=0.7)
    parser.add_argument("--background", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--manifest", type=Path, help="Reutilizar la partición del otro equipo")
    args = parser.parse_args()
    if not 0 < args.visible <= 1 or not 0 <= args.background <= 1 or args.workers < 1:
        parser.error("Parámetros fuera de rango")
    if args.command == "split":
        preparar(args)
    else:
        cortar(args)
