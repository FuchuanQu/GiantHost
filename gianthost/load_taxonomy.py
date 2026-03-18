import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Union

Node = Tuple[int, str]
Lineage = List[Node]


def load_taxonomy_json(path: Union[str, Path]) -> tuple[Dict[str, int], Dict[str, Lineage]]:
    """
    从 JSON 文件读取 taxonomy 数据，并转成原来 Python 代码更方便使用的格式：
    label -> [(taxid, name), (taxid, name), ...]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pseudo_taxids = raw.get("pseudo_taxids", {})

    label_to_lineage: Dict[str, Lineage] = {
        label: [(int(node["taxid"]), node["name"]) for node in lineage]
        for label, lineage in raw["label_to_lineage"].items()
    }

    return pseudo_taxids, label_to_lineage


_TAXONOMY_JSON_PATH = Path(__file__).resolve().with_name("taxonomy_lineage.json")
PSEUDO_TAXIDS, LABEL_TO_LINEAGE = load_taxonomy_json(_TAXONOMY_JSON_PATH)


def lineage_str(label: str, lineage_map: Dict[str, Lineage] = LABEL_TO_LINEAGE) -> str:
    """
    把某个 label 的 lineage 打印成字符串。
    """
    if label not in lineage_map:
        raise KeyError(f"未知 label: {label}")
    return " > ".join(f"{name}({taxid})" for taxid, name in lineage_map[label])


def lca(
    labels: Union[str, Iterable[str]],
    lineage_map: Dict[str, Lineage] = LABEL_TO_LINEAGE,
    return_lineage: bool = False,
):
    """
    给定一个或多个 label，返回它们的最低共同祖先（LCA）。
    """
    if isinstance(labels, str):
        labels = [labels]

    labels = list(labels)
    if not labels:
        raise ValueError("`labels` 不能为空")

    unknown = [x for x in labels if x not in lineage_map]
    if unknown:
        raise KeyError(f"未知 labels: {unknown}")

    lineages = [lineage_map[label] for label in labels]
    min_len = min(len(x) for x in lineages)

    common: Lineage = []
    for i in range(min_len):
        column = [lin[i] for lin in lineages]
        taxid0, name0 = column[0]
        if all(taxid == taxid0 for taxid, _ in column[1:]):
            common.append((taxid0, name0))
        else:
            break

    if not common:
        return None

    taxid, name = common[-1]
    out = {"taxid": taxid, "name": name}
    if return_lineage:
        out["lineage"] = common
    return out


def lca_taxid(labels, lineage_map: Dict[str, Lineage] = LABEL_TO_LINEAGE) -> int:
    return lca(labels, lineage_map=lineage_map)["taxid"]


def lca_name(labels, lineage_map: Dict[str, Lineage] = LABEL_TO_LINEAGE) -> str:
    return lca(labels, lineage_map=lineage_map)["name"]


if __name__ == "__main__":
    print("PSEUDO_TAXIDS =", PSEUDO_TAXIDS)
    print()

    print(lineage_str("Actinopteri"))
    print()

    print(lca(["Mammalia", "Aves"]))
    print(lca(["Chlorophyta - Mamiellophyceae", "Chlorophyta - Trebouxiophyceae"]))
    print(lca(["Amoebozoa", "Haptophyta/Stramenopiles"]))
    print(lca(["metazoa", "Mammalia"]))