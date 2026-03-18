import argparse
import copy
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

from gianthost.load_taxonomy import lca_name
from src.evaluation.conformal.score import create_score_fn, to_logit_space
from src.features.gt_extractor import GeneContentExtractor
from src.features.gvog_extractor import GVOGOneHotExtractor
from src.models.multi_task_mlp import MultiTaskMLPModel


LOGGER = logging.getLogger("release")


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass
class BlastConfig:
    evalue: float = 1e-10
    pident: float = 95.0
    qcov: float = 70.0


@dataclass
class StrictnessConfig:
    base_alpha: float
    strictness: str
    factors: Dict[str, float]

    def alpha(self) -> float:
        factor = float(self.factors.get(self.strictness, 1.0))
        return min(max(self.base_alpha * factor, 1e-6), 0.95)


class ReleasePredictor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).resolve().parents[1]
        self.db_path = Path(args.db).resolve()
        self.model_dir = (self.db_path / args.model_subdir).resolve()
        self.midfolder = Path(args.midfolder).resolve()
        self.feature_dir = self.midfolder / "features"
        self.feature_dir.mkdir(parents=True, exist_ok=True)

        self.inference_cfg = self._load_json(self.model_dir / "inference_config.json")
        self.base_alpha = float(args.alpha) if args.alpha is not None else float(
            self.inference_cfg.get("conformal", {}).get("alpha", 0.1)
        )
        strictness_cfg = self.inference_cfg.get("strictness", {})
        self.strict_cfg = StrictnessConfig(
            self.base_alpha,
            args.strictness,
            {
                "normal": strictness_cfg.get("normal_factor", 1.0),
                "strict": strictness_cfg.get("strict_factor", 1.5),
                "very_strict": strictness_cfg.get("very_strict_factor", 2.0),
            },
        )

        self.blast_cfg = BlastConfig(
            evalue=float(self.inference_cfg.get("blast", {}).get("evalue", 1e-10)),
            pident=float(self.inference_cfg.get("blast", {}).get("pident", 95.0)),
            qcov=float(self.inference_cfg.get("blast", {}).get("qcov", 70.0)),
        )

        self.nested_bundle = self._load_nested_bundle_optional()
        self.nested_models = self._load_nested_models() if self.nested_bundle is not None else []
        self.model = None if self.nested_models else self._load_model()
        self.taxonomy_df = self._load_taxonomy()
        self.temperature_cfg = self._load_temperature_config()

    def _load_nested_bundle_optional(self) -> Optional[dict]:
        bundle_path = self.model_dir / "nested_kfold_bundle.pt"
        if not bundle_path.exists():
            return None
        LOGGER.info("Loading nested K-fold bundle from %s", bundle_path)
        return torch.load(bundle_path, map_location=self.args.device, weights_only=False)

    def _build_runtime_cfg(self) -> dict:
        bundle_runtime = (self.nested_bundle or {}).get("runtime_cfg", {}) if hasattr(self, "nested_bundle") else {}
        bundle_model = bundle_runtime.get("model", {}) if isinstance(bundle_runtime, dict) else {}
        bundle_mt = bundle_runtime.get("multi_task", {}) if isinstance(bundle_runtime, dict) else {}

        model_params = dict(self.inference_cfg.get("model", {}).get("params", {}))
        model_params.update(dict(bundle_model.get("params", {})))

        multi_task_cfg = self.inference_cfg.get("multi_task", {})
        if bundle_mt:
            multi_task_cfg = bundle_mt

        return {
            "general": {
                "device": self.args.device,
                "threads": self.args.threads,
            },
            "model": {
                "name": bundle_model.get("name", self.inference_cfg.get("model", {}).get("name", "MultiTaskMLP")),
                "params": {
                    **model_params,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "num_epochs": 1,
                    "batch_size": 128,
                },
            },
            "multi_task": multi_task_cfg,
        }

    @staticmethod
    def _fill_missing_dims_from_state(runtime_cfg: dict, model_state_dict: dict, model_meta: Optional[dict] = None) -> dict:
        params = runtime_cfg.setdefault("model", {}).setdefault("params", {})
        multi_task = runtime_cfg.setdefault("multi_task", {})
        tasks = multi_task.setdefault("tasks", {})
        host_cfg = tasks.setdefault("host_classification", {})
        stage1_cfg = host_cfg.setdefault("stage_1", {})

        if params.get("input_dim1") is None:
            for key in ["backbone.tower_dense.0.weight", "backbone.tower_dense.0.bias"]:
                if key in model_state_dict and key.endswith("weight"):
                    params["input_dim1"] = int(model_state_dict[key].shape[1])
                    break

        if params.get("input_dim2") is None:
            for key in ["backbone.tower_sparse.0.weight", "backbone.tower_sparse.0.bias"]:
                if key in model_state_dict and key.endswith("weight"):
                    params["input_dim2"] = int(model_state_dict[key].shape[1])
                    break

        if params.get("hidden_dim1") is None and "backbone.tower_dense.0.weight" in model_state_dict:
            params["hidden_dim1"] = int(model_state_dict["backbone.tower_dense.0.weight"].shape[0])
        if params.get("hidden_dim2") is None and "backbone.tower_sparse.0.weight" in model_state_dict:
            params["hidden_dim2"] = int(model_state_dict["backbone.tower_sparse.0.weight"].shape[0])
        if params.get("hidden_dim3") is None and "backbone.tower_dense.8.weight" in model_state_dict:
            params["hidden_dim3"] = int(model_state_dict["backbone.tower_dense.8.weight"].shape[0])

        # stage_1 classes / host groups
        stage1_classes = list(stage1_cfg.get("classes", []) or [])
        if model_meta is not None:
            meta_classes = model_meta.get("stage_1_classes")
            if meta_classes:
                stage1_classes = [str(x) for x in meta_classes]
        if not stage1_classes:
            # infer from expert module names in state_dict
            expert_names = set()
            prefix = "host_moe_classifier.experts."
            for key in model_state_dict.keys():
                if key.startswith(prefix):
                    rest = key[len(prefix):]
                    name = rest.split(".", 1)[0]
                    if name:
                        expert_names.add(name)
            if expert_names:
                stage1_classes = sorted(expert_names)
        if stage1_classes:
            stage1_cfg["classes"] = stage1_classes

        if params.get("num_host_groups") is None:
            router_key = "host_moe_classifier.router.classifier.4.weight"
            if router_key in model_state_dict:
                params["num_host_groups"] = int(model_state_dict[router_key].shape[0])
            elif stage1_classes:
                params["num_host_groups"] = int(len(stage1_classes))

        # num_classes_per_group
        ncp = params.get("num_classes_per_group")
        if not isinstance(ncp, dict) or len(ncp) == 0:
            inferred: Dict[str, int] = {}
            for group_name in stage1_classes:
                key = f"host_moe_classifier.experts.{group_name}.classifier.4.weight"
                if key in model_state_dict:
                    inferred[group_name] = int(model_state_dict[key].shape[0])
            if inferred:
                params["num_classes_per_group"] = inferred

        # virus taxonomy classes
        if params.get("num_virus_orders") is None:
            virus_key = "virus_taxonomy_head.classifier.4.weight"
            if virus_key in model_state_dict:
                params["num_virus_orders"] = int(model_state_dict[virus_key].shape[0])
            elif model_meta is not None and model_meta.get("virus_order_encoder") is not None:
                enc = model_meta.get("virus_order_encoder")
                if hasattr(enc, "classes_"):
                    params["num_virus_orders"] = int(len(enc.classes_))

        if params.get("num_virus_orders") is None:
            params["num_virus_orders"] = 0

        return runtime_cfg

    def _load_nested_models(self) -> List[MultiTaskMLPModel]:
        if self.nested_bundle is None:
            return []
        models: List[MultiTaskMLPModel] = []
        runtime_cfg = self._build_runtime_cfg()
        for fold_entry in self.nested_bundle.get("folds", []):
            m_state = fold_entry.get("model", {})
            if not m_state or "model_state_dict" not in m_state:
                continue
            fold_cfg = self._fill_missing_dims_from_state(
                copy.deepcopy(runtime_cfg),
                m_state["model_state_dict"],
                model_meta=m_state,
            )
            model = MultiTaskMLPModel(fold_cfg)
            model.model.load_state_dict(m_state["model_state_dict"])
            model.virus_order_encoder = m_state.get("virus_order_encoder")
            model.host_group_1_encoder = m_state.get("host_group_1_encoder")
            model.host_group_2_encoders = m_state.get("host_group_2_encoders") or {}
            model.unified_host_g2_encoder = m_state.get("unified_host_g2_encoder")
            if m_state.get("unified_host_g2_classes") is not None:
                model._unified_host_g2_classes = m_state.get("unified_host_g2_classes")
            if m_state.get("class_to_unified_idx") is not None:
                model._class_to_unified_idx = m_state.get("class_to_unified_idx")
            models.append(model)
        LOGGER.info("Loaded %d nested fold models for release prediction", len(models))
        return models

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model(self) -> MultiTaskMLPModel:
        runtime_cfg = self._build_runtime_cfg()

        ckpt_path = self._resolve_final_model_path()
        LOGGER.info("Loading model from %s", ckpt_path)
        model = MultiTaskMLPModel(runtime_cfg)
        model.load(str(ckpt_path))
        return model

    def _resolve_final_model_path(self) -> Path:
        if self.args.model_file:
            return Path(self.args.model_file).resolve()
        candidates = sorted(self.model_dir.glob("*_final.pt"))
        if not candidates:
            raise FileNotFoundError(f"No *_final.pt found in {self.model_dir}")
        return candidates[0]

    def _load_taxonomy(self) -> pd.DataFrame:
        tax_path = self.db_path / "taxonomy.csv"
        df = pd.read_csv(tax_path)
        # Normalize key names
        ref_col = "refseq id" if "refseq id" in df.columns else "refseq_id"
        keep_cols = [c for c in [ref_col, "virus_order", "host_group_0", "host_group_2"] if c in df.columns]
        if ref_col not in keep_cols:
            raise ValueError("taxonomy.csv is missing refseq id column")

        df = df[keep_cols].copy()
        df.rename(columns={ref_col: "refseq_id", "host_group_0": "host_group_1"}, inplace=True)

        def _mode_or_first(x: pd.Series):
            x = x.dropna().astype(str)
            if x.empty:
                return np.nan
            m = x.mode()
            return m.iloc[0] if not m.empty else x.iloc[0]

        agg = {
            col: _mode_or_first for col in ["virus_order", "host_group_1", "host_group_2"] if col in df.columns
        }
        return df.groupby("refseq_id", as_index=False).agg(agg)

    def _load_temperature_config(self) -> Dict[str, float]:
        """Load optional temperature scaling values for each task.

        Expected format in inference_config.json:
        {
          "calibration": {
            "temperature_scaling": {
              "enable": true,
              "default": 1.0,
              "virus_order": 1.2,
              "host_group_1": 1.1,
              "host_group_2": 1.3
            }
          }
        }
        """
        cfg = self.inference_cfg.get("calibration", {}).get("temperature_scaling", {})
        if not cfg or not bool(cfg.get("enable", False)):
            return {
                "enabled": False,
                "default": 1.0,
                "virus_order": 1.0,
                "host_group_1": 1.0,
                "host_group_2": 1.0,
            }

        default_t = float(cfg.get("default", 1.0))
        return {
            "enabled": True,
            "default": max(default_t, 1e-6),
            "virus_order": max(float(cfg.get("virus_order", default_t)), 1e-6),
            "host_group_1": max(float(cfg.get("host_group_1", default_t)), 1e-6),
            "host_group_2": max(float(cfg.get("host_group_2", default_t)), 1e-6),
        }

    def _get_temperature(self, task_name: str) -> float:
        if not self.temperature_cfg.get("enabled", False):
            return 1.0
        return float(self.temperature_cfg.get(task_name, self.temperature_cfg.get("default", 1.0)))

    def _softmax_with_temperature(self, logits: np.ndarray, task_name: str) -> np.ndarray:
        t = self._get_temperature(task_name)
        return _softmax_numpy(logits / t)

    def run(self) -> pd.DataFrame:
        filtered_fasta, sample_ids = self._prepare_input_fasta()
        if not sample_ids:
            raise ValueError("No valid sequences after filtering")

        direct_assign = self._blast_and_direct_assign(filtered_fasta)

        model_ids = [sid for sid in sample_ids if sid not in direct_assign]
        model_rows = pd.DataFrame()
        if model_ids:
            features = self._extract_features(filtered_fasta, model_ids)
            model_rows = self._predict_model(model_ids, features)

        direct_rows = self._format_direct_assign_rows(direct_assign)
        result = pd.concat([direct_rows, model_rows], ignore_index=True)
        result = result.set_index("sample_id").reindex(sample_ids).reset_index()
        return result

    def _prepare_input_fasta(self) -> Tuple[Path, List[str]]:
        out_fasta = self.midfolder / "filtered_input.fasta"
        out_fasta.parent.mkdir(parents=True, exist_ok=True)

        input_type = self.args.input_type
        input_path = Path(self.args.input).resolve()

        records = []
        if input_type == "contig":
            for rec in SeqIO.parse(str(input_path), "fasta"):
                if len(rec.seq) >= self.args.min_length:
                    records.append(rec)
        else:
            mag_files = sorted(
                [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in {".fa", ".fasta", ".fna"}]
            )
            for fp in mag_files:
                seqs = [str(r.seq) for r in SeqIO.parse(str(fp), "fasta")]
                if not seqs:
                    continue
                merged = "NNNNN".join(seqs)
                rid = re.sub(r"[^A-Za-z0-9_.-]", "_", fp.stem)
                from Bio.Seq import Seq
                from Bio.SeqRecord import SeqRecord

                records.append(SeqRecord(Seq(merged), id=rid, description=""))

        SeqIO.write(records, str(out_fasta), "fasta")
        LOGGER.info("Input prepared: %d sequences -> %s", len(records), out_fasta)
        return out_fasta, [r.id for r in records]

    def _blast_and_direct_assign(self, query_fasta: Path) -> Dict[str, Dict[str, str]]:
        db_fasta = self.db_path / "ncldv_complete.fasta"
        db_prefix = self.midfolder / "blastdb" / "ncldv_complete"
        db_prefix.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_blast_db(db_fasta, db_prefix)

        out_tsv = self.midfolder / "blast.tsv"
        cmd = [
            "blastn",
            "-query", str(query_fasta),
            "-db", str(db_prefix),
            "-evalue", str(self.blast_cfg.evalue),
            "-outfmt", "6 qseqid sseqid pident qcovs evalue bitscore",
            "-num_threads", str(self.args.threads),
            "-max_target_seqs", "5",
            "-out", str(out_tsv),
        ]
        LOGGER.info("Running BLASTN ...")
        subprocess.run(cmd, check=True)

        if not out_tsv.exists() or out_tsv.stat().st_size == 0:
            return {}

        blast = pd.read_csv(
            out_tsv,
            sep="\t",
            header=None,
            names=["qseqid", "sseqid", "pident", "qcovs", "evalue", "bitscore"],
        )
        blast = blast[
            (blast["evalue"] < self.blast_cfg.evalue)
            & (blast["pident"] > self.blast_cfg.pident)
            & (blast["qcovs"] > self.blast_cfg.qcov)
        ]
        if blast.empty:
            return {}

        blast = blast.sort_values(["qseqid", "bitscore"], ascending=[True, False]).drop_duplicates("qseqid")
        blast["refseq_id"] = blast["sseqid"].astype(str).str.split().str[0]

        merged = blast.merge(self.taxonomy_df, on="refseq_id", how="left")
        direct: Dict[str, Dict[str, str]] = {}
        for _, row in merged.iterrows():
            direct[str(row["qseqid"])] = {
                "virus_order": row.get("virus_order", np.nan),
                "host_group_1": row.get("host_group_1", np.nan),
                "host_group_2": row.get("host_group_2", np.nan),
                "refseq_id": row.get("refseq_id", np.nan),
            }
        LOGGER.info("Direct-assigned by BLASTN: %d", len(direct))
        return direct

    @staticmethod
    def _ensure_blast_db(db_fasta: Path, db_prefix: Path) -> None:
        needed = [db_prefix.with_suffix(s) for s in [".nhr", ".nin", ".nsq"]]
        if all(p.exists() for p in needed):
            return
        cmd = [
            "makeblastdb",
            "-in", str(db_fasta),
            "-dbtype", "nucl",
            "-out", str(db_prefix),
        ]
        LOGGER.info("Building BLAST database ...")
        subprocess.run(cmd, check=True)

    def _extract_features(self, fasta_path: Path, sample_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        source_suffix = self.inference_cfg.get("features", {}).get("source_suffix", {})
        feat_cfg = {
            "general": {"threads": self.args.threads},
            "paths": {
                "gvog_hmm": str(self.db_path / "gvog.complete.hmm"),
                "source_suffix": {
                    "cds": source_suffix.get("cds", "cds.fna"),
                    "protein": source_suffix.get("protein", "prot.faa"),
                },
            },
            "features": {
                "sources": {
                    "kmer": {"k": 4}
                }
            }
        }

        trait_extractor = GeneContentExtractor(feat_cfg)
        gvog_extractor = GVOGOneHotExtractor(feat_cfg)

        trait_path = self.feature_dir / "trait.parquet"
        gvog_path = self.feature_dir / "gvog.parquet"

        trait_extractor.generate(str(fasta_path), str(trait_path)) if not trait_path.exists() else None
        gvog_extractor.generate(str(fasta_path), str(gvog_path)) if not gvog_path.exists() else None

        trait_df = trait_extractor.load(str(trait_path))
        gvog_df = gvog_extractor.load(str(gvog_path))

        trait_df = trait_df.loc[list(sample_ids)]
        gvog_df = gvog_df.loc[list(sample_ids)]

        return {
            "dense": trait_df.values.astype(np.float32),
            "sparse": gvog_df.values.astype(np.float32),
        }

    def _predict_model(self, sample_ids: Sequence[str], features: Dict[str, np.ndarray]) -> pd.DataFrame:
        if self.nested_models:
            return self._predict_model_nested(sample_ids, features)

        out = self.model.predict_proba(features)

        g1_logits = out["host_group_1"]
        g2_probs = out["host_group_2_aggregated"]
        virus_logits = out.get("virus_taxonomy")
        if virus_logits is None:
            virus_logits = out.get("virus_order")

        g1_idx = np.argmax(g1_logits, axis=1)
        g2_idx = np.argmax(g2_probs, axis=1)
        g1_labels = self.model.host_group_1_encoder.inverse_transform(g1_idx)
        g2_classes = self.model.get_unified_host_g2_classes()
        g2_labels = np.array([g2_classes[i] for i in g2_idx], dtype=object)

        virus_labels = np.array([np.nan] * len(sample_ids), dtype=object)
        if virus_logits is not None and self.model.virus_order_encoder is not None:
            v_idx = np.argmax(virus_logits, axis=1)
            virus_labels = self.model.virus_order_encoder.inverse_transform(v_idx)

        # 输出原始 logits 的 softmax（可选 temperature scaling）
        g1_softmax = self._softmax_with_temperature(g1_logits, "host_group_1")
        g1_softmax_max = np.max(g1_softmax, axis=1)

        # host_group_2 聚合输出为概率；在 softmax 前做温度缩放可通过 log-prob 近似恢复 logits
        g2_logits = to_logit_space(g2_probs, eps=float(self.inference_cfg.get("conformal", {}).get("eps", 1e-12)))
        g2_softmax = self._softmax_with_temperature(g2_logits, "host_group_2")
        g2_softmax_max = np.max(g2_softmax, axis=1)

        virus_softmax_max = np.full(len(sample_ids), np.nan, dtype=float)
        if virus_logits is not None:
            virus_softmax = self._softmax_with_temperature(virus_logits, "virus_order")
            virus_softmax_max = np.max(virus_softmax, axis=1)

        conf_scores, conf_thr, accepted = self._conformal_acceptance(g2_probs, g2_idx)

        g1_labels = np.where(accepted, g1_labels, "unclassified")
        g2_labels = np.where(accepted, g2_labels, "unclassified")

        return pd.DataFrame(
            {
                "sample_id": list(sample_ids),
                "virus_order": virus_labels,
                "host_group_1": g1_labels,
                "host_group_2": g2_labels,
                "host_group_2_lca": g2_labels,
                "prediction_source": "model",
                "matched_refseq_id": np.nan,
                "virus_order_softmax": virus_softmax_max,
                "host_group_1_softmax": g1_softmax_max,
                "host_group_2_softmax": g2_softmax_max,
                "conformal_score": conf_scores,
                "conformal_threshold": conf_thr,
                "accepted": accepted,
                "strictness": self.args.strictness,
                "alpha": self.strict_cfg.alpha(),
            }
        )

    def _predict_model_nested(self, sample_ids: Sequence[str], features: Dict[str, np.ndarray]) -> pd.DataFrame:
        bundle = self.nested_bundle or {}
        conformal_cfg = bundle.get("conformal", {})
        primary_classes = bundle.get("primary_classes", [])
        classes_in_level1 = bundle.get("classes_in_level1", {})
        alpha_total = self.strict_cfg.alpha()
        alpha1 = min(max(alpha_total, 1e-8), 0.999999)
        alpha2 = min(max(alpha_total, 1e-8), 0.999999)
        LOGGER.info(
            "Nested conformal thresholds: alpha_total=%.6f alpha1=%.6f alpha2=%.6f",
            alpha_total,
            alpha1,
            alpha2,
        )
        score_name = str(conformal_cfg.get("score", "aps"))

        fold_probs = []
        fold_g1_probs = []
        s1_list = []
        s2_list = []
        for model, fold_entry in zip(self.nested_models, bundle.get("folds", [])):
            out = model.predict_proba(features)
            g2_probs = out.get("host_group_2_aggregated")
            if g2_probs is None:
                continue

            g1_logits = out.get("host_group_1")
            if g1_logits is not None:
                g1_probs = self._softmax_with_temperature(np.asarray(g1_logits), "host_group_1")
                fold_g1_probs.append(g1_probs)

            g2_classes = model.get_unified_host_g2_classes()
            aligned = _align_probs_to_classes(g2_probs, g2_classes, primary_classes)
            fold_probs.append(_row_normalize(aligned))
            scores = fold_entry.get("scores", {})
            s1_list.append({k: np.asarray(v) for k, v in (scores.get("s1", {}) or {}).items()})
            s2_list.append({k: np.asarray(v) for k, v in (scores.get("s2", {}) or {}).items()})

        if not fold_probs:
            raise RuntimeError("Nested bundle is present but no fold predictions could be computed.")

        gamma1, gamma2_by_branch, gamma2_flat = _predict_sets_from_fold_banks(
            fold_probs=fold_probs,
            s1_list=s1_list,
            s2_list=s2_list,
            primary_classes=primary_classes,
            classes_in_level1=classes_in_level1,
            alpha1=alpha1,
            alpha2=alpha2,
            score_name=score_name,
            raps_lambda=float(conformal_cfg.get("raps_lambda", 0.01)),
            raps_k_reg=int(conformal_cfg.get("raps_k_reg", 1)),
        )

        avg_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
        g2_idx = np.argmax(avg_probs, axis=1)
        point_g2 = np.array([primary_classes[i] for i in g2_idx], dtype=object)
        level2_to_level1 = bundle.get("level2_to_level1", {})

        if fold_g1_probs:
            avg_g1_probs = np.mean(np.stack(fold_g1_probs, axis=0), axis=0)
            g1_softmax_max = np.max(avg_g1_probs, axis=1)
            g1_idx = np.argmax(avg_g1_probs, axis=1)
            g1_encoder = None
            for m in self.nested_models:
                if getattr(m, "host_group_1_encoder", None) is not None:
                    g1_encoder = m.host_group_1_encoder
                    break
            if g1_encoder is not None:
                point_g1 = g1_encoder.inverse_transform(g1_idx)
            else:
                # fallback: map indices to configured class order if encoder is unavailable
                stage1_classes = list(
                    self.inference_cfg.get("multi_task", {})
                    .get("tasks", {})
                    .get("host_classification", {})
                    .get("stage_1", {})
                    .get("classes", [])
                )
                if len(stage1_classes) == avg_g1_probs.shape[1]:
                    point_g1 = np.array([stage1_classes[i] for i in g1_idx], dtype=object)
                else:
                    point_g1 = np.array(["unclassified"] * len(sample_ids), dtype=object)
        else:
            # fallback to aggregation from level2 if host_group_1 head outputs are missing
            point_g1 = np.array([level2_to_level1.get(lbl, "unclassified") for lbl in point_g2], dtype=object)
            g1_prob = np.zeros((avg_probs.shape[0], max(len(classes_in_level1), 1)), dtype=float)
            for j, (_, branch_classes) in enumerate(classes_in_level1.items()):
                idxs = [primary_classes.index(c) for c in branch_classes if c in primary_classes]
                if idxs:
                    g1_prob[:, j] = np.sum(avg_probs[:, idxs], axis=1)
            g1_softmax_max = np.max(g1_prob, axis=1) if g1_prob.shape[1] > 0 else np.full(avg_probs.shape[0], np.nan)
        g2_softmax_max = np.max(avg_probs, axis=1)

        accepted = np.array([len(s) > 0 for s in gamma2_flat], dtype=bool)
        out_g2 = np.array([s[0] if len(s) == 1 else "unclassified" for s in gamma2_flat], dtype=object)
        out_g1 = np.array([s[0] if len(s) == 1 else "unclassified" for s in gamma1], dtype=object)
        out_g2_lca = np.array([self._lca_from_label_set(s) for s in gamma2_flat], dtype=object)

        return pd.DataFrame(
            {
                "sample_id": list(sample_ids),
                "virus_order": np.array([np.nan] * len(sample_ids), dtype=object),
                "host_group_1": out_g1,
                "host_group_2": out_g2,
            "host_group_2_lca": out_g2_lca,
                "host_group_1_point": point_g1,
                "host_group_2_point": point_g2,
                "host_group_1_set": [json.dumps(s, ensure_ascii=False) for s in gamma1],
                "host_group_2_set": [json.dumps(s, ensure_ascii=False) for s in gamma2_flat],
                "host_group_2_set_by_branch": [json.dumps(v, ensure_ascii=False) for v in gamma2_by_branch],
                "prediction_source": "model_nested_conformal",
                "matched_refseq_id": np.nan,
                "virus_order_softmax": np.nan,
                "host_group_1_softmax": g1_softmax_max,
                "host_group_2_softmax": g2_softmax_max,
                "conformal_score": np.nan,
                "conformal_threshold": np.nan,
                "accepted": accepted,
                "strictness": self.args.strictness,
                "alpha": alpha_total,
            }
        )

    def _conformal_acceptance(
        self,
        host_group_2_probs: np.ndarray,
        pred_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        npz_path = self.model_dir / "conformal_scores.npz"
        if not npz_path.exists():
            n = host_group_2_probs.shape[0]
            return np.full(n, np.nan), np.full(n, np.nan), np.ones(n, dtype=bool)
        
        scores = np.load(npz_path)
        score_arrays = [scores[k] for k in scores.files if "host_group_2" in k]
        if not score_arrays:
            n = host_group_2_probs.shape[0]
            return np.full(n, np.nan), np.full(n, np.nan), np.ones(n, dtype=bool)

        cal_scores = np.concatenate([a for a in score_arrays if a.size > 0])
        if cal_scores.size == 0:
            n = host_group_2_probs.shape[0]
            return np.full(n, np.nan), np.full(n, np.nan), np.ones(n, dtype=bool)

        conformal_alpha = self.strict_cfg.alpha()
        q = np.quantile(cal_scores, 1.0 - conformal_alpha, method="higher")

        score_cfg = {
            "calibration": {
                "conformal": self.inference_cfg.get("conformal", {})
            }
        }
        score_fn = create_score_fn(score_cfg)

        logits = to_logit_space(host_group_2_probs, eps=float(self.inference_cfg.get("conformal", {}).get("eps", 1e-12)))
        score_matrix = score_fn.compute_matrix(logits)
        # print(score_matrix)
        pred_scores = score_matrix[np.arange(len(pred_idx)), pred_idx]
        accepted = pred_scores <= q
        # print(pred_scores, q, accepted)
        return pred_scores, np.full(len(pred_idx), q, dtype=float), accepted

    @staticmethod
    def _lca_from_label_set(labels: Sequence[str]) -> str:
        valid_labels = [str(x) for x in labels if pd.notna(x) and str(x) and str(x) != "unclassified"]
        if not valid_labels:
            return "unclassified"
        if len(valid_labels) == 1:
            return valid_labels[0]
        try:
            return lca_name(valid_labels)
        except Exception:
            return "unclassified"

    @staticmethod
    def _format_direct_assign_rows(direct_assign: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        if not direct_assign:
            return pd.DataFrame(columns=[
                "sample_id", "virus_order", "host_group_1", "host_group_2",
                "host_group_2_lca",
                "prediction_source", "matched_refseq_id",
                "virus_order_softmax", "host_group_1_softmax", "host_group_2_softmax",
                "conformal_score", "conformal_threshold", "accepted", "strictness", "alpha",
            ])
        rows = []
        for sid, info in direct_assign.items():
            rows.append(
                {
                    "sample_id": sid,
                    "virus_order": info.get("virus_order"),
                    "host_group_1": info.get("host_group_1"),
                    "host_group_2": info.get("host_group_2"),
                    "host_group_2_lca": info.get("host_group_2"),
                    "prediction_source": "blast_direct",
                    "matched_refseq_id": info.get("refseq_id"),
                    "virus_order_softmax": np.nan,
                    "host_group_1_softmax": np.nan,
                    "host_group_2_softmax": np.nan,
                    "conformal_score": np.nan,
                    "conformal_threshold": np.nan,
                    "accepted": True,
                    "strictness": "direct",
                    "alpha": np.nan,
                }
            )
        return pd.DataFrame(rows)


def _align_probs_to_classes(probs: np.ndarray, source_classes: Sequence[str], target_classes: Sequence[str]) -> np.ndarray:
    source_classes = list(source_classes)
    target_classes = list(target_classes)
    if source_classes == target_classes:
        return probs
    idx_map = {c: i for i, c in enumerate(source_classes)}
    out = np.zeros((probs.shape[0], len(target_classes)), dtype=float)
    for j, cls in enumerate(target_classes):
        i = idx_map.get(cls)
        if i is not None and i < probs.shape[1]:
            out[:, j] = probs[:, i]
    return out


def _row_normalize(x: np.ndarray) -> np.ndarray:
    s = np.sum(x, axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return x / s


def _count_ge(sorted_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_scores.size == 0:
        return np.zeros_like(values, dtype=float)
    idx = np.searchsorted(sorted_scores, values, side="left")
    return len(sorted_scores) - idx


def _aps_candidate_scores(probs: np.ndarray, candidate_idx: int) -> np.ndarray:
    n = probs.shape[0]
    order = np.argsort(-probs, axis=1, kind="mergesort")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == candidate_idx)[0][0])
    return cumsum[np.arange(n), ranks]


def _level2_candidate_scores(
    probs: np.ndarray,
    candidate_idx: int,
    score_name: str,
    raps_lambda: float,
    raps_k_reg: int,
) -> np.ndarray:
    p = _row_normalize(np.asarray(probs, dtype=float))
    py = np.clip(p[:, int(candidate_idx)], 1e-12, 1.0)
    if score_name == "aps":
        return _aps_candidate_scores(p, candidate_idx)
    if score_name == "raps":
        base = _aps_candidate_scores(p, candidate_idx)
        order = np.argsort(-p, axis=1, kind="mergesort")
        n = p.shape[0]
        rank = np.zeros(n, dtype=np.int64)
        for i in range(n):
            rank[i] = int(np.where(order[i] == int(candidate_idx))[0][0])
        return base + raps_lambda * np.maximum(0, rank - raps_k_reg)
    if score_name == "softmax_response":
        return -np.log(py)
    if score_name == "softmax_margin":
        return np.max(p, axis=1) - py
    logits = np.log(np.clip(p, 1e-12, 1.0))
    return np.max(logits, axis=1) - logits[:, int(candidate_idx)]


def _predict_sets_from_fold_banks(
    fold_probs: List[np.ndarray],
    s1_list: List[Dict[str, np.ndarray]],
    s2_list: List[Dict[str, np.ndarray]],
    primary_classes: List[str],
    classes_in_level1: Dict[str, List[str]],
    alpha1: float,
    alpha2: float,
    score_name: str,
    raps_lambda: float,
    raps_k_reg: int,
):
    n = fold_probs[0].shape[0]
    class_to_idx = {c: i for i, c in enumerate(primary_classes)}

    pvals_level1 = {y1: np.zeros(n, dtype=float) for y1 in classes_in_level1.keys()}
    for y1, branch_classes in classes_in_level1.items():
        branch_idx = [class_to_idx[c] for c in branch_classes if c in class_to_idx]
        if not branch_idx:
            continue
        numer = np.ones(n, dtype=float)
        denom = np.ones(n, dtype=float)
        n_cal = 0
        for j in range(len(fold_probs)):
            scores = np.asarray(s1_list[j].get(y1, np.array([])))
            if scores.size == 0:
                continue
            s_star = 1.0 - np.sum(fold_probs[j][:, branch_idx], axis=1)
            numer += _count_ge(scores, s_star)
            denom += float(scores.size)
            n_cal += int(scores.size)
        pvals_level1[y1] = np.zeros(n, dtype=float) if n_cal == 0 else numer / np.maximum(denom, 1.0)

    gamma1 = [[y1 for y1 in classes_in_level1.keys() if pvals_level1[y1][i] > alpha1] for i in range(n)]

    pvals_level2 = {c: np.zeros(n, dtype=float) for c in primary_classes}
    for y1, branch_classes in classes_in_level1.items():
        branch_idx = [class_to_idx[c] for c in branch_classes if c in class_to_idx]
        if not branch_idx:
            continue
        for y2 in branch_classes:
            if y2 not in class_to_idx:
                continue
            local_idx = branch_classes.index(y2)
            numer = np.ones(n, dtype=float)
            denom = np.ones(n, dtype=float)
            n_cal = 0
            for j in range(len(fold_probs)):
                scores = np.asarray(s2_list[j].get(y1, np.array([])))
                if scores.size == 0:
                    continue
                p_branch = _row_normalize(fold_probs[j][:, branch_idx])
                s_star = _level2_candidate_scores(p_branch, local_idx, score_name, raps_lambda, raps_k_reg)
                numer += _count_ge(scores, s_star)
                denom += float(scores.size)
                n_cal += int(scores.size)
            pvals_level2[y2] = np.zeros(n, dtype=float) if n_cal == 0 else numer / np.maximum(denom, 1.0)

    gamma2_by_branch = []
    gamma2_flat = []
    for i in range(n):
        by_branch = {}
        flat = []
        for y1 in gamma1[i]:
            branch_classes = classes_in_level1.get(y1, [])
            acc = [y2 for y2 in branch_classes if pvals_level2.get(y2, np.zeros(n))[i] > alpha2]
            by_branch[y1] = acc
            flat.extend(acc)
        gamma2_by_branch.append(by_branch)
        gamma2_flat.append(flat)

    return gamma1, gamma2_by_branch, gamma2_flat
    
